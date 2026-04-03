#!/usr/bin/env python3
# video_upscaler.py - Python Qt6 implementation of the Video Upscaler application

import os
import sys
import shutil
import subprocess
import tempfile
import time
import re
import shlex
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable
from functools import partial
from datetime import datetime
import traceback
import glob

# Qt6 imports - using PySide6, but could also use PyQt6 with minor adjustments
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QProgressBar, 
    QFileDialog, QMessageBox, QLabel, QLineEdit, QGroupBox, QFormLayout, QHBoxLayout
)
from PySide6.QtCore import (
    Qt, QTimer, QObject, Signal, QThread, QThreadPool, QRunnable, Slot, QUrl
)
from PySide6.QtGui import QIcon

# Constants class similar to C++ version
class Constants:
    # Process weightings
    EXTRACT_FRAMES_WEIGHT = 20
    INFERENCE_WEIGHT = 70
    MERGE_VIDEO_WEIGHT = 10
    TOTAL_WEIGHT = EXTRACT_FRAMES_WEIGHT + INFERENCE_WEIGHT + MERGE_VIDEO_WEIGHT
    
    # Default settings
    DEFAULT_REALESRGAN_PATH = "./realesrgan-ncnn-vulkan"
    DEFAULT_FRAME_RATE = "23.98"
    
    # Temp directory names
    TEMP_FRAMES_DIR = "tmp_frames"
    OUTPUT_FRAMES_DIR = "out_frames"
    
    # UI update interval in milliseconds
    UI_UPDATE_INTERVAL_MS = 100
    FRAME_COUNT_CHECK_INTERVAL_MS = 500

# Worker class to run tasks in a separate thread
class Worker(QRunnable):
    """
    Worker thread for running background tasks
    """
    class Signals(QObject):
        finished = Signal()
        error = Signal(str)
        progress = Signal(int)
        status = Signal(str)
        
    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = Worker.Signals()
        
    @Slot()
    def run(self):
        try:
            self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit()
        except Exception as e:
            traceback.print_exc()
            self.signals.error.emit(str(e))

# RAII-like TemporaryDirectory class
class TemporaryDirectory:
    def __init__(self, path: str):
        self.path = path
        self.valid = False
        
        # Create directory if it doesn't exist
        try:
            path_obj = Path(path)
            if path_obj.exists() and not path_obj.is_dir():
                raise RuntimeError(f"Path exists but is not a directory: {path}")
            
            if not path_obj.exists():
                path_obj.mkdir(parents=True, exist_ok=True)
            
            self.valid = True
        except Exception as e:
            raise RuntimeError(f"Failed to create directory: {path} (Error: {str(e)})")
    
    def __del__(self):
        self.cleanup()
    
    def cleanup(self):
        if self.valid:
            try:
                shutil.rmtree(self.path, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Failed to remove temporary directory: {self.path} (Error: {str(e)})")
            self.valid = False

# Safe command builder to prevent command injection
class CommandBuilder:
    def __init__(self, executable_name: str):
        self.executable = executable_name
        self.arguments = []
    
    def add_argument(self, arg: str) -> 'CommandBuilder':
        self.arguments.append(arg)
        return self
    
    def add_flag(self, flag: str, value: str) -> 'CommandBuilder':
        self.arguments.append(flag)
        self.arguments.append(value)
        return self
    
    def build(self) -> str:
        """Build the command as a string for printing/logging purposes"""
        command = shlex.quote(self.executable)
        
        for arg in self.arguments:
            command += ' ' + shlex.quote(arg)
        
        return command
    
    def build_args_list(self) -> List[str]:
        """Build the command as a list suitable for subprocess"""
        return [self.executable] + self.arguments

# Main application class
class VideoUpscaler(QMainWindow):
    def __init__(self):
        super(VideoUpscaler, self).__init__()
        
        # Set window properties
        self.setWindowTitle("Video Upscaler")
        self.setMinimumSize(600, 400)
        
        # Setup central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # Processing state
        self.input_files = []
        self.current_file_index = 0
        self.progress_percent = 0
        self.processing_canceled = False
        self.current_status = "Ready"
        self.last_upscale_factor = 4
        self.last_model_name = "realesrgan-x4plus"
        self.last_error = None
        
        # Frame counting state
        self.total_frame_count = 0
        self.processed_input_frames = 0
        self.processed_output_frames = 0
        self.current_process_weight = 0
        
        # Thread pool for background processing
        self.thread_pool = QThreadPool()
        
        # Create UI components
        self.create_buttons()
        self.create_progress_section()
        self.create_settings_section()
        
        # Setup progress update timer
        self.progress_timer = QTimer(self)
        self.progress_timer.timeout.connect(self.update_progress_ui)
        self.progress_timer.start(Constants.UI_UPDATE_INTERVAL_MS)
        
        # Frame counter timer (created when needed)
        self.frame_counter_timer = None
        
        # Create temporary directories (these will be recreated for each file)
        self.tmp_dir = None
        self.out_dir = None
    
    def create_buttons(self):
        select_files_button = QPushButton("Select Input Files", self)
        select_files_button.setIcon(QIcon.fromTheme("document-open"))
        select_files_button.clicked.connect(self.select_multiple_files)
        self.layout.addWidget(select_files_button)
        
        button_info = [
            ("Upscale 3D/Real-Life Videos (4x)", 4, "realesrgan-x4plus"),
            ("Upscale 2D Animations (4x)", 4, "realesr-animevideov3-x4"),
            ("Upscale 2D Animations (2x)", 2, "realesr-animevideov3-x2")
        ]
        
        for text, upscale_factor, model_name in button_info:
            button = QPushButton(text, self)
            button.clicked.connect(
                partial(self.start_processing, upscale_factor, model_name)
            )
            self.layout.addWidget(button)
        
        # Cancel button
        cancel_button = QPushButton("Cancel Processing", self)
        cancel_button.setIcon(QIcon.fromTheme("process-stop"))
        cancel_button.clicked.connect(self.cancel_processing)
        self.layout.addWidget(cancel_button)
    
    def create_progress_section(self):
        # Status label
        self.status_label = QLabel("Ready", self)
        self.layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)
    
    def create_settings_section(self):
        settings_group = QGroupBox("Settings", self)
        settings_layout = QFormLayout(settings_group)
        
        # RealESRGAN path setting
        self.realesrgan_path_edit = QLineEdit(Constants.DEFAULT_REALESRGAN_PATH, self)
        browse_button = QPushButton("Browse", self)
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.realesrgan_path_edit)
        path_layout.addWidget(browse_button)
        
        browse_button.clicked.connect(self.browse_realesrgan_path)
        
        settings_layout.addRow("RealESRGAN Path:", path_layout)
        self.layout.addWidget(settings_group)
    
    def browse_realesrgan_path(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select RealESRGAN Executable")
        if path:
            self.realesrgan_path_edit.setText(path)
    
    def select_multiple_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Input Video Files", "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)"
        )
        
        if files:
            self.input_files = files
            self.show_message(f"Selected {len(files)} files for processing")
    
    def start_processing(self, upscale_factor, model_name):
        if not self.input_files:
            self.show_message("Please select input files first", True)
            return
        
        if self.thread_pool.activeThreadCount() > 0:
            self.show_message("Processing already in progress", True)
            return
        
        self.current_file_index = 0
        self.last_upscale_factor = upscale_factor
        self.last_model_name = model_name
        self.process_next_file(upscale_factor, model_name)
    
    def process_next_file(self, upscale_factor, model_name):
        if self.current_file_index >= len(self.input_files):
            self.show_message("All files processed successfully")
            self.update_status_message("Ready")
            self.progress_percent = 0
            self.progress_bar.setValue(0)
            return
        
        # Reset progress and cancellation flag
        self.progress_percent = 0
        self.processing_canceled = False
        self.last_error = None
        
        current_file = self.input_files[self.current_file_index]
        filename = Path(current_file).name
        self.update_status_message(f"Processing: {filename}")
        
        # Create worker to run the processing in a background thread
        worker = Worker(
            self.perform_video_upscaling, 
            current_file, upscale_factor, model_name
        )
        
        worker.signals.finished.connect(self.on_processing_finished)
        worker.signals.error.connect(self.on_processing_error)
        
        # Start the worker
        self.thread_pool.start(worker)
    
    def on_processing_finished(self):
        # Check if processing was canceled
        if self.processing_canceled:
            self.show_message("Processing canceled by user")
            self.update_status_message("Ready")
            self.progress_percent = 0
            self.progress_bar.setValue(0)
            return
        
        # Move to next file
        self.current_file_index += 1
        self.process_next_file(self.last_upscale_factor, self.last_model_name)
    
    def on_processing_error(self, error_message):
        self.show_message(
            f"Error processing file {self.current_file_index + 1}: {error_message}", 
            True
        )
        self.update_status_message("Error")
    
    def cancel_processing(self):
        if self.thread_pool.activeThreadCount() > 0:
            self.processing_canceled = True
            self.update_status_message("Canceling...")
            self.show_message("Canceling current process. Please wait...")
    
    def perform_video_upscaling(self, input_file, upscale_factor, model_name):
        start_time = time.time()
        
        try:
            # Create temporary directories
            self.tmp_dir = TemporaryDirectory(Constants.TEMP_FRAMES_DIR)
            self.out_dir = TemporaryDirectory(Constants.OUTPUT_FRAMES_DIR)
            
            # First, get estimated frame count from the video
            estimated_frames = self.get_frame_count(input_file)
            self.total_frame_count = estimated_frames if estimated_frames > 0 else 1000
            self.processed_input_frames = 0
            self.processed_output_frames = 0
            
            # Start frame counting timer
            self.frame_counter_timer = QTimer()
            self.frame_counter_timer.timeout.connect(self.update_frame_counts)
            self.frame_counter_timer.start(Constants.FRAME_COUNT_CHECK_INTERVAL_MS)
            
            # Set initial process weight
            self.current_process_weight = Constants.EXTRACT_FRAMES_WEIGHT
            
            # Extract frames
            self.update_status_message("Extracting frames...")
            self.extract_frames(input_file, self.tmp_dir.path)
            if self.processing_canceled:
                return
            
            # Update process weight for inference
            self.current_process_weight = Constants.INFERENCE_WEIGHT
            
            # Run inference
            self.update_status_message("Running inference...")
            self.run_inference(self.tmp_dir.path, self.out_dir.path, model_name, upscale_factor)
            if self.processing_canceled:
                return
            
            # Update for final stage
            self.current_process_weight = Constants.MERGE_VIDEO_WEIGHT
            
            # Get frame rate
            self.update_status_message("Getting frame rate...")
            frame_rate = self.get_frame_rate(input_file)
            if self.processing_canceled:
                return
            
            # Merge video
            self.update_status_message("Merging video...")
            self.merge_video(frame_rate, self.out_dir.path, input_file)
            
            if not self.processing_canceled:
                # Calculate time taken
                elapsed_time = time.time() - start_time
                hours, remainder = divmod(int(elapsed_time), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                completion_msg = f"Upscaling completed in {hours}h {minutes}m {seconds}s"
                self.update_status_message(completion_msg)
                print(completion_msg)
            
        except Exception as e:
            print(f"Error during upscaling: {str(e)}")
            raise
        finally:
            # Clean up temporary directories
            if self.frame_counter_timer:
                self.frame_counter_timer.stop()
                self.frame_counter_timer = None
            
            # Clean up temp directories
            if self.tmp_dir:
                self.tmp_dir.cleanup()
                self.tmp_dir = None
            
            if self.out_dir:
                self.out_dir.cleanup()
                self.out_dir = None
    
    def update_frame_counts(self):
        if self.processing_canceled:
            if self.frame_counter_timer:
                self.frame_counter_timer.stop()
            return
        
        # Check current process stage and count appropriate frames
        if self.current_process_weight == Constants.EXTRACT_FRAMES_WEIGHT:
            current_input_frames = self.count_frames_in_directory(self.tmp_dir.path, ".png")
            self.processed_input_frames = current_input_frames
        elif self.current_process_weight == Constants.INFERENCE_WEIGHT:
            current_output_frames = self.count_frames_in_directory(self.out_dir.path, ".jpg")
            self.processed_output_frames = current_output_frames
        
        # Update progress based on current frame counts
        self.calculate_frame_based_progress()
        
        # Stop the timer when processing is complete or canceled
        if self.progress_percent >= 100 or self.processing_canceled:
            if self.frame_counter_timer:
                self.frame_counter_timer.stop()
    
    def count_frames_in_directory(self, dir_path, extension):
        try:
            return len(glob.glob(os.path.join(dir_path, f"*{extension}")))
        except Exception as e:
            print(f"Error counting frames: {str(e)}")
            return 0
    
    def get_frame_count(self, input_file):
        cmd = CommandBuilder("ffprobe")
        cmd.add_argument("-v").add_argument("error") \
           .add_argument("-count_frames") \
           .add_argument("-select_streams").add_argument("v:0") \
           .add_argument("-show_entries").add_argument("stream=nb_read_frames") \
           .add_argument("-of").add_argument("csv=p=0") \
           .add_argument(input_file)
        
        ffprobe_cmd = cmd.build()
        print(f"Frame count command: {ffprobe_cmd}")
        
        try:
            result = subprocess.check_output(
                cmd.build_args_list(), 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Trim whitespace and parse
            result = result.strip()
            
            try:
                return int(result)
            except ValueError:
                print(f"Warning: Failed to parse frame count: {result}")
                return 0
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to get frame count: {e.output}")
            return 0
    
    def extract_frames(self, input_file, tmp_frames_dir):
        cmd = CommandBuilder("ffmpeg")
        cmd.add_argument("-analyzeduration").add_argument("0") \
           .add_argument("-probesize").add_argument("32") \
           .add_argument("-i").add_argument(input_file) \
           .add_argument("-qscale:v").add_argument("1") \
           .add_argument("-qmin").add_argument("1") \
           .add_argument("-qmax").add_argument("1") \
           .add_argument("-vsync").add_argument("0") \
           .add_argument("-threads").add_argument("16") \
           .add_argument(os.path.join(tmp_frames_dir, "frame%08d.png"))
        
        ffmpeg_cmd = cmd.build()
        print(f"Extract frames command: {ffmpeg_cmd}")
        
        try:
            subprocess.check_call(cmd.build_args_list())
        except subprocess.CalledProcessError as e:
            if not self.processing_canceled:
                raise RuntimeError(f"Failed to extract frames, error code: {e.returncode}")
            return
        
        # Count actual frames extracted for accurate progress
        frame_count = self.count_frames_in_directory(tmp_frames_dir, ".png")
        if frame_count > 0 and frame_count != self.total_frame_count:
            self.total_frame_count = frame_count
        self.processed_input_frames = frame_count
        
        # Calculate progress based on frames
        self.calculate_frame_based_progress()
    
    def run_inference(self, tmp_frames_dir, out_frames_dir, model_name, upscale_factor):
        cmd = CommandBuilder(self.realesrgan_path_edit.text())
        cmd.add_argument("-i").add_argument(tmp_frames_dir) \
            .add_argument("-o").add_argument(out_frames_dir) \
            .add_argument("-n").add_argument(model_name) \
            .add_argument("-s").add_argument(str(upscale_factor)) \
            .add_argument("-f").add_argument("jpg")
    
        inference_cmd = cmd.build()
        print(f"Inference command: {inference_cmd}")
    
        try:
        # Use subprocess.run for better control over output
            result = subprocess.run(
                cmd.build_args_list(),
                capture_output=True,  # Capture stdout and stderr
                text=True,            # Decode output as text
                check=True            # Automatically raise CalledProcessError on non-zero exit codes
            )
        except subprocess.CalledProcessError as e:
            if not self.processing_canceled:
            # Create a more informative error message including stderr
                error_details = e.stderr or e.stdout or "No output captured."
                error_msg = (
                    f"Failed to run inference, error code: {e.returncode}\n"
                    f"Error Output:\n{error_details}"
                )
                raise RuntimeError(error_msg)
            return
    
    # Count actual processed frames for accurate progress
        output_frame_count = self.count_frames_in_directory(out_frames_dir, ".jpg")
        self.processed_output_frames = output_frame_count
    
    # Calculate progress based on frames
        self.calculate_frame_based_progress()
    
    def get_frame_rate(self, input_file):
        cmd = CommandBuilder("ffprobe")
        cmd.add_argument("-analyzeduration").add_argument("0") \
           .add_argument("-probesize").add_argument("32") \
           .add_argument("-v").add_argument("error") \
           .add_argument("-select_streams").add_argument("v:0") \
           .add_argument("-show_entries").add_argument("stream=avg_frame_rate") \
           .add_argument("-of").add_argument("default=noprint_wrappers=1:nokey=1") \
           .add_argument(input_file)
        
        ffprobe_cmd = cmd.build()
        print(f"Get frame rate command: {ffprobe_cmd}")
        
        try:
            result = subprocess.check_output(
                cmd.build_args_list(), 
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Trim whitespace
            result = result.strip()
            
            if not result:
                print(f"Frame rate detection failed, using default: {Constants.DEFAULT_FRAME_RATE}")
                return Constants.DEFAULT_FRAME_RATE
            
            # Check if it's a fraction
            if '/' in result:
                try:
                    numerator, denominator = map(int, result.split('/'))
                    if denominator != 0:
                        frame_rate = numerator / denominator
                        return str(frame_rate)
                except ValueError as e:
                    print(f"Error parsing frame rate: {str(e)}")
                    return Constants.DEFAULT_FRAME_RATE
            
            return result
                
        except subprocess.CalledProcessError:
            return Constants.DEFAULT_FRAME_RATE
    
    def merge_video(self, frame_rate, out_frames_dir, input_file):
        input_path = Path(input_file)
        output_file = f"output_{input_path.stem}_{self.last_upscale_factor}x.mp4"
        
        # Filter argument
        filter_arg = "format=nv12|vaapi,hwupload"
        
        cmd = CommandBuilder("ffmpeg")
        cmd.add_argument("-analyzeduration").add_argument("0") \
           .add_argument("-probesize").add_argument("32") \
           .add_argument("-vaapi_device").add_argument("/dev/dri/renderD128") \
           .add_argument("-r").add_argument(frame_rate) \
           .add_argument("-i").add_argument(os.path.join(out_frames_dir, "frame%08d.jpg")) \
           .add_argument("-i").add_argument(input_file) \
           .add_argument("-map").add_argument("0:v:0") \
           .add_argument("-map").add_argument("1:a:0?") \
           .add_argument("-c:a").add_argument("copy") \
           .add_argument("-c:v").add_argument("hevc_vaapi") \
           .add_argument("-vf").add_argument(filter_arg) \
           .add_argument("-color_range").add_argument("pc") \
           .add_argument("-r").add_argument(frame_rate) \
           .add_argument("-threads").add_argument("16") \
           .add_argument(output_file)
        
        merge_cmd = cmd.build()
        print(f"Merge video command: {merge_cmd}")
        
        try:
            subprocess.check_call(cmd.build_args_list())
        except subprocess.CalledProcessError as e:
            if not self.processing_canceled:
                raise RuntimeError(f"Failed to merge video, error code: {e.returncode}")
            return
        
        # Set progress to 100% when complete
        self.progress_percent = 100
    
    def calculate_frame_based_progress(self):
        if self.total_frame_count <= 0:
            return
        
        progress = 0
        stage_weight = self.current_process_weight
        
        if stage_weight == Constants.EXTRACT_FRAMES_WEIGHT:
            # During extraction, progress is based on input frames extracted
            stage_progress = self.processed_input_frames / self.total_frame_count
            progress = int(stage_progress * Constants.EXTRACT_FRAMES_WEIGHT)
            
        elif stage_weight == Constants.INFERENCE_WEIGHT:
            # During inference, we've already completed extraction
            # Add extraction weight plus progress on inference
            stage_progress = self.processed_output_frames / self.total_frame_count
            progress = Constants.EXTRACT_FRAMES_WEIGHT + \
                      int(stage_progress * Constants.INFERENCE_WEIGHT)
            
        elif stage_weight == Constants.MERGE_VIDEO_WEIGHT:
            # For merge, estimate progress within the remaining 10%
            merge_progress = self.progress_percent / 100.0
            if merge_progress > 1.0:
                merge_progress = 1.0
            
            progress = Constants.EXTRACT_FRAMES_WEIGHT + Constants.INFERENCE_WEIGHT + \
                      int(merge_progress * Constants.MERGE_VIDEO_WEIGHT)
        
        # Ensure progress stays within bounds
        if progress < 0:
            progress = 0
        if progress > 100:
            progress = 100
        
        self.progress_percent = progress
    
    def update_progress_ui(self):
        if self.thread_pool.activeThreadCount() > 0:
            self.progress_bar.setValue(self.progress_percent)
            self.update_status_ui(self.current_status)
    
    def update_status_message(self, status):
        self.current_status = status
        
    def update_status_ui(self, status):
        self.status_label.setText(status)
    
    def show_message(self, message, error=False):
        msg_box = QMessageBox()
        msg_box.setText(message)
        msg_box.setIcon(QMessageBox.Critical if error else QMessageBox.Information)
        msg_box.exec()

# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application metadata
    app.setApplicationName("Video Upscaler")
    app.setApplicationVersion("1.0.0")
    
    # Style adjustments for better look and feel
    app.setStyle("Fusion")
    
    try:
        window = VideoUpscaler()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        QMessageBox.critical(None, "Critical Error",
                           f"Failed to start application: {str(e)}")
        sys.exit(1)
