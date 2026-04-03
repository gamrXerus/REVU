// main.cpp
#include <QApplication>
#include <QMainWindow>
#include <QWidget>
#include <QVBoxLayout>
#include <QPushButton>
#include <QProgressBar>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QtConcurrent/QtConcurrent>
#include <QFutureWatcher>
#include <QLabel>
#include <QLineEdit>
#include <QGroupBox>
#include <QFormLayout>
#include <QHBoxLayout>
#include <filesystem>
#include <chrono>
#include <thread>
#include <memory>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <vector>
#include <span>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <optional>
#include <mutex>

namespace fs = std::filesystem;

// RAII wrapper for temporary directories
class TemporaryDirectory {
public:
    explicit TemporaryDirectory(const std::string& path)
        : path_(path) {
        std::error_code ec;
        if (fs::exists(path_, ec) && !fs::is_directory(path_, ec)) {
            throw std::runtime_error("Path exists but is not a directory: " + path_);
        }
        
        if (!fs::exists(path_, ec)) {
            if (!fs::create_directories(path_, ec) && ec) {
                throw std::runtime_error("Failed to create directory: " + path_ + 
                                        " (Error: " + ec.message() + ")");
            }
        }
        
        valid_ = true;
    }
    
    ~TemporaryDirectory() {
        if (valid_) {
            std::error_code ec;
            fs::remove_all(path_, ec);
            if (ec) {
                std::cerr << "Warning: Failed to remove temporary directory: " 
                         << path_ << " (Error: " << ec.message() << ")" << std::endl;
            }
        }
    }
    
    // Disable copying
    TemporaryDirectory(const TemporaryDirectory&) = delete;
    TemporaryDirectory& operator=(const TemporaryDirectory&) = delete;
    
    // Allow moving
    TemporaryDirectory(TemporaryDirectory&& other) noexcept
        : path_(std::move(other.path_)), valid_(other.valid_) {
        other.valid_ = false;
    }
    
    TemporaryDirectory& operator=(TemporaryDirectory&& other) noexcept {
        if (this != &other) {
            // Clean up current directory if valid
            if (valid_) {
                std::error_code ec;
                fs::remove_all(path_, ec);
            }
            
            // Move from other
            path_ = std::move(other.path_);
            valid_ = other.valid_;
            
            // Invalidate other
            other.valid_ = false;
        }
        return *this;
    }
    
    const std::string& path() const { return path_; }
    
private:
    std::string path_;
    bool valid_{false};
};

// Safe command builder to prevent command injection
class CommandBuilder {
public:
    explicit CommandBuilder(const std::string& executable_name)
        : executable_(executable_name) {}
    
    CommandBuilder& add_argument(const std::string& arg) {
        arguments_.push_back(arg);
        return *this;
    }
    
    CommandBuilder& add_flag(const std::string& flag, const std::string& value) {
        arguments_.push_back(flag);
        arguments_.push_back(value);
        return *this;
    }
    
    std::string build() const {
        std::ostringstream command;
        command << escape_argument(executable_);
        
        for (const auto& arg : arguments_) {
            command << ' ' << escape_argument(arg);
        }
        
        return command.str();
    }
    
private:
    std::string executable_;
    std::vector<std::string> arguments_;
    
    std::string escape_argument(const std::string& arg) const {
        if (arg.empty()) {
            return "\"\"";
        }
        
        bool needs_quotes = arg.find_first_of(" \t\n\v\"") != std::string::npos;
        
        if (!needs_quotes) {
            return arg;
        }
        
        std::ostringstream result;
        result << '"';
        
        for (auto it = arg.begin(); it != arg.end(); ++it) {
            if (*it == '"' || *it == '\\') {
                result << '\\';
            }
            result << *it;
        }
        
        result << '"';
        return result.str();
    }
};

// Constants for progress updates and other settings
namespace Constants {
    // Process weightings - how much of total progress each step takes
    constexpr int EXTRACT_FRAMES_WEIGHT = 20;
    constexpr int INFERENCE_WEIGHT = 70;
    constexpr int MERGE_VIDEO_WEIGHT = 10;
    constexpr int TOTAL_WEIGHT = EXTRACT_FRAMES_WEIGHT + INFERENCE_WEIGHT + MERGE_VIDEO_WEIGHT;
    
    // Default settings
    constexpr const char* DEFAULT_REALESRGAN_PATH = "./realesrgan-ncnn-vulkan";
    constexpr const char* DEFAULT_FRAME_RATE = "23.98";
    
    // Temp directory names
    constexpr const char* TEMP_FRAMES_DIR = "tmp_frames";
    constexpr const char* OUTPUT_FRAMES_DIR = "out_frames";
    
    // UI update interval in milliseconds
    constexpr int UI_UPDATE_INTERVAL_MS = 100;
    constexpr int FRAME_COUNT_CHECK_INTERVAL_MS = 500;
}

// Thread-safe string class since std::atomic<std::string> is not allowed
class ThreadSafeString {
public:
    ThreadSafeString(const std::string& initial = "") : value_(initial) {}
    
    void store(const std::string& new_value) {
        std::lock_guard<std::mutex> lock(mutex_);
        value_ = new_value;
    }
    
    std::string load() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return value_;
    }
    
private:
    mutable std::mutex mutex_;
    std::string value_;
};

// Main application class
class VideoUpscaler : public QMainWindow {
public:
    VideoUpscaler() {
        // Set window properties
        setWindowTitle("Video Upscaler");
        setMinimumSize(600, 400);
        
        // Setup central widget and layout
        central_widget = new QWidget(this);
        setCentralWidget(central_widget);
        layout = new QVBoxLayout(central_widget);
        
        // Create UI components
        create_buttons();
        create_progress_section();
        create_settings_section();
        
        // Connect future watcher signals
        connect(&future_watcher_, &QFutureWatcher<void>::finished,
                this, &VideoUpscaler::on_processing_finished);
        
        // Setup progress update timer
        QTimer* progress_timer = new QTimer(this);
        connect(progress_timer, &QTimer::timeout, this, [this]() {
            if (future_watcher_.isRunning()) {
                progress_bar->setValue(progress_percent_.load());
                update_status_message(current_status_.load());
                QApplication::processEvents();
            }
        });
        progress_timer->start(Constants::UI_UPDATE_INTERVAL_MS);
    }

private:
    // UI components
    QWidget* central_widget;
    QVBoxLayout* layout;
    QProgressBar* progress_bar;
    QLabel* status_label;
    QLineEdit* realesrgan_path_edit{nullptr};
    
    // Processing state
    QStringList input_files;
    int current_file_index = 0;
    QFutureWatcher<void> future_watcher_;
    std::atomic<int> progress_percent_{0};
    std::atomic<bool> processing_canceled_{false};
    ThreadSafeString current_status_{"Ready"};
    int last_upscale_factor_{4};
    std::string last_model_name_{"realesrgan-x4plus"};
    std::optional<std::string> last_error_;
    
    // Frame counting state
    std::atomic<int> total_frame_count_{0};
    std::atomic<int> processed_input_frames_{0};
    std::atomic<int> processed_output_frames_{0};
    std::atomic<int> current_process_weight_{0};
    
    void create_buttons() {
        auto select_files_button = new QPushButton("Select Input Files", this);
        select_files_button->setIcon(QIcon::fromTheme("document-open"));
        connect(select_files_button, &QPushButton::clicked, this, &VideoUpscaler::select_multiple_files);
        layout->addWidget(select_files_button);
        
        std::vector<std::tuple<const char*, int, const char*>> button_info = {
            {"Upscale 3D/Real-Life Videos (4x)", 4, "realesrgan-x4plus"},
            {"Upscale 2D Animations (4x)", 4, "realesr-animevideov3-x4"},
            {"Upscale 2D Animations (2x)", 2, "realesr-animevideov3-x2"}
        };
        
        for (const auto& [text, upscale_factor, model_name] : button_info) {
            auto button = new QPushButton(text, this);
            connect(button, &QPushButton::clicked, this, [this, upscale_factor, model_name]() {
                if (input_files.empty()) {
                    show_message("Please select input files first", true);
                    return;
                }
                
                if (future_watcher_.isRunning()) {
                    show_message("Processing already in progress", true);
                    return;
                }
                
                current_file_index = 0;
                last_upscale_factor_ = upscale_factor;
                last_model_name_ = model_name;
                process_next_file(upscale_factor, model_name);
            });
            layout->addWidget(button);
        }
        
        // Cancel button
        auto cancel_button = new QPushButton("Cancel Processing", this);
        cancel_button->setIcon(QIcon::fromTheme("process-stop"));
        connect(cancel_button, &QPushButton::clicked, this, [this]() {
            if (future_watcher_.isRunning()) {
                processing_canceled_ = true;
                update_status_message("Canceling...");
                show_message("Canceling current process. Please wait...");
            }
        });
        layout->addWidget(cancel_button);
    }
    
    void create_progress_section() {
        // Status label
        status_label = new QLabel("Ready", this);
        layout->addWidget(status_label);
        
        // Progress bar
        progress_bar = new QProgressBar(this);
        progress_bar->setMinimum(0);
        progress_bar->setMaximum(100);
        progress_bar->setValue(0);
        layout->addWidget(progress_bar);
    }
    
    void create_settings_section() {
        auto settings_group = new QGroupBox("Settings", this);
        auto settings_layout = new QFormLayout(settings_group);
        
        // RealESRGAN path setting
        realesrgan_path_edit = new QLineEdit(Constants::DEFAULT_REALESRGAN_PATH, this);
        auto browse_button = new QPushButton("Browse", this);
        auto path_layout = new QHBoxLayout();
        path_layout->addWidget(realesrgan_path_edit);
        path_layout->addWidget(browse_button);
        
        connect(browse_button, &QPushButton::clicked, this, [this]() {
            QString path = QFileDialog::getOpenFileName(this, "Select RealESRGAN Executable");
            if (!path.isEmpty()) {
                realesrgan_path_edit->setText(path);
            }
        });
        
        settings_layout->addRow("RealESRGAN Path:", path_layout);
        layout->addWidget(settings_group);
    }
    
    void select_multiple_files() {
        QStringList new_files = QFileDialog::getOpenFileNames(
            this, "Select Input Video Files",
            QString(), "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv);;All Files (*.*)"
        );
        
        if (!new_files.empty()) {
            input_files = new_files;
            show_message(QString("Selected %1 files for processing").arg(input_files.size()).toStdString());
        }
    }
    
    void process_next_file(int upscale_factor, const std::string& model_name) {
        if (current_file_index >= input_files.size()) {
            show_message("All files processed successfully");
            update_status_message("Ready");
            progress_percent_ = 0;
            progress_bar->setValue(0);
            return;
        }
        
        // Reset progress and cancellation flag
        progress_percent_ = 0;
        processing_canceled_ = false;
        last_error_.reset();
        
        QString current_file = input_files[current_file_index];
        update_status_message("Processing: " + fs::path(current_file.toStdString()).filename().string());
        
        // Start the upscaling in a background thread
        QFuture<void> future = QtConcurrent::run([this, current_file, upscale_factor, model_name]() {
            try {
                auto tmp_dir = std::make_unique<TemporaryDirectory>(Constants::TEMP_FRAMES_DIR);
                auto out_dir = std::make_unique<TemporaryDirectory>(Constants::OUTPUT_FRAMES_DIR);
                
                perform_video_upscaling(
                    current_file.toStdString(),
                    tmp_dir->path(),
                    out_dir->path(),
                    upscale_factor,
                    model_name
                );
                
                // Note: tmp_dir and out_dir will be automatically cleaned up
                // when they go out of scope thanks to RAII
            } catch (const std::exception& e) {
                last_error_ = e.what();
                throw; // Rethrow to be caught by the watcher
            }
        });
        
        // Set up the watcher to monitor the future
        future_watcher_.setFuture(future);
    }
    
    void on_processing_finished() {
        try {
            // This will rethrow any exception that occurred in the worker thread
            future_watcher_.future().waitForFinished();
            
            // Check if processing was canceled
            if (processing_canceled_.load()) {
                show_message("Processing canceled by user");
                update_status_message("Ready");
                progress_percent_ = 0;
                progress_bar->setValue(0);
                return;
            }
            
            // If we get here, processing completed successfully
            current_file_index++;
            process_next_file(last_upscale_factor_, last_model_name_);
        }
        catch (const std::exception&) {
            // Use the saved error message
            std::string error_msg = last_error_ ? *last_error_ : "Unknown error";
            show_message(std::string("Error processing file ") + 
                        std::to_string(current_file_index + 1) + ": " + error_msg, true);
            update_status_message("Error");
        }
    }
    
    int count_frames_in_directory(const std::string& dir_path, const std::string& extension) {
        int count = 0;
        try {
            for (const auto& entry : fs::directory_iterator(dir_path)) {
                if (entry.is_regular_file() && entry.path().extension() == extension) {
                    count++;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error counting frames: " << e.what() << std::endl;
        }
        return count;
    }
    
    void perform_video_upscaling(
        const std::string& input_file,
        const std::string& tmp_frames_dir,
        const std::string& out_frames_dir,
        int upscale_factor,
        const std::string& model_name
    ) {
        auto start_time = std::chrono::steady_clock::now();
        
        try {
            // First, get estimated frame count from the video
            int estimated_frames = get_frame_count(input_file);
            total_frame_count_.store(estimated_frames > 0 ? estimated_frames : 1000); // fallback if can't detect
            processed_input_frames_.store(0);
            processed_output_frames_.store(0);
            
            // Start frame counting timer
            QTimer* frame_counter_timer = new QTimer();
            frame_counter_timer->moveToThread(QApplication::instance()->thread());
            frame_counter_timer->setSingleShot(false);
            QMetaObject::invokeMethod(frame_counter_timer, [this, frame_counter_timer, tmp_frames_dir, out_frames_dir]() {
                connect(frame_counter_timer, &QTimer::timeout, [this, tmp_frames_dir, out_frames_dir, frame_counter_timer]() {
                    // Only run if not canceled
                    if (!processing_canceled_.load()) {
                        // Check current process stage and count appropriate frames
                        if (current_process_weight_.load() == Constants::EXTRACT_FRAMES_WEIGHT) {
                            int current_input_frames = count_frames_in_directory(tmp_frames_dir, ".png");
                            processed_input_frames_.store(current_input_frames);
                        } else if (current_process_weight_.load() == Constants::INFERENCE_WEIGHT) {
                            int current_output_frames = count_frames_in_directory(out_frames_dir, ".jpg");
                            processed_output_frames_.store(current_output_frames);
                        }
                        
                        // Update progress based on current frame counts
                        calculate_frame_based_progress();
                    }
                    
                    // Stop the timer when processing is complete or canceled
                    if (progress_percent_.load() >= 100 || processing_canceled_.load()) {
                        frame_counter_timer->stop();
                        frame_counter_timer->deleteLater();
                    }
                });
                frame_counter_timer->start(Constants::FRAME_COUNT_CHECK_INTERVAL_MS);
            });
            
            // Set initial process weight
            current_process_weight_.store(Constants::EXTRACT_FRAMES_WEIGHT);
            
            update_status_message("Extracting frames...");
            extract_frames(input_file, tmp_frames_dir);
            if (processing_canceled_.load()) return;
            
            // Update process weight for inference
            current_process_weight_.store(Constants::INFERENCE_WEIGHT);
            
            update_status_message("Running inference...");
            run_inference(tmp_frames_dir, out_frames_dir, model_name, upscale_factor);
            if (processing_canceled_.load()) return;
            
            // Update for final stage
            current_process_weight_.store(Constants::MERGE_VIDEO_WEIGHT);
            
            update_status_message("Getting frame rate...");
            std::string frame_rate = get_frame_rate(input_file);
            if (processing_canceled_.load()) return;
            
            update_status_message("Merging video...");
            merge_video(frame_rate, out_frames_dir, input_file);
            
            if (!processing_canceled_.load()) {
                auto [hours, minutes, seconds] = calculate_time_taken(start_time);
                std::string completion_msg = std::format(
                    "Upscaling completed in {}h {}m {}s",
                    hours, minutes, seconds
                );
                update_status_message(completion_msg);
                std::cout << completion_msg << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error during upscaling: " << e.what() << std::endl;
            throw;
        }
    }
    
    int get_frame_count(const std::string& input_file) {
        CommandBuilder cmd("ffprobe");
        cmd.add_argument("-v").add_argument("error")
           .add_argument("-count_frames")
           .add_argument("-select_streams").add_argument("v:0")
           .add_argument("-show_entries").add_argument("stream=nb_read_frames")
           .add_argument("-of").add_argument("csv=p=0")
           .add_argument(input_file);
        
        std::string ffprobe_cmd = cmd.build();
        std::cout << "Frame count command: " << ffprobe_cmd << std::endl;
        
        std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(ffprobe_cmd.c_str(), "r"), pclose);
        if (!pipe) {
            std::cerr << "Warning: Failed to get frame count" << std::endl;
            return 0;
        }
        
        std::array<char, 128> buffer;
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        
        // Trim whitespace and newlines
        result.erase(result.find_last_not_of(" \n\r\t") + 1);
        result.erase(0, result.find_first_not_of(" \n\r\t"));
        
        try {
            return std::stoi(result);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to parse frame count: " << e.what() << std::endl;
            return 0;
        }
    }
    
    void extract_frames(const std::string& input_file, const std::string& tmp_frames_dir) {
        CommandBuilder cmd("ffmpeg");
        cmd.add_argument("-analyzeduration").add_argument("0")
           .add_argument("-probesize").add_argument("32")
           .add_argument("-i").add_argument(input_file)
           .add_argument("-qscale:v").add_argument("1")
           .add_argument("-qmin").add_argument("1")
           .add_argument("-qmax").add_argument("1")
           .add_argument("-vsync").add_argument("0")
           .add_argument("-threads").add_argument("16")
           .add_argument(tmp_frames_dir + "/frame%08d.png");
        
        std::string ffmpeg_cmd = cmd.build();
        std::cout << "Extract frames command: " << ffmpeg_cmd << std::endl;
        
        int result = std::system(ffmpeg_cmd.c_str());
        if (result != 0 && !processing_canceled_.load()) {
            throw std::runtime_error("Failed to extract frames, error code: " + 
                                    std::to_string(result));
        }
        
        // Count actual frames extracted for accurate progress
        int frame_count = count_frames_in_directory(tmp_frames_dir, ".png");
        if (frame_count > 0 && frame_count != total_frame_count_.load()) {
            total_frame_count_.store(frame_count);
        }
        processed_input_frames_.store(frame_count);
        
        // Calculate progress based on frames
        calculate_frame_based_progress();
    }
    
    void run_inference(
        const std::string& tmp_frames_dir,
        const std::string& out_frames_dir,
        const std::string& model_name,
        int upscale_factor
    ) {
        CommandBuilder cmd(realesrgan_path_edit->text().toStdString());
        cmd.add_argument("-i").add_argument(tmp_frames_dir)
           .add_argument("-o").add_argument(out_frames_dir)
           .add_argument("-n").add_argument(model_name)
           .add_argument("-s").add_argument(std::to_string(upscale_factor))
           .add_argument("-f").add_argument("jpg");
        
        std::string inference_cmd = cmd.build();
        std::cout << "Inference command: " << inference_cmd << std::endl;
        
        int result = std::system(inference_cmd.c_str());
        if (result != 0 && !processing_canceled_.load()) {
            throw std::runtime_error("Failed to run inference, error code: " + 
                                    std::to_string(result));
        }
        
        // Count actual processed frames for accurate progress
        int output_frame_count = count_frames_in_directory(out_frames_dir, ".jpg");
        processed_output_frames_.store(output_frame_count);
        
        // Calculate progress based on frames
        calculate_frame_based_progress();
    }
    
    std::string get_frame_rate(const std::string& input_file) {
        CommandBuilder cmd("ffprobe");
        cmd.add_argument("-analyzeduration").add_argument("0")
           .add_argument("-probesize").add_argument("32")
           .add_argument("-v").add_argument("error")
           .add_argument("-select_streams").add_argument("v:0")
           .add_argument("-show_entries").add_argument("stream=avg_frame_rate")
           .add_argument("-of").add_argument("default=noprint_wrappers=1:nokey=1")
           .add_argument(input_file);
        
        std::string ffprobe_cmd = cmd.build();
        std::cout << "Get frame rate command: " << ffprobe_cmd << std::endl;
        
        std::unique_ptr<FILE, int(*)(FILE*)> pipe(popen(ffprobe_cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("Failed to run ffprobe command");
        }
        
        std::array<char, 128> buffer;
        std::string result;
        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        
        // Trim whitespace and newlines
        result.erase(result.find_last_not_of(" \n\r\t") + 1);
        result.erase(0, result.find_first_not_of(" \n\r\t"));
        
        if (result.empty()) {
            std::cout << "Frame rate detection failed, using default: " 
                     << Constants::DEFAULT_FRAME_RATE << std::endl;
            return Constants::DEFAULT_FRAME_RATE;
        }
        
        std::string::size_type pos = result.find('/');
        if (pos != std::string::npos) {
            try {
                int numerator = std::stoi(result.substr(0, pos));
                int denominator = std::stoi(result.substr(pos + 1));
                if (denominator != 0) {
                    double frame_rate = static_cast<double>(numerator) / denominator;
                    return std::to_string(frame_rate);
                }
            } catch (const std::exception& e) {
                std::cerr << "Error parsing frame rate: " << e.what() 
                         << ", using default" << std::endl;
                return Constants::DEFAULT_FRAME_RATE;
            }
        }
        
        return result;
    }
    
void merge_video(
    const std::string& frame_rate,
    const std::string& out_frames_dir,
    const std::string& input_file
) {
    fs::path input_path(input_file);
    std::string output_file = std::format(
        "output_{}_{}x.mp4",
        input_path.stem().string(),
        last_upscale_factor_
    );
    
    // Fix: use single quotes around the filter expression
    std::string filter_arg = "format=nv12|vaapi,hwupload";
    
    CommandBuilder cmd("ffmpeg");
    cmd.add_argument("-analyzeduration").add_argument("0")
       .add_argument("-probesize").add_argument("32")
       .add_argument("-vaapi_device").add_argument("/dev/dri/renderD128")
       .add_argument("-r").add_argument(frame_rate)
       .add_argument("-i").add_argument(out_frames_dir + "/frame%08d.jpg")
       .add_argument("-i").add_argument(input_file)
       .add_argument("-map").add_argument("0:v:0")
       .add_argument("-map").add_argument("1:a:0?")
       .add_argument("-c:a").add_argument("copy")
       .add_argument("-c:v").add_argument("hevc_vaapi")
       .add_argument("-vf").add_argument("'" + filter_arg + "'") // Add single quotes 
       .add_argument("-color_range").add_argument("pc")
       .add_argument("-r").add_argument(frame_rate)
       .add_argument("-threads").add_argument("16")
       .add_argument(output_file);
    
    std::string merge_cmd = cmd.build();
    std::cout << "Merge video command: " << merge_cmd << std::endl;
    
    int result = std::system(merge_cmd.c_str());
    if (result != 0 && !processing_canceled_.load()) {
        throw std::runtime_error("Failed to merge video, error code: " + 
                                std::to_string(result));
    }
    
    // Set progress to 100% when complete
    progress_percent_.store(100);
}
    
    void calculate_frame_based_progress() {
        int total_frames = total_frame_count_.load();
        if (total_frames <= 0) return;
        
        int progress = 0;
        int stage_weight = current_process_weight_.load();
        
        if (stage_weight == Constants::EXTRACT_FRAMES_WEIGHT) {
            // During extraction, progress is based on input frames extracted
            double stage_progress = 
                static_cast<double>(processed_input_frames_.load()) / total_frames;
            progress = static_cast<int>(stage_progress * Constants::EXTRACT_FRAMES_WEIGHT);
            
        } else if (stage_weight == Constants::INFERENCE_WEIGHT) {
            // During inference, we've already completed extraction
            // Add extraction weight plus progress on inference
            double stage_progress = 
                static_cast<double>(processed_output_frames_.load()) / total_frames;
            progress = Constants::EXTRACT_FRAMES_WEIGHT + 
                      static_cast<int>(stage_progress * Constants::INFERENCE_WEIGHT);
            
        } else if (stage_weight == Constants::MERGE_VIDEO_WEIGHT) {
            // For merge, estimate progress within the remaining 10%
            double merge_progress = progress_percent_.load() / 100.0;
            if (merge_progress > 1.0) merge_progress = 1.0;
progress = Constants::EXTRACT_FRAMES_WEIGHT + Constants::INFERENCE_WEIGHT +
                      static_cast<int>(merge_progress * Constants::MERGE_VIDEO_WEIGHT);
        }
        
        // Ensure progress stays within bounds
        if (progress < 0) progress = 0;
        if (progress > 100) progress = 100;
        
        progress_percent_.store(progress);
    }
    
    // This method is used for the final merge stage
    void update_progress(int value) {
        // Only used for the final merge stage now
        if (current_process_weight_.load() == Constants::MERGE_VIDEO_WEIGHT) {
            calculate_frame_based_progress();
        }
    }
    
    void update_status_message(const std::string& status) {
        current_status_.store(status);
        if (QThread::currentThread() == QApplication::instance()->thread()) {
            status_label->setText(QString::fromStdString(status));
        }
    }
    
    std::tuple<int, int, int> calculate_time_taken(const std::chrono::steady_clock::time_point& start_time) {
        auto end_time = std::chrono::steady_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count();
        int hours = time_diff / 3600;
        int minutes = (time_diff % 3600) / 60;
        int seconds = time_diff % 60;
        return {hours, minutes, seconds};
    }
    
    void show_message(const std::string& message, bool error = false) {
        QMessageBox msg_box;
        msg_box.setText(QString::fromStdString(message));
        msg_box.setIcon(error ? QMessageBox::Critical : QMessageBox::Information);
        msg_box.exec();
    }
};

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    
    // Set application metadata
    QApplication::setApplicationName("Video Upscaler");
    QApplication::setApplicationVersion("1.1.0");
    
    // Style adjustments for better look and feel
    app.setStyle("Fusion");
    
    try {
        VideoUpscaler window;
        window.show();
        return app.exec();
    } catch (const std::exception& e) {
        QMessageBox::critical(nullptr, "Critical Error",
                             QString("Failed to start application: %1").arg(e.what()));
        return 1;
    }
}            
            
