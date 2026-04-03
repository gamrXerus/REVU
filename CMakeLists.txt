cmake_minimum_required(VERSION 3.16)
project(VideoUpscaler)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CPU-specific optimizations for Intel i9-11900K (Rocket Lake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=rocketlake -mtune=rocketlake")

# Advanced optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-signed-zeros -fno-trapping-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprefetch-loop-arrays")

# Intel-specific optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mavx512f -mavx512vl -mavx512bw -mavx512dq -mavx512vbmi")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize")

# Cache optimization for Intel
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -falign-functions=32")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -freorder-blocks-algorithm=stc")

# Enable OpenMP with Intel optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# Intel SIMD optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mprefer-vector-width=512")

# Link-time optimization
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto=auto")

# Find required packages
find_package(Qt6 REQUIRED COMPONENTS Widgets Concurrent)
find_package(OpenMP REQUIRED)

# Define source files
set(SOURCES
    main.cpp
)

# Create the executable first
add_executable(VideoUpscaler ${SOURCES})

# IMPORTANT: Add Qt MOC processing AFTER creating the target
set_target_properties(VideoUpscaler PROPERTIES
    AUTOMOC ON
    AUTOUIC ON
    AUTORCC ON
)

# Link libraries
target_link_libraries(VideoUpscaler PRIVATE 
    Qt6::Widgets 
    Qt6::Concurrent
    OpenMP::OpenMP_CXX
)

# Strip debug symbols in Release and enable more aggressive optimizations
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fomit-frame-pointer")
set_target_properties(VideoUpscaler PROPERTIES 
    LINK_FLAGS_RELEASE "-s -Wl,--as-needed -Wl,-O3"
)

# Profile Guided Optimization (PGO) setup
option(USE_PGO "Enable Profile Guided Optimization" OFF)
if(USE_PGO)
    # Generate profile data
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-generate")
    message(STATUS "Building with profile generation. Run the application with typical workloads, then rebuild with -DUSE_PGO_OPTIMIZE=ON")
endif()

option(USE_PGO_OPTIMIZE "Use collected profile data for optimization" OFF)
if(USE_PGO_OPTIMIZE)
    # Use profile data
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fprofile-use -fprofile-correction")
    message(STATUS "Building with profile-guided optimizations")
endif()

# Enable parallel compilation - adjust the number to match your CPU core count
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pipe")
set_property(GLOBAL PROPERTY JOB_POOLS compile_pool=8)  # 8 threads for i9-11900K
set(CMAKE_JOB_POOL_COMPILE compile_pool)

# Add install targets
install(TARGETS VideoUpscaler DESTINATION bin)

# Package generation with CPack
include(InstallRequiredSystemLibraries)
set(CPACK_PACKAGE_VENDOR "VideoUpscaler")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Video Upscaler using ESRGAN")
set(CPACK_PACKAGE_VERSION "2.0.0")
set(CPACK_GENERATOR "RPM")
set(CPACK_PACKAGE_CONTACT "your.email@example.com")
set(CPACK_RPM_PACKAGE_REQUIRES "ffmpeg >= 5.0, qt6-qtbase >= 6.2")
include(CPack)
