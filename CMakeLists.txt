# cmake_minimum_required(VERSION 3.15)
# project(HPC4WC LANGUAGES C CXX)
# # project(HPC4WC LANGUAGES C CXX CUDA)

# set(CMAKE_CXX_STANDARD 17)

# # Find dependencies
# find_package(ZLIB REQUIRED)
# find_package(CUDAToolkit QUIET)

# # Directories with Python/runtime scripts
# set(CONTAINS_PYTHON
#     ${PROJECT_SOURCE_DIR}/src/gt4py
#     ${PROJECT_SOURCE_DIR}/src/reference
#     ${PROJECT_SOURCE_DIR}/tests
# )

# # Source directories
# set(SRC_DIR ${PROJECT_SOURCE_DIR}/src/cuda)

# # CNpy sources
# set(CNPY_SOURCES ${SRC_DIR}/cnpy.cpp)
# set(CNPY_INCLUDE_DIR ${SRC_DIR})

# # Example1 executable sources
# set(EXAMPLE1_SOURCES
#     ${SRC_DIR}/example1.cpp
#     ${CNPY_SOURCES}
# )

# # Main executable sources (add CUDA files only if toolkit found)
# set(MAIN_SOURCES
#     ${SRC_DIR}/main.cpp
#     ${CNPY_SOURCES}
# )

# if(CUDAToolkit_FOUND)
#     message(STATUS "CUDA Toolkit found. Adding CUDA sources.")
#     # list(APPEND MAIN_SOURCES ${SRC_DIR}/compute.cu)
# else()
#     message(WARNING "CUDA Toolkit not found. Building C++ only.")
# endif()

# # Create example1 executable
# add_executable(example1 ${EXAMPLE1_SOURCES})
# target_include_directories(example1 PRIVATE ${CNPY_INCLUDE_DIR})
# target_link_libraries(example1 PRIVATE ZLIB::ZLIB)

# # Create main executable
# add_executable(main ${MAIN_SOURCES})
# target_include_directories(main PRIVATE ${CNPY_INCLUDE_DIR})

# target_link_libraries(main PRIVATE ZLIB::ZLIB)
# if(CUDAToolkit_FOUND)
#     target_link_libraries(main PRIVATE CUDA::cudart)
# endif()

# # Create symlinks for Python/runtime folders in build directory
# foreach(INPUT_DIR ${CONTAINS_PYTHON})
#     get_filename_component(DIR_NAME ${INPUT_DIR} NAME)
#     add_custom_command(TARGET main POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E create_symlink
#             ${INPUT_DIR} $<TARGET_FILE_DIR:main>/${DIR_NAME}
#         COMMENT "Creating symlink to ${DIR_NAME} in build directory"
#     )
# endforeach()


cmake_minimum_required(VERSION 3.18)
project(CUDAArccosProject LANGUAGES CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional flag for laptop builds
option(LAPTOP_BUILD "Disable sm_90 architecture for local/laptop builds" OFF)

if(NOT LAPTOP_BUILD)
    message(STATUS "Cluster build: setting CUDA architecture to sm_90")
    set(CMAKE_CUDA_ARCHITECTURES 90)
else()
    message(STATUS "Laptop build: using default CUDA architecture")
endif()

# Define source files
set(CUDA_SOURCES
    src/cuda/main.cu
    src/cuda/arccos_cuda.cu
)

# Add the executable
add_executable(cuda_arccos ${CUDA_SOURCES})

# Include the headers
target_include_directories(cuda_arccos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Add stencil GPU executable
set(STENCIL_SOURCES
    src/cuda/stencil2d-gpu.cu
    src/cuda/stencil2d_helper/stencil_kernels.cu
)

add_executable(stencil2d_gpu ${STENCIL_SOURCES})
target_include_directories(stencil2d_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Add stencil GPU executable with multiple streams
set(STENCIL_STREAM_SOURCES
    src/cuda/stencil2d-gpu_streams.cu
    src/cuda/stencil2d_helper/stencil_kernels.cu
)

add_executable(stencil2d_gpu_streams ${STENCIL_STREAM_SOURCES})
target_include_directories(stencil2d_gpu_streams PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)
