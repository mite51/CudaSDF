cmake_minimum_required(VERSION 3.18)
project(CudaSDF LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

include(FetchContent)

message(STATUS "Fetching GLFW...")
FetchContent_Declare(
  glfw
  GIT_REPOSITORY https://github.com/glfw/glfw
  GIT_TAG        3.3.8
)
FetchContent_MakeAvailable(glfw)

message(STATUS "Fetching GLAD...")
# Using libigl's pre-generated GLAD. Removing specific tag to fetch latest default branch.
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/libigl/libigl-glad.git
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glad)

find_package(OpenGL REQUIRED)

include_directories(src)

set(SOURCES
    src/main.cpp
    src/SDFMesh/CudaSDFMesh.cpp
    src/SDFMesh/MarchingCubesKernels.cu
    src/SDFMesh/Wrappers.cu
)

add_executable(CudaSDF ${SOURCES})

# Check if glad target exists (it should from FetchContent)
if(TARGET glad)
    target_link_libraries(CudaSDF PRIVATE glad)
else()
    message(WARNING "GLAD target not found. You might need to adjust include paths or link manually.")
    include_directories(${glad_SOURCE_DIR}/include)
    target_sources(CudaSDF PRIVATE ${glad_SOURCE_DIR}/src/glad.c)
endif()

target_link_libraries(CudaSDF PRIVATE 
    glfw
    OpenGL::GL
)

set_target_properties(CudaSDF PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")
