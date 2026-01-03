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
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(glfw)

message(STATUS "Fetching GLAD...")
# Using libigl's pre-generated GLAD. Removing specific tag to fetch latest default branch.
FetchContent_Declare(
  glad
  GIT_REPOSITORY https://github.com/libigl/libigl-glad.git
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(glad)

message(STATUS "Fetching Eigen...")
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
set(EIGEN_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Eigen)

# Copy Assets folder to build directories
file(COPY ${CMAKE_SOURCE_DIR}/assets/ DESTINATION ${CMAKE_BINARY_DIR}/Release/assets)
file(COPY ${CMAKE_SOURCE_DIR}/assets/ DESTINATION ${CMAKE_BINARY_DIR}/Debug/assets)
file(COPY ${CMAKE_SOURCE_DIR}/assets/ DESTINATION ${CMAKE_BINARY_DIR}/assets)

# Also copy individual test_pattern.jpg to root for backward compatibility
file(COPY ${CMAKE_SOURCE_DIR}/assets/test_pattern.jpg DESTINATION ${CMAKE_BINARY_DIR}/Release)
file(COPY ${CMAKE_SOURCE_DIR}/assets/test_pattern.jpg DESTINATION ${CMAKE_BINARY_DIR}/Debug)
file(COPY ${CMAKE_SOURCE_DIR}/assets/test_pattern.jpg DESTINATION ${CMAKE_BINARY_DIR})

find_package(OpenGL REQUIRED)

include_directories(src)

set(SOURCES
    src/main.cpp
    src/SDFMesh/CudaSDFMesh.cpp
    src/SDFMesh/MarchingCubesKernels.cu
    src/SDFMesh/GridUVPacker.cu
    src/SDFMesh/Wrappers.cu
    src/uv_unwrap_harmonic_parameterization/common/mesh.cpp
    src/uv_unwrap_harmonic_parameterization/common/adjacency.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/chart_builder.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/boundary_loops.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/flatten_lscm.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/atlas_packer.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/seam_splitter.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/unwrap_pipeline.cpp
    src/uv_unwrap_harmonic_parameterization/unwrap/uv_validation.cpp
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
    Eigen3::Eigen
)

set_target_properties(CudaSDF PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")
