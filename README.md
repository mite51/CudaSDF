# CudaSDF

This project implements a Marching Cubes algorithm using CUDA and OpenGL for visualization, based on the NVIDIA PhysX implementation.

## Prerequisites

*   **CUDA Toolkit** (11.0 or newer recommended, requires `cub` which is included in modern Toolkits)
*   **CMake** (3.18+)
*   **C++ Compiler** (MSVC, GCC, Clang)
*   **OpenGL** libraries
*   **GLFW3** library
*   **GLAD** (Used for OpenGL function loading)

## Build Instructions

1.  Create a build directory:
    ```bash
    mkdir build
    cd build
    ```

2.  Configure with CMake:
    ```bash
    cmake ..
    ```
    *Note: If CMake cannot find GLAD, you may need to download `glad.c` and `glad.h` from https://glad.dav1d.de/ and place them in the `src` directory or adjust `CMakeLists.txt`.*

3.  Build:
    ```bash
    cmake --build . --config Release
    ```

4.  Run:
    ```bash
    ./Release/CudaSDF.exe
    ```

## Project Structure

*   `src/main.cpp`: Main entry point, OpenGL setup, and rendering loop.
*   `src/MarchingCubesKernels.cu`: CUDA kernels for SDF generation, vertex counting/creation, and index generation.
*   `src/Wrappers.cu`: Wrapper functions to call CUDA kernels from C++.
*   `src/Commons.cuh`: Common data structures (`SDFGrid`) and helper functions.
*   `src/MarchingCubesTables.cuh`: Lookup tables for the Marching Cubes algorithm.

## Implementation Details

The implementation follows a standard GPU Marching Cubes pipeline:
1.  **Generate SDF**: Evaluates a Signed Distance Function (sphere) on a grid.
2.  **Count Vertices**: Checks each cell to see if it intersects the surface and counts required vertices.
3.  **Scan Vertices**: Uses `cub::DeviceScan` to compute prefix sums of vertex counts, determining the memory offset for each cell's vertices.
4.  **Create Vertices**: Interpolates vertex positions along edges and stores them in a Vertex Buffer Object (VBO) mapped from OpenGL.
5.  **Count Indices**: Determines the number of triangle indices required for each cell.
6.  **Scan Indices**: Computes prefix sums for index offsets.
7.  **Create Indices**: Generates triangle topology indices into an Element Array Buffer (EBO) mapped from OpenGL.
8.  **Render**: Draws the mesh using OpenGL.

## Controls

*   The application renders an animated sphere.
*   Close the window to exit.


<video src="https://raw.githubusercontent.com/mite51/CudaSDF/refs/heads/main/docs/demo.mp4" width="640" height="480" controls></video>


