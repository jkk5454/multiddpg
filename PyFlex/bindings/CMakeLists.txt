cmake_minimum_required(VERSION 3.5)
project(PyFleX)

find_package(CUDA 9.0 REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MMD -m64 -Wall -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES -fPIC -fpermissive -fno-strict-aliasing -O3")

set(ROOT $ENV{PYFLEXROOT})
#set(ROOT ${CMAKE_SOURCE_DIR}/multiddpg/PyFlex)
set(EXAMPLE_BIN pyflex)

set(SOURCES ${SOURCES} ${ROOT}/bindings/pyflex.cpp)
set(SOURCES ${SOURCES} ${ROOT}/bindings/imgui.cpp)
set(SOURCES ${SOURCES} ${ROOT}/bindings/shadersDemoContext.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/aabbtree.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/core.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/extrude.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/maths.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/mesh.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/perlin.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/pfm.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/platform.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/png.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/sdf.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/tga.cpp)
set(SOURCES ${SOURCES} ${ROOT}/core/voxelize.cpp)
set(SOURCES ${SOURCES} ${ROOT}/bindings/opengl/imguiRenderGL.cpp)
set(SOURCES ${SOURCES} ${ROOT}/bindings/opengl/shader.cpp)
set(SOURCES ${SOURCES} ${ROOT}/bindings/opengl/shadersGL.cpp)

#set(FLEXHEADERS ${ROOT})
set(FLEXHEADERS
    ${ROOT}
    ${ROOT}/core
    ${ROOT}/bindings
    ${ROOT}/bindings/opengl
    ${ROOT}/external/SDL2-2.0.4/include
)
set(FLEXLIB ${ROOT} ${ROOT}/lib/linux64 ${ROOT}/bindings ${ROOT}/external/SDL2-2.0.4/lib/x64)

pybind11_add_module(${EXAMPLE_BIN} ${SOURCES})
target_include_directories(${EXAMPLE_BIN} PRIVATE ${FLEXHEADERS})
target_include_directories(${EXAMPLE_BIN} PRIVATE ${FLEXLIB})
target_link_libraries(${EXAMPLE_BIN} PRIVATE EGL)
target_link_libraries(${EXAMPLE_BIN} PRIVATE ${ROOT}/lib/linux64/NvFlexExtReleaseCUDA_x64.a)
target_link_libraries(${EXAMPLE_BIN} PRIVATE ${ROOT}/lib/linux64/NvFlexReleaseCUDA_x64.a)
target_link_libraries(${EXAMPLE_BIN} PRIVATE ${ROOT}/external/SDL2-2.0.4/lib/x64/libSDL2-2.0.so.0.4.0)
target_link_libraries(${EXAMPLE_BIN} PRIVATE ${ROOT}/external/SDL2-2.0.4/lib/x64/libSDL2.a)
target_link_libraries(${EXAMPLE_BIN} PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(${EXAMPLE_BIN} PRIVATE GL GLU)

