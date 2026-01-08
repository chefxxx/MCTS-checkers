message("Loading project...")

# ------------------------------------------------------------------------------ #
# Gather source files #
# ------------------------------------------------------------------------------ #

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cu"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp"
)

# ------------------------------------------------------------------------------ #
# Define shared lib #
# ------------------------------------------------------------------------------ #

message("Loading CUDA library...")

add_library(${CUDA_LIB} SHARED ${SOURCES})

# CUDA Architecture setting
set(CMAKE_CUDA_ARCHITECTURES native)
set_property(TARGET ${CUDA_LIB} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
message("Supported CUDA architectures: " "${CMAKE_CUDA_ARCHITECTURES}")

target_include_directories(${CUDA_LIB} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cpu"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/cuda_utils"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
         ${CUDAToolkit_INCLUDE_DIRS}
)

# TODO: linking here to CUDA::cudart breaks on cluster, but without it code does not work
target_link_libraries(${CUDA_LIB} PUBLIC CUDA::cudart)

set_target_properties(${CUDA_LIB} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ------------------------------------------------------------------------------ #
# Define executable #
# ------------------------------------------------------------------------------ #

add_executable(${MAIN} main.cu)
target_link_libraries(${MAIN} PRIVATE ${CUDA_LIB})

# ------------------------------------------------------------------------------ #
# Compilation settings #
# ------------------------------------------------------------------------------ #

# Set build type to Release by default:
if (NOT CMAKE_BUILD_TYPE)
    message(STATUS "No build type provided, default to Release.")
    set(CMAKE_BUILD_TYPE "Release")
endif ()

message("Build type " "${CMAKE_BUILD_TYPE}")
message("Using host compiler " ${CMAKE_CXX_COMPILER_ID})


if (NOT MSVC)
    # C++ compile options
    target_compile_options(${CUDA_LIB} PUBLIC
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>: -g -O0 -Wall -Wextra -pedantic -Wshadow -Wnon-virtual-dtor -Werror>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: -O3 -march=native -funroll-loops>
    )

    # CUDA compile options
    target_compile_options(${CUDA_LIB} PUBLIC
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>: -v -g3 -fno-omit-frame-pointer -G -O0>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: -lineinfo --ptxas-options=-v -Xptxas -O3>
    )
else ()
    # Windows / MSVC - Release Only
    # CXX (Host) Options
    target_compile_options(${CUDA_LIB} PUBLIC
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: /W4 /O2 /Oi /Ot /EHsc /Zc:__cplusplus>
    )

    # CUDA (Device) Options
    target_compile_options(${CUDA_LIB} PUBLIC
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>: -lineinfo --ptxas-options=-v -Xptxas -O3 -Xcompiler /O2,/Oi,/Ot>
    )
endif ()