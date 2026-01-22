message("Loading project...")

# ------------------------------------------------------------------------------ #
# Gather source files #
# ------------------------------------------------------------------------------ #

file(GLOB SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp"
)

# ------------------------------------------------------------------------------ #
# Define shared lib #
# ------------------------------------------------------------------------------ #

message("Loading solution library...")

add_library(${LIB} SHARED ${SOURCES})


target_include_directories(${LIB} PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include/general"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/utils"
)

# ------------------------------------------------------------------------------ #
# Define executable #
# ------------------------------------------------------------------------------ #

add_executable(${MAIN} main.cpp)
target_link_libraries(${MAIN} PRIVATE ${LIB})

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
    target_compile_options(${LIB} PUBLIC
            $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>: -g -O0 -Wall -Wextra -pedantic -Wshadow -Wnon-virtual-dtor -Werror>
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: -O3 -march=native -funroll-loops>
    )
else ()
    # Windows / MSVC - Release Only
    # CXX (Host) Options
    target_compile_options(${LIB} PUBLIC
            $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>: /W4 /O2 /Oi /Ot /EHsc /Zc:__cplusplus>
    )
endif ()