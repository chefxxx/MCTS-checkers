# ------------------------------------------------------------------------------ #
#                                Project setup                                   #
# ------------------------------------------------------------------------------ #

cmake_minimum_required(VERSION 3.30)
set(PROJECT_NAME "MCTS-checkers")
project(${PROJECT_NAME} LANGUAGES C CXX)

set(MAIN ${PROJECT_NAME})
set(TEST ${PROJECT_NAME}_test)
set(LIB ${PROJECT_NAME}_lib)

set(PROJECT_VERSION "1.0")
set(PROJECT_AUTHOR "Mateusz Mikiciuk")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
    message("Windows system detected...")
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

# enable linking static libs to project
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ------------------------------------------------------------------------------ #
#                                  Load dirs                                     #
# ------------------------------------------------------------------------------ #

add_subdirectory(dependencies)
add_subdirectory(project)
add_subdirectory(tests)
enable_testing()