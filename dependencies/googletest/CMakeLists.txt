# ------------------------------------------------------------------------------ #
#                               Add google test                                  #
# ------------------------------------------------------------------------------ #

message(STATUS "Adding googletest...")

include(FetchContent)

FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest
        GIT_TAG main
        GIT_PROGRESS   TRUE
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)