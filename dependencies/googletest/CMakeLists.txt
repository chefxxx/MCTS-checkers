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
FetchContent_MakeAvailable(googletest)