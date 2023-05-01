include(FetchContent)
FetchContent_Declare(
    clipp
    GIT_REPOSITORY https://github.com/muellan/clipp
    GIT_TAG v1.2.3
)
FetchContent_MakeAvailable(clipp)
<<<<<<< HEAD
<<<<<<< HEAD
=======
include_directories(${clipp_SOURCE_DIR}/include)
>>>>>>> 5461e0a (Add: Argument parser for C++ benchmarks)
=======
include_directories(${clipp_SOURCE_DIR}/include)
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
