# Only `safegraphs_format.hpp` reaches for yyjson, and only a compiled target ever includes it, so it is acquired under
# the same condition as `src/`, leaving the header-only target consumable without it. Its objects go inside our own
# libraries, so a packager installing this tree must not publish yyjson's header, archive and CMake config beside ours.
set(YYJSON_INSTALL
    OFF
    CACHE BOOL "" FORCE
)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.12.0
)
FetchContent_MakeAvailable(yyjson)
