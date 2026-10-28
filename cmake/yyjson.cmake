# Only `safegraphs_format.hpp` reaches for yyjson, and only a compiled target ever includes it, so it builds only for
# the targets that link it, leaving the header-only target consumable without it. Its objects go inside our own
# libraries, so a packager installing this tree must not publish yyjson's header, archive and CMake config beside ours.
set(YYJSON_INSTALL OFF)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.13.0
    EXCLUDE_FROM_ALL SYSTEM
)
FetchContent_MakeAvailable(yyjson)
