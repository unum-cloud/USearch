# The pin for yyjson, which builds only for the targets that link it, so header-only consumers never need it.
# Its objects land inside the project's own libraries and modules, so they compile position-independent, and a packager
# installing this tree must not publish yyjson's header, archive and CMake config beside the project's.
include(FetchContent)
set(YYJSON_INSTALL OFF)
FetchContent_Declare(
    yyjson
    GIT_REPOSITORY https://github.com/ibireme/yyjson.git
    GIT_TAG 0.13.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL SYSTEM
)
FetchContent_MakeAvailable(yyjson)
set_property(TARGET yyjson PROPERTY POSITION_INDEPENDENT_CODE ON)
# yyjson's own build declares no namespaced name, and a parent project that fetched it first may have added one.
if (NOT TARGET yyjson::yyjson)
    add_library(yyjson::yyjson ALIAS yyjson)
endif ()
