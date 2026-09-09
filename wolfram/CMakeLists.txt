cmake_minimum_required(VERSION 3.15.0)

project(usearch)

if (NOT SYSTEMID)
    # set system id and build platform
    set(BITNESS 32)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(BITNESS 64)
    endif ()

    set(SYSTEMID NOTFOUND)

    # Determine the current machine's systemid.
    if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND BITNESS EQUAL 64)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
            set(SYSTEMID Linux-ARM64)
        else ()
            set(SYSTEMID Linux-x86-64)
        endif ()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND BITNESS EQUAL 64)
        if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm*")
            set(SYSTEMID MacOSX-ARM64)
        else ()
            set(SYSTEMID MacOSX-x86-64)
        endif ()
    elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows" AND BITNESS EQUAL 64)
        if (_MSVC_C_ARCHITECTURE_FAMILY STREQUAL "ARM64")
            set(SYSTEMID Windows-ARM64)
        else ()
            set(SYSTEMID Windows-x86-64)
        endif ()
    endif ()

    if (NOT SYSTEMID)
        message(FATAL_ERROR "Unable to determine System ID.")
    endif ()
endif ()

if (NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/usr/local/Wolfram/Mathematica")
endif ()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
                      "${WOLFRAM_PATH}/SystemFiles/Links/WSTP/DeveloperKit/${SYSTEMID}/CompilerAdditions/"
)
include(WSTP)
include_directories("${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
link_directories("${WOLFRAM_PATH}/SystemFiles/Libraries/${SYSTEMID}/")

add_library(usearch_wolfram SHARED lib.cpp)

# Link against the Wolfram LibraryLink library
target_link_libraries(usearch_wolfram WSTP::STATIC_LIBRARY)

set_target_properties(usearch_wolfram PROPERTIES CXX_STANDARD 11)
