<<<<<<< HEAD
include(ExternalProject)
set(JEMALLOC_DISABLE_TLS)

if (${USTORE_BUILD_SDK_PYTHON})
    set(JEMALLOC_DISABLE_TLS --disable-initial-exec-tls)
endif ()

if (${USTORE_REBUILD_JEMALLOC})
=======

include(ExternalProject)
set(JEMALLOC_DISABLE_TLS)

if(${USTORE_BUILD_SDK_PYTHON})
    set(JEMALLOC_DISABLE_TLS --disable-initial-exec-tls)
endif()

if(${USTORE_REBUILD_JEMALLOC})
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
    set(JEMALLOC_PREFIX_DIR ${CMAKE_BINARY_DIR}/_deps/jemalloc)
    set(JEMALLOC_SRC_DIR ${JEMALLOC_PREFIX_DIR}/src/jemalloc)
    set(JEMALLOC_INSTALL_DIR ${JEMALLOC_PREFIX_DIR}/install)
    set(JEMALLOC_DEPENDENCY jemalloc)

<<<<<<< HEAD
    ExternalProject_Add(
        jemalloc
        GIT_REPOSITORY https://github.com/jemalloc/jemalloc.git
        GIT_TAG 5.3.0
        PREFIX ${JEMALLOC_PREFIX_DIR}
        CONFIGURE_COMMAND
            echo Configuring jemalloc && cd ${JEMALLOC_SRC_DIR} && ./autogen.sh && ./configure
            --prefix=${JEMALLOC_INSTALL_DIR} --enable-prof ${JEMALLOC_DISABLE_TLS} # --with-jemalloc-prefix=je_
        BUILD_COMMAND echo Building jemalloc && cd ${JEMALLOC_SRC_DIR} && make install_lib_static install_include
=======
    ExternalProject_Add(jemalloc
        GIT_REPOSITORY https://github.com/jemalloc/jemalloc.git
        GIT_TAG 5.3.0
        PREFIX ${JEMALLOC_PREFIX_DIR}
        CONFIGURE_COMMAND echo Configuring jemalloc
        && cd ${JEMALLOC_SRC_DIR}
        && ./autogen.sh && ./configure
        --prefix=${JEMALLOC_INSTALL_DIR} --enable-prof ${JEMALLOC_DISABLE_TLS} # --with-jemalloc-prefix=je_
        BUILD_COMMAND echo Building jemalloc && cd ${JEMALLOC_SRC_DIR}
        && make install_lib_static install_include
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
        INSTALL_COMMAND ""
        UPDATE_COMMAND ""
    )

<<<<<<< HEAD
    # Create libjemalloc and libjemalloc_pic targets to be used as dependencies
    add_library(libjemalloc STATIC IMPORTED GLOBAL)
    add_library(libjemalloc_pic STATIC IMPORTED GLOBAL)

    set_property(TARGET libjemalloc PROPERTY IMPORTED_LOCATION ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a)
    set_property(TARGET libjemalloc_pic PROPERTY IMPORTED_LOCATION ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc_pic.a)
=======
    # Create libjemalloc and libjemalloc_pic targets to be used as
    # dependencies
    add_library(libjemalloc STATIC IMPORTED GLOBAL)
    add_library(libjemalloc_pic STATIC IMPORTED GLOBAL)

    set_property(TARGET libjemalloc
        PROPERTY IMPORTED_LOCATION
        ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a
    )
    set_property(TARGET libjemalloc_pic
        PROPERTY IMPORTED_LOCATION
        ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc_pic.a
    )
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)

    link_libraries(-Wl,--no-as-needed)
    link_libraries(dl ${libjemalloc_pic})

    # Export the include directory path.
    set(JEMALLOC_SOURCE_DIR ${JEMALLOC_INSTALL_DIR})
    set(JEMALLOC_LIBRARIES ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc_pic.a ${JEMALLOC_INSTALL_DIR}/lib/libjemalloc.a)

    include_directories(${JEMALLOC_SOURCE_DIR}/include)

<<<<<<< HEAD
else ()
    find_path(JEMALLOC_ROOT_DIR NAMES include/jemalloc/jemalloc.h)

    find_library(
        JEMALLOC_LIBRARIES
=======
else()
    find_path(JEMALLOC_ROOT_DIR
        NAMES include/jemalloc/jemalloc.h
    )

    find_library(JEMALLOC_LIBRARIES
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
        NAMES jemalloc
        HINTS ${JEMALLOC_ROOT_DIR}/lib
    )

<<<<<<< HEAD
    find_path(
        JEMALLOC_INCLUDE_DIR
=======
    find_path(JEMALLOC_INCLUDE_DIR
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
        NAMES jemalloc/jemalloc.h
        HINTS ${JEMALLOC_ROOT_DIR}/include
    )

    include(FindPackageHandleStandardArgs)
<<<<<<< HEAD
    find_package_handle_standard_args(JeMalloc DEFAULT_MSG JEMALLOC_LIBRARIES JEMALLOC_INCLUDE_DIR)

    mark_as_advanced(JEMALLOC_ROOT_DIR JEMALLOC_LIBRARIES JEMALLOC_INCLUDE_DIR)
endif ()
=======
    find_package_handle_standard_args(JeMalloc DEFAULT_MSG
        JEMALLOC_LIBRARIES
        JEMALLOC_INCLUDE_DIR
    )

    mark_as_advanced(
        JEMALLOC_ROOT_DIR
        JEMALLOC_LIBRARIES
        JEMALLOC_INCLUDE_DIR
    )
endif()

message("JEMALLOC_LIBRARIES:" ${JEMALLOC_LIBRARIES})
>>>>>>> 8e74662 (Merge pull request #3 from ashvardanian/main)
