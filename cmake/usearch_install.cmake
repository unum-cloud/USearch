# The package files, the install rules and export sets, the Swift artifact, and the preview SDK archive.

# The package files: a pkg-config file, so other tools can find this, and the CMake config with its version file. The
# package holds the compiled libraries, so its config lands under `LIBDIR`, one per architecture.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(usearch_config_install_dir_ "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")
set(usearch_cmake_version_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(usearch_cmake_project_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
write_basic_package_version_file(
    ${usearch_cmake_version_config_file_}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)
configure_file(cmake/usearchConfig.cmake.in ${usearch_cmake_project_config_file_} @ONLY)

# The `.pc` spells its prefix against `${pcfiledir}`, so it stays correct after `cmake --install --prefix` moves the
# tree; `file(RELATIVE_PATH)` ends a path to an ancestor with a slash, which the `.` closes. `GNUInstallDirs` returns a
# relative path unless a distro forced an absolute one, and only the former may follow a prefix.
file(RELATIVE_PATH usearch_pc_relative_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
set(usearch_pc_prefix_ "\${pcfiledir}/${usearch_pc_relative_prefix_}.")
if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
    set(usearch_pc_libdir_ "${CMAKE_INSTALL_LIBDIR}")
else ()
    set(usearch_pc_libdir_ "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
endif ()
if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
    set(usearch_pc_includedir_ "${CMAKE_INSTALL_INCLUDEDIR}")
else ()
    set(usearch_pc_includedir_ "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
endif ()
configure_file(cmake/usearch.pc.in "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" @ONLY)

# The headers, the package files and three export sets, one per component. The preview SDK takes the stable C interface,
# its C++ wrappers and `usearchTargets`, the shared library; `usearchStaticTargets` holds the archive. The header-only
# core is a component of its own, with `usearchHeaderTargets`, which records the siblings without headers - each owns
# its own - so `usearch::header` names nothing a consumer cannot resolve. Every export reads its set as it stands when
# called, so they trail every target that belongs to it.
if (USEARCH_INSTALL)
    install(
        DIRECTORY c/usearch cpp/usearch
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT sdk
    )
    install(
        DIRECTORY include/usearch
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT core
    )
    install(
        FILES LICENSE
        DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/usearch
        COMPONENT sdk
    )
    install(
        FILES ${usearch_cmake_project_config_file_} ${usearch_cmake_version_config_file_}
        DESTINATION ${usearch_config_install_dir_}
        COMPONENT sdk
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig"
        COMPONENT sdk
    )
    install(
        TARGETS usearch_core usearch_header
        EXPORT ${PROJECT_NAME}HeaderTargets
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT core
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    if (usearch_sibling_targets_)
        install(
            TARGETS ${usearch_sibling_targets_}
            EXPORT ${PROJECT_NAME}HeaderTargets
            COMPONENT core
        )
    endif ()
    export(
        TARGETS usearch_core usearch_header ${usearch_sibling_targets_}
        NAMESPACE ${PROJECT_NAME}::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}HeaderTargets.cmake"
    )
    install(
        EXPORT ${PROJECT_NAME}HeaderTargets
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${usearch_config_install_dir_}
        COMPONENT core
    )
    if (USEARCH_BUILD_SHARED)
        install(
            EXPORT ${PROJECT_NAME}Targets
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${usearch_config_install_dir_}
            COMPONENT sdk
        )
        install(
            EXPORT ${PROJECT_NAME}StaticTargets
            NAMESPACE ${PROJECT_NAME}::
            DESTINATION ${usearch_config_install_dir_}
            COMPONENT static
        )
    endif ()
endif ()

# The C library as SwiftPM links it: an XCFramework on Apple platforms and an SE-0482 artifact bundle elsewhere, packed
# by NumKong's script beside the C headers and their module map. Each build adds its target to the artifact in
# USEARCH_SWIFT_DIRECTORY, so a release fills one artifact from a build per target.
if (USEARCH_BUILD_SWIFT)
    set(USEARCH_SWIFT_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        CACHE PATH "Where the Swift artifact lands"
    )
    set(USEARCH_SWIFT_SDK
        ""
        CACHE STRING "The Apple SDK the archive targets, like `macosx` or `iphonesimulator`"
    )
    set(USEARCH_SWIFT_TRIPLES
        "${CMAKE_SYSTEM_PROCESSOR}-unknown-linux-gnu"
        CACHE STRING "Swift target triples the artifact bundle serves"
    )
    if (APPLE)
        set(usearch_swift_target_ -D "NUMKONG_SWIFT_SDK=${USEARCH_SWIFT_SDK}" -D
                                  "NUMKONG_SWIFT_ARCHITECTURE=${CMAKE_OSX_ARCHITECTURES}"
        )
    else ()
        set(usearch_swift_target_ -D "NUMKONG_SWIFT_TRIPLES=$<JOIN:${USEARCH_SWIFT_TRIPLES},,>" -D
                                  "NUMKONG_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    set(usearch_swift_headers_ "${PROJECT_BINARY_DIR}/swift_headers")
    add_custom_target(
        usearch_swift ALL
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${PROJECT_SOURCE_DIR}/c/usearch"
                "${usearch_swift_headers_}/usearch"
        COMMAND "${CMAKE_COMMAND}" -E copy "${PROJECT_SOURCE_DIR}/include/module.modulemap" "${usearch_swift_headers_}"
        COMMAND
            "${CMAKE_COMMAND}" -D NUMKONG_SWIFT_MODULE=USearchC -D "NUMKONG_SWIFT_ARCHIVE=$<TARGET_FILE:usearch_static>"
            -D "NUMKONG_SWIFT_HEADERS=${usearch_swift_headers_}" -D "NUMKONG_SWIFT_DIRECTORY=${USEARCH_SWIFT_DIRECTORY}"
            ${usearch_swift_target_} -P "${numkong_SOURCE_DIR}/cmake/numkong_swift_artifact.cmake"
        DEPENDS usearch_static
        VERBATIM
    )
endif ()

# The preview SDK: `cpack --config build_release/CPackConfig.cmake` archives the `sdk` component alone, the stripped
# shared library with the C headers, the C++ wrappers, the CMake package, the `.pc` and the license, as
# `usearch-<version>-<platform>-sdk.tar.gz`. A parent project packages itself.
if (USEARCH_INSTALL AND PROJECT_IS_TOP_LEVEL)
    set(CPACK_PACKAGE_NAME ${PROJECT_NAME})
    if (WIN32)
        set(CPACK_GENERATOR ZIP)
    else ()
        set(CPACK_GENERATOR TGZ)
    endif ()
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
    set(CPACK_COMPONENT_INCLUDE_TOPLEVEL_DIRECTORY ON)
    set(CPACK_COMPONENTS_ALL sdk)
    set(CPACK_STRIP_FILES ON)
    include(CPack)
endif ()
