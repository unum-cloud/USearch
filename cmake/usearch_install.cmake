# The package files, the install rules and export set, and the Swift artifact.

# The package files: a pkg-config file, so other tools can find this, and the CMake config with its version file. A
# header-only package fits any architecture, and `ARCH_INDEPENDENT` declares exactly that - no hand-written version
# template needed.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
set(usearch_config_install_dir_ "${CMAKE_INSTALL_DATADIR}/cmake/${PROJECT_NAME}")
set(usearch_targets_export_name_ "${PROJECT_NAME}Targets")
set(usearch_cmake_version_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake")
set(usearch_cmake_project_config_file_ "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pkg-config.pc.in" "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc")
write_basic_package_version_file(
    ${usearch_cmake_version_config_file_}
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion ARCH_INDEPENDENT
)
configure_file(cmake/config.cmake.in ${usearch_cmake_project_config_file_} @ONLY)

# The headers, the package files and the export set, which records the siblings without headers - each owns its own - so
# `usearch::header` names nothing a consumer cannot resolve. Both exports read the set as it stands when called, so they
# trail every target that belongs to it.
if (USEARCH_INSTALL)
    install(DIRECTORY c/usearch include/usearch cpp/usearch DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(FILES ${usearch_cmake_project_config_file_} ${usearch_cmake_version_config_file_}
            DESTINATION ${usearch_config_install_dir_}
    )
    install(
        TARGETS usearch_core usearch_header
        EXPORT ${usearch_targets_export_name_}
        FILE_SET HEADERS
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}.pc" DESTINATION "${CMAKE_INSTALL_DATADIR}/pkgconfig")
    if (usearch_sibling_targets_)
        install(TARGETS ${usearch_sibling_targets_} EXPORT ${usearch_targets_export_name_})
    endif ()
    export(
        TARGETS usearch_core usearch_header ${usearch_sibling_targets_}
        NAMESPACE ${PROJECT_NAME}::
        FILE "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Targets.cmake"
    )
    install(
        EXPORT ${usearch_targets_export_name_}
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${usearch_config_install_dir_}
    )
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
