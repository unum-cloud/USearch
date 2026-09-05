# The brute-force stand-in for the C interface: a separate library defining the same symbols a different way, so any
# consumer - a binding suite, a downstream SDK, a bisect - runs in seconds against the ABI instead of waiting on a build
# of the engine. It needs the headers and nothing else - no submodules, no instantiation grid, no engine sources - so it
# skips `setup_target` deliberately and sets by hand the few properties consumers depend on. The one runtime it does
# resolve is the platform's threads, since the stand-ins keep the headers' concurrency promises with a
# `std::shared_mutex` rather than a comment.
#
# The file names match what `c/` produces, because the SDKs hardcode them: `libusearch_c` in the C# `NativeMethods`,
# `-lusearch_c` in the Go directives, `usearch_c` in `Package.swift`. They land in a directory of their own, so a suite
# chooses an implementation with one `-L` and the two libraries can never shadow one another.
set(USEARCH_STUB_SOURCES types.cpp dense.cpp sequence.cpp matrix.cpp geospatial.cpp)

find_package(Threads REQUIRED)

add_library(usearch_c_stub SHARED ${USEARCH_STUB_SOURCES})
add_library(usearch_static_c_stub STATIC ${USEARCH_STUB_SOURCES})
add_library(${PROJECT_NAME}::c_stub ALIAS usearch_c_stub)

foreach (TARGET_NAME usearch_c_stub usearch_static_c_stub)
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TARGET_NAME} PRIVATE Threads::Threads)
    set_target_properties(
        ${TARGET_NAME}
        PROPERTIES PREFIX "lib"
                   POSITION_INDEPENDENT_CODE ON
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
                   ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
                   RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/stub"
    )
endforeach ()

set_target_properties(usearch_c_stub PROPERTIES OUTPUT_NAME "usearch_c")
set_target_properties(usearch_static_c_stub PROPERTIES OUTPUT_NAME "usearch_static_c")

# A stand-in is not part of the published package, so it stays out of the export set and installs under a subdirectory
# of its own, where it cannot shadow the engine's library.
if (USEARCH_INSTALL)
    install(
        TARGETS usearch_c_stub usearch_static_c_stub
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/usearch-stub
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/usearch-stub
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}/usearch-stub
    )
endif ()
