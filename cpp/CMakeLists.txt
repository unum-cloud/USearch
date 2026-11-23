# The header-only C++ SDK over the C interface: `cpp/usearch/*.hpp` include only the C headers `usearch::abi` carries
# and the standard library, which the interface target enforces, so a consumer linking `usearch::cpp` gets the archive
# and sees no engine header.
add_library(usearch_cpp INTERFACE)
add_library(usearch::cpp ALIAS usearch_cpp)
set_target_properties(usearch_cpp PROPERTIES EXPORT_NAME cpp)
target_include_directories(
    usearch_cpp INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(usearch_cpp INTERFACE usearch::abi usearch::static)
target_compile_features(usearch_cpp INTERFACE cxx_std_20)

# The smoke test against the engine, linked as the C one is; `usearch_setup_target` adds warnings and sanitizers.
if (USEARCH_BUILD_TEST AND USEARCH_BUILD_SHARED)
    add_executable(usearch_test_cpp test.cpp)
    usearch_setup_target(usearch_test_cpp)
    target_link_libraries(usearch_test_cpp PRIVATE usearch::cpp fmt::fmt ${USEARCH_PUNNED_LIBRARIES})
    add_test(NAME usearch_test_cpp COMMAND usearch_test_cpp)
endif ()

# The same test against the stand-in, which builds without the engine, so an SDK header reaching past `c/` fails to
# compile here. It takes the SDK's headers rather than `usearch::cpp`, which would link the engine's archive.
if (USEARCH_BUILD_TEST AND USEARCH_BUILD_STUB)
    add_executable(usearch_test_cpp_stub test.cpp)
    target_include_directories(usearch_test_cpp_stub PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_compile_features(usearch_test_cpp_stub PRIVATE cxx_std_20)
    target_link_libraries(
        usearch_test_cpp_stub PRIVATE usearch::abi usearch_static_stub fmt::fmt usearch_warnings usearch_sanitizers
    )
    add_test(NAME usearch_test_cpp_stub COMMAND usearch_test_cpp_stub)
endif ()
