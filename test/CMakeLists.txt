# traversal_queue unit test - single TU, no framework, exit-code-driven. Same pattern as
# bench/CMakeLists.txt, except fmt is required here (the test uses fmt::print for diagnostics).

add_executable(test_traversal_queue traversal_queue.cpp)
target_compile_features(test_traversal_queue PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(test_traversal_queue PRIVATE -march=native)
endif ()

target_include_directories(test_traversal_queue PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_traversal_queue PRIVATE fmt::fmt-header-only)

add_test(NAME test_traversal_queue COMMAND test_traversal_queue)
