if (USEARCH_BUILD_TEST_CPP)
    add_executable(test_cpp test.cpp)
    setup_target(test_cpp)
    add_test(NAME test_cpp COMMAND test_cpp)

    target_include_directories(test_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../stringzilla/include)

    # Export the dynamic symbol table so `backtrace_symbols` / `std::stacktrace` can resolve function names when the
    # in-test crash handler fires.
    set_target_properties(test_cpp PROPERTIES ENABLE_EXPORTS ON)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(test_cpp PRIVATE -Wno-vla -Wno-unused-function -Wno-cast-function-type)
    endif ()

    add_executable(inner_volatile_index_test_cpp inner_volatile_index_test.cpp)
    setup_target(inner_volatile_index_test_cpp)
    target_link_libraries(inner_volatile_index_test_cpp PRIVATE fmt::fmt)
    add_test(NAME inner_volatile_index_test_cpp COMMAND inner_volatile_index_test_cpp)

    add_executable(inner_metric_test_cpp inner_metric_test.cpp)
    setup_target(inner_metric_test_cpp)
    target_link_libraries(inner_metric_test_cpp PRIVATE fmt::fmt)
    add_test(NAME inner_metric_test_cpp COMMAND inner_metric_test_cpp)
endif ()

if (USEARCH_BUILD_BENCH_CPP)
    include(clipp)
    add_executable(bench_cpp bench.cpp)
    setup_target(bench_cpp)
    target_link_libraries(bench_cpp PRIVATE clipp)
endif ()
