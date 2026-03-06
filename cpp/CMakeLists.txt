if (USEARCH_BUILD_TEST_CPP)
    add_executable(test_cpp test.cpp)
    setup_target(test_cpp)
    include(CTest)
    enable_testing()
    add_test(NAME test_cpp COMMAND test_cpp)

    target_include_directories(test_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../stringzilla/include)

    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(test_cpp PRIVATE -Wno-vla -Wno-unused-function -Wno-cast-function-type)
    endif ()
endif ()

if (USEARCH_BUILD_BENCH_CPP)
    include(clipp)
    add_executable(bench_cpp bench.cpp)
    setup_target(bench_cpp)
    target_include_directories(bench_cpp PRIVATE ${clipp_SOURCE_DIR}/include)
endif ()

option(USEARCH_BUILD_TEST_CPP_INDEX_GT "Compile index_gt GTest-based unit tests" OFF)

if (USEARCH_BUILD_TEST_CPP_INDEX_GT)
    find_package(Threads REQUIRED)
    include(FetchContent)
    FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.17.0)
    FetchContent_MakeAvailable(googletest)

    add_executable(index_gt_test index_gt_test.cpp)
    target_include_directories(index_gt_test BEFORE PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../fp16/include)
    target_compile_features(index_gt_test PRIVATE cxx_std_23)
    target_link_libraries(index_gt_test PRIVATE Threads::Threads GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(index_gt_test)
endif ()
