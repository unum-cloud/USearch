if (USEARCH_BUILD_TEST_CPP)
    add_executable(inner_index_test_cpp inner_index_test.cpp)
    setup_target(inner_index_test_cpp)
    target_link_libraries(inner_index_test_cpp PRIVATE fmt::fmt)
    add_test(NAME inner_index_test_cpp COMMAND inner_index_test_cpp)

    add_executable(inner_metric_test_cpp inner_metric_test.cpp)
    setup_target(inner_metric_test_cpp)
    target_link_libraries(inner_metric_test_cpp PRIVATE fmt::fmt)
    add_test(NAME inner_metric_test_cpp COMMAND inner_metric_test_cpp)
endif ()

if (USEARCH_BUILD_BENCH_CPP)
    add_executable(inner_index_bench_cpp inner_index_bench.cpp)
    setup_target(inner_index_bench_cpp)
    target_link_libraries(
        inner_index_bench_cpp PRIVATE clipp fmt::fmt forkunion::static stringzilla::stringzilla_header yyjson
    )
endif ()

# The CUDA translation units consume the same scenario & phase libraries under nvcc. They skip `setup_target`: its
# host-compiler flags - `-march=native`, `-fmax-errors` - don't survive the CUDA driver, so the stubs wire the interface
# includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(inner_index_test_cu inner_index_test.cu)
    add_executable(inner_index_bench_cu inner_index_bench.cu)
    foreach (cuda_target inner_index_test_cu inner_index_bench_cu)
        target_include_directories(
            ${cuda_target} PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                   ${PROJECT_SOURCE_DIR}/numkong/include ${PROJECT_SOURCE_DIR}/forkunion/include
        )
        set_target_properties(
            ${cuda_target}
            PROPERTIES CXX_STANDARD 20
                       CUDA_STANDARD 20
                       CUDA_ARCHITECTURES "90a"
        )
        target_compile_options(${cuda_target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
        target_link_libraries(${cuda_target} PRIVATE fmt::fmt CUDA::cuda_driver)
    endforeach ()
    target_link_libraries(inner_index_bench_cu PRIVATE clipp forkunion::static stringzilla::stringzilla_header yyjson)
endif ()
