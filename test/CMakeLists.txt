# One suite executable listing the per-family test units, sharded per dtype so the
# instantiation grids build in parallel. Optimization level comes from the configuration:
# measured on these shards, -O1 costs as much as -O3 and only slows the suite down.
add_executable(
    usearch_test
    test.cpp
    ram_queue.cpp
    dense_metric_f64.cpp
    dense_metric_f32.cpp
    dense_metric_bf16.cpp
    dense_metric_f16.cpp
    dense_metric_e5m2.cpp
    dense_metric_e4m3.cpp
    dense_metric_e3m2.cpp
    dense_metric_e2m3.cpp
    dense_metric_i8.cpp
    dense_metric_i4.cpp
    dense_metric_u8.cpp
    dense_metric_u4.cpp
    dense_metric_u1.cpp
    dense_index_f32.cpp
    dense_index_u8.cpp
    dense_index_i8.cpp
    dense_index_bf16.cpp
    dense_index_i4.cpp
)
setup_target(usearch_test)
target_link_libraries(usearch_test PRIVATE fmt::fmt)
add_test(NAME usearch_test COMMAND usearch_test)

# The core tier's proof: types.hpp, ram_queue.hpp & ram_hints.hpp compile with the
# standard library alone - only -Iinclude, no submodule paths, no third-party targets.
add_executable(usearch_test_core core.cpp)
target_include_directories(usearch_test_core PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(usearch_test_core PROPERTIES CXX_STANDARD 20)
add_test(NAME usearch_test_core COMMAND usearch_test_core)

# The CUDA suite skips `setup_target`: its host-compiler flags - `-march=native`,
# `-fmax-errors` - don't survive the CUDA driver, so the target wires the interface
# includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(usearch_test_cuda dense_index.cu)
    target_include_directories(
        usearch_test_cuda PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                  ${PROJECT_SOURCE_DIR}/numkong/include ${PROJECT_SOURCE_DIR}/forkunion/include
    )
    set_target_properties(
        usearch_test_cuda
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
    )
    target_compile_options(usearch_test_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(usearch_test_cuda PRIVATE fmt::fmt CUDA::cuda_driver)
endif ()
