# One benchmark executable: the dtype-free CLI unit beside the per-dtype shards, each
# carrying its slice of the instantiation grid in a parallel LLVM backend.
add_executable(
    usearch_bench
    dense_index.cpp
    dense_index_bf16.cpp
    dense_index_i8.cpp
    dense_index_u8.cpp
    dense_index_i4.cpp
)
setup_target(usearch_bench)
target_include_directories(usearch_bench PRIVATE ${PROJECT_SOURCE_DIR}/test)
target_link_libraries(usearch_bench PRIVATE clipp fmt::fmt forkunion::static stringzilla::stringzilla_header yyjson)

# The CUDA bench skips `setup_target`: its host-compiler flags don't survive the CUDA
# driver, so the target wires the interface includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_executable(usearch_bench_cuda dense_index.cu)
    target_include_directories(
        usearch_bench_cuda PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                   ${PROJECT_SOURCE_DIR}/numkong/include ${PROJECT_SOURCE_DIR}/forkunion/include
                                   ${PROJECT_SOURCE_DIR}/test
    )
    set_target_properties(
        usearch_bench_cuda
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
    )
    target_compile_options(usearch_bench_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(
        usearch_bench_cuda PRIVATE clipp fmt::fmt forkunion::static stringzilla::stringzilla_header yyjson
                                   CUDA::cuda_driver
    )
endif ()
