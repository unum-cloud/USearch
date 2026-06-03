if (USEARCH_BUILD_BENCH_INNER_METRIC)
    add_executable(bench_inner_metric inner_metric.cpp)
    target_compile_features(bench_inner_metric PRIVATE cxx_std_20)

    # `-march=native` so NumKong's `__AVX2__` / `__AVX512F__` auto-detection lights up the
    # `NK_TARGET_*` macros and the chain rows actually pick the SIMD kits. Without it every
    # kernel_chain falls back to serial.
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_inner_metric PRIVATE -march=native)
    endif ()

    target_include_directories(
        bench_inner_metric PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/numkong/include
    )

    # google/benchmark is brought in by numkong/CMakeLists.txt under NK_BUILD_BENCH; fmt by the
    # top-level CMakeLists. Both are required.
    target_link_libraries(bench_inner_metric PRIVATE benchmark fmt::fmt-header-only)
endif ()

# traversal_queue microbench - single-element and four-wide paths on the three containers,
# plus an end-to-end HNSW traversal-step cell. No NumKong / fmt dependency, only google/benchmark.
if (USEARCH_BUILD_BENCH_TRAVERSAL_QUEUE)
    add_executable(bench_traversal_queue traversal_queue.cpp)
    target_compile_features(bench_traversal_queue PRIVATE cxx_std_17)

    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_traversal_queue PRIVATE -march=native)
    endif ()

    target_include_directories(bench_traversal_queue PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(bench_traversal_queue PRIVATE benchmark)
endif ()
