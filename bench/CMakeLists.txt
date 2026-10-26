# Every benchmark is named after the modality and the family it drives. The dense search and algebra benches and the
# sequence, shapes and geospatial search benches run over the C interface, linking the library as users do, so one
# binary serves every residency, CPU class and GPU class the library compiled.
foreach (name_ IN ITEMS dense_index_search dense_index_algebra sequence_index_search shapes_index_search
                        geospatial_index_search
)
    add_executable(usearch_bench_${name_} ${name_}.cpp)
    usearch_setup_target(usearch_bench_${name_} HOST)
    target_include_directories(usearch_bench_${name_} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools)
    target_link_libraries(
        usearch_bench_${name_} PRIVATE fmt::fmt yyjson forkunion::static stringzilla::header
                                       ${USEARCH_PUNNED_LIBRARIES} ${CMAKE_DL_LIBS}
    )
endforeach ()

# The sparse single-shard probe instantiates the CPU-resident dense index itself, at f32 and bf16 storage and at the
# serial class or the newest one the host flags compile, for what only a benchmark looks inside.
add_executable(usearch_bench_dense_ram_index_search dense_ram_index_search.cpp)
usearch_setup_target(usearch_bench_dense_ram_index_search HOST)
target_include_directories(
    usearch_bench_dense_ram_index_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools
)
target_link_libraries(
    usearch_bench_dense_ram_index_search PRIVATE fmt::fmt yyjson forkunion::static stringzilla::static
                                                 ${USEARCH_PUNNED_LIBRARIES}
)

# The kernel microbench times the distance kits alone, one row per kernel, dtype, capability tier and shape; it links
# the library only for the C names its shared flags parse.
add_executable(usearch_bench_dense_ram_metric dense/ram_metric.cpp)
usearch_setup_target(usearch_bench_dense_ram_metric HOST)
target_include_directories(
    usearch_bench_dense_ram_metric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools
                                           ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(usearch_bench_dense_ram_metric PRIVATE fmt::fmt ${USEARCH_PUNNED_LIBRARIES})
