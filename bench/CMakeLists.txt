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
        usearch_bench_${name_} PRIVATE fmt::fmt yyjson forkunion::cpp stringzilla::types
                                       ${USEARCH_PUNNED_LIBRARIES} ${CMAKE_DL_LIBS}
    )
endforeach ()

# The sparse single-shard probe instantiates the CPU-resident dense index itself, at f32 and bf16 storage and at the
# serial class or the newest one the host flags compile, for what only a benchmark looks inside. Its kits compile
# NumKong header-only, so the dataset streaming, which calls NumKong's C++ SDK, is a unit of its own in the binary.
add_library(usearch_bench_dense_streaming OBJECT dense_streaming.cpp)
usearch_setup_target(usearch_bench_dense_streaming HOST SDK)
target_include_directories(usearch_bench_dense_streaming PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools)
target_link_libraries(usearch_bench_dense_streaming PRIVATE fmt::fmt yyjson forkunion::cpp stringzilla::types)
add_executable(usearch_bench_dense_ram_index_search dense_ram_index_search.cpp)
usearch_setup_target(usearch_bench_dense_ram_index_search HOST)
target_include_directories(
    usearch_bench_dense_ram_index_search PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools
)
target_link_libraries(
    usearch_bench_dense_ram_index_search PRIVATE usearch_bench_dense_streaming fmt::fmt yyjson forkunion::cpp
                                                 stringzilla::types ${USEARCH_PUNNED_LIBRARIES}
)

# The engine reads bench drives the I/O engines themselves over a live monolith on a drive, beside the matching `fio`
# line; it links the library only for the C names its shared flags parse.
add_executable(usearch_bench_nvme_device_reads nvme_device_reads.cpp)
usearch_setup_target(usearch_bench_nvme_device_reads HOST)
target_include_directories(
    usearch_bench_nvme_device_reads PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools
)
target_link_libraries(
    usearch_bench_nvme_device_reads PRIVATE fmt::fmt yyjson forkunion::cpp ${USEARCH_PUNNED_LIBRARIES}
)

# The kernel microbench times the distance kits alone, one row per kernel, dtype, capability tier and shape; it links
# the library only for the C names its shared flags parse. Its NumKong baseline rows call NumKong's C++ SDK, so they
# are a unit of their own beside the kits' header-only one.
add_library(usearch_bench_dense_ram_metric_numkong OBJECT dense/ram_metric_numkong.cpp)
usearch_setup_target(usearch_bench_dense_ram_metric_numkong HOST SDK)
target_include_directories(usearch_bench_dense_ram_metric_numkong PRIVATE ${PROJECT_SOURCE_DIR}/src)
add_executable(usearch_bench_dense_ram_metric dense/ram_metric.cpp)
usearch_setup_target(usearch_bench_dense_ram_metric HOST)
target_include_directories(
    usearch_bench_dense_ram_metric PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/tools
                                           ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(
    usearch_bench_dense_ram_metric PRIVATE usearch_bench_dense_ram_metric_numkong fmt::fmt yyjson
                                           ${USEARCH_PUNNED_LIBRARIES}
)
