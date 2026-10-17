# One MPI driver per modality, each calling the C interfaces of USearch and of the MPI it was configured against, so the
# library itself never depends on MPI. Flags parse through the table the benchmarks share.

add_executable(usearch_distributed_dense dense.cpp)
usearch_setup_target(usearch_distributed_dense HOST)
target_include_directories(usearch_distributed_dense PRIVATE ${PROJECT_SOURCE_DIR}/bench)
target_link_libraries(
    usearch_distributed_dense PRIVATE ${USEARCH_PUNNED_LIBRARIES} fmt::fmt forkunion::static MPI::MPI_C
)
