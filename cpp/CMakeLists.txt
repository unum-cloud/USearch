# The router library every binding links: the dtype-free trampoline beside the per-dtype
# shards, each carrying its own slice of the instantiation grid in a parallel LLVM backend.
add_library(
    usearch_router STATIC
    dense_indexes.cpp
    dense_indexes_bf16.cpp
    dense_indexes_i4.cpp
    dense_indexes_i8.cpp
    dense_indexes_u8.cpp
    sequence_indexes.cpp
)
setup_target(usearch_router)
target_link_libraries(usearch_router PRIVATE forkunion::static stringzilla::stringzilla_header)
