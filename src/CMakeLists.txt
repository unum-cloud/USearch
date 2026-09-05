# The dtype shards instantiate the RAM arms once and both libraries link the same objects; the
# dispatch TU only visits the variant, so the CUDA sibling recompiles it alone through nvcc.
add_library(
    usearch_dense_ram_indexes OBJECT
    dense/ram_indexes_bf16.cpp
    dense/ram_indexes_f16.cpp
    dense/ram_indexes_e5m2.cpp
    dense/ram_indexes_e4m3.cpp
    dense/ram_indexes_e3m2.cpp
    dense/ram_indexes_e2m3.cpp
    dense/ram_indexes_i8.cpp
    dense/ram_indexes_i4.cpp
    dense/ram_indexes_u8.cpp
    dense/ram_indexes_u4.cpp
    dense/ram_indexes_u1.cpp
)
usearch_setup_target(usearch_dense_ram_indexes)
target_include_directories(usearch_dense_ram_indexes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_dense_ram_indexes PRIVATE "USEARCH_BUILD_CUDA=0")
target_link_libraries(usearch_dense_ram_indexes PRIVATE yyjson forkunion::static)

add_library(usearch_indexes STATIC types.cpp dense_index.cpp sequence.cpp $<TARGET_OBJECTS:usearch_dense_ram_indexes>)
usearch_setup_target(usearch_indexes)
target_include_directories(usearch_indexes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_indexes PRIVATE "USEARCH_BUILD_CUDA=0")
target_link_libraries(usearch_indexes PRIVATE yyjson forkunion::static stringzilla::stringzilla_header)

# The CUDA library skips `usearch_setup_target`: its host-compiler flags don't survive the CUDA driver,
# so the target wires the interface includes directly.
if (USEARCH_BUILD_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    add_library(
        usearch_indexes_cuda STATIC
        dense_index.cu
        dense/vram_indexes_bf16.cu
        dense/vram_indexes_f16.cu
        dense/vram_indexes_i8.cu
        dense/vram_indexes_i4.cu
        dense/vram_indexes_u8.cu
        dense/vram_indexes_u1.cu
        $<TARGET_OBJECTS:usearch_dense_ram_indexes>
    )
    target_include_directories(
        usearch_indexes_cuda PRIVATE $<TARGET_PROPERTY:${USEARCH_TARGET_NAME},INTERFACE_INCLUDE_DIRECTORIES>
                                     ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_definitions(usearch_indexes_cuda PRIVATE "USEARCH_BUILD_CUDA=1")
    set_target_properties(
        usearch_indexes_cuda
        PROPERTIES CXX_STANDARD 20
                   CUDA_STANDARD 20
                   CUDA_ARCHITECTURES "${USEARCH_CUDA_ARCHITECTURES}"
                   POSITION_INDEPENDENT_CODE ON
    )
    target_compile_options(usearch_indexes_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
    target_link_libraries(usearch_indexes_cuda PRIVATE yyjson CUDA::cuda_driver forkunion::static)
endif ()
