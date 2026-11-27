# The sibling helper, the switches forwarded to the siblings, and the siblings themselves.

include(FetchContent)

# A sibling builds in place from its submodule when a recursive clone checked one out, or else from the commit that
# submodule pins, fetched, so a plain clone or a `FetchContent` of USearch still configures.
# The caller names the sibling's `FETCHCONTENT_SOURCE_DIR_<NAME>`, so nothing derives it.
# A non-empty one points the sibling at a local checkout, as in a workspace keeping the siblings side by side, and
# outranks the submodule.
# A parent project that already added the sibling keeps its copy.
# Every sibling is a system dependency that builds only what USearch's targets link, so its own warnings never fail a
# build under USEARCH_WERROR.
function (usearch_sibling name_ override_ repository_ tag_)
    if (TARGET ${name_}::header)
        return()
    endif ()
    if (EXISTS "${PROJECT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT ${override_})
        add_subdirectory("${PROJECT_SOURCE_DIR}/${name_}" "${PROJECT_BINARY_DIR}/${name_}" EXCLUDE_FROM_ALL SYSTEM)
        return()
    endif ()
    message(STATUS "usearch: ${name_} from its local checkout, or else fetched at ${tag_}")
    FetchContent_Declare(
        ${name_}
        GIT_REPOSITORY ${repository_}
        GIT_TAG ${tag_}
        EXCLUDE_FROM_ALL SYSTEM
    )
    FetchContent_MakeAvailable(${name_})
endfunction ()

# The siblings build for the floor, the build kind, the sanitizer and the GPU codes USearch builds for, each taking only
# the switches it reads. ThreadSanitizer then sees their barriers, and ForkUnion leaves out the instructions it cannot
# see.
set(NUMKONG_MARCH "${USEARCH_MARCH}")
set(NUMKONG_MULTIVERSIONED "${USEARCH_MULTIVERSIONED}")
set(NUMKONG_BUILD_SANITIZERS "${USEARCH_BUILD_SANITIZERS}")
set(NUMKONG_CUDA_ARCH "${USEARCH_CUDA_ARCH}")
set(NUMKONG_ROCM_ARCH "${USEARCH_ROCM_ARCH}")
set(NUMKONG_METAL_ARCH "${USEARCH_METAL_ARCH}")
set(STRINGZILLA_MARCH "${USEARCH_MARCH}")
set(STRINGZILLA_MULTIVERSIONED "${USEARCH_MULTIVERSIONED}")
set(STRINGZILLA_BUILD_SANITIZERS "${USEARCH_BUILD_SANITIZERS}")
set(STRINGZILLA_CUDA_ARCH "${USEARCH_CUDA_ARCH}")
set(STRINGZILLA_ROCM_ARCH "${USEARCH_ROCM_ARCH}")
set(STRINGZILLA_METAL_ARCH "${USEARCH_METAL_ARCH}")
set(FORKUNION_MARCH "${USEARCH_MARCH}")
set(FORKUNION_MULTIVERSIONED "${USEARCH_MULTIVERSIONED}")
set(FORKUNION_BUILD_SANITIZERS "${USEARCH_BUILD_SANITIZERS}")
set(SMASHTABLE_BUILD_SANITIZERS "${USEARCH_BUILD_SANITIZERS}")
set(SMASHTABLE_CUDA_ARCH "${USEARCH_CUDA_ARCH}")

# Everything below is this project's own: the siblings it builds on, at the commits their submodules pin.

# NumKong: the distance kits and tensors, header-only, as every unit inlines the kernels its CPU or GPU class runs.
# ForkUnion: the thread pools, the memory topology, and the ISA-specific waiters and atomic references. StringZilla: the
# string kits behind the sequence modality, from the sketches to the overlaps. SmashTable: the Elias-Fano row
# directories of the NVMe indexes.
usearch_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG https://github.com/ashvardanian/NumKong
    f85bdd47ca28d03c3c82716a1a1e44c465587928
)
usearch_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION https://github.com/ashvardanian/ForkUnion
    2797304869337c41500e8e2d1ca9bbc278c351c5
)
usearch_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA https://github.com/ashvardanian/StringZilla
    e0bf6b066bd7f966e6eb302458c4f2d863e77f2b
)
usearch_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE https://github.com/ashvardanian/SmashTable
    dd4dfa9147a1957fb302c1f0e4ca89d028094d94
)

# The siblings are what `dense/`, the NVMe and VRAM tiers and the modalities are written against, while `usearch::header`
# stays the dependency-free RAM tier of `basic/` and links none of them. `usearch_setup_target` puts their include
# directories on every compiled target, rather than linking the targets, to keep the C++20 feature requirement off the
# pure-C ones; each `header` carries the C ABI it links. ForkUnion's `cpp/` holds `fu::pool_t`, which the executors own,
# and its archive comes with the engine that links `forkunion::cpp`.
set(usearch_sibling_includes_
    $<TARGET_PROPERTY:numkong::types,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:forkunion::header,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:stringzilla::types,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:smashtable::header,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:forkunion::cpp,INTERFACE_INCLUDE_DIRECTORIES>
)
