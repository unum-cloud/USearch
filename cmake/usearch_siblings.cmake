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
# string kits behind the sequence modality, from the Levenshtein steps to the overlaps. SmashTable: the Elias-Fano row
# directories of the NVMe indexes.
usearch_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG https://github.com/ashvardanian/NumKong
    0d4115072efd2f8cb93c1c16e095742a9d5cccdf
)
usearch_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION https://github.com/ashvardanian/ForkUnion
    6bd10ce124b3804719ca297ac61092ac5773d627
)
usearch_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA https://github.com/ashvardanian/StringZilla
    a01d3e34da6049b871e3d647333688e216325f1b
)
usearch_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE https://github.com/ashvardanian/SmashTable
    09caec7b2084dd20a6563c8f0e87448eebf90d43
)

# The siblings are the engine rather than an option: every header reaches for them unguarded, so `usearch_header`
# holds them all. Include directories are copied onto it, because `usearch_setup_target` reads
# `INTERFACE_INCLUDE_DIRECTORIES` directly, a genex that sees only what this target declares, to keep the C++20 feature
# requirement off the pure-C targets. The export set collects them, as their interfaces hold a language level and
# `Threads::Threads` the copied paths do not, and each sibling's `header` export takes a name that cannot collide with
# ours under `usearch::`.
set(usearch_sibling_targets_)
foreach (sibling_ IN ITEMS numkong forkunion stringzilla smashtable)
    target_link_libraries(usearch_header INTERFACE ${sibling_}::header)
    target_include_directories(
        usearch_header SYSTEM INTERFACE $<TARGET_PROPERTY:${sibling_}::header,INTERFACE_INCLUDE_DIRECTORIES>
    )
    get_target_property(sibling_target_ ${sibling_}::header ALIASED_TARGET)
    if (NOT sibling_target_)
        set(sibling_target_ ${sibling_}::header)
    endif ()
    set_target_properties(${sibling_target_} PROPERTIES EXPORT_NAME ${sibling_}_header)
    list(APPEND usearch_sibling_targets_ ${sibling_target_})
endforeach ()
