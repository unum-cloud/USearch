# The sibling helper, the switches forwarded to the siblings, and the siblings themselves.

# Sibling libraries build themselves: each ships a CMakeLists.txt handling its own ISA detection and dispatch. They
# arrive as submodules in a recursive clone, and as pinned fetches otherwise - a plain `git clone` of USearch, or a
# `FetchContent` of it from another project, still configures. Pins match the submodule commits. The siblings are the
# engine rather than an option: every header reaches for them unguarded, so there is no build without them and no switch
# pretending otherwise. Include directories are copied onto the interface target, because `usearch_setup_target` reads
# `INTERFACE_INCLUDE_DIRECTORIES` directly - a genex that sees only what this target declares, not what its link
# interface would carry - to keep the C++20 feature requirement off the pure-C targets. Every sibling is a system
# dependency, so its own warnings never fail a USearch build under `USEARCH_WERROR`, and builds only what a USearch
# target links. A defined @p override_, the sibling's `FETCHCONTENT_SOURCE_DIR_<NAME>`, outranks the submodule.
function (usearch_sibling name_ override_ link_target_ repository_ tag_)
    if (TARGET ${link_target_})
        # A parent project that already added the sibling keeps its copy.
    elseif (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT DEFINED ${override_})
        add_subdirectory(${name_} EXCLUDE_FROM_ALL SYSTEM)
    else ()
        message(STATUS "usearch: ${name_} submodule absent, fetching ${tag_}")
        FetchContent_Declare(
            ${name_}
            GIT_REPOSITORY ${repository_}
            GIT_TAG ${tag_}
            EXCLUDE_FROM_ALL SYSTEM
        )
        FetchContent_MakeAvailable(${name_})
    endif ()

    target_link_libraries(usearch_header INTERFACE ${link_target_})

    # Collected for the export set: their interfaces carry a language level and `Threads::Threads`, which the paths
    # copied below do not.
    get_target_property(sibling_target_ ${link_target_} ALIASED_TARGET)
    if (NOT sibling_target_)
        set(sibling_target_ ${link_target_})
    endif ()
    # Each sibling exports its header target as `header`, which would collide with ours under `usearch::`.
    set_target_properties(${sibling_target_} PROPERTIES EXPORT_NAME ${name_}_header)
    set(usearch_sibling_targets_
        ${usearch_sibling_targets_} ${sibling_target_}
        PARENT_SCOPE
    )

    target_include_directories(
        usearch_header SYSTEM INTERFACE $<TARGET_PROPERTY:${link_target_},INTERFACE_INCLUDE_DIRECTORIES>
    )
endfunction ()

# The siblings build for the floor, the build kind and the GPU codes USearch builds for.
foreach (sibling_ IN ITEMS NUMKONG STRINGZILLA FORKUNION SMASHTABLE)
    foreach (switch_ IN ITEMS MARCH MULTIVERSIONED CUDA_ARCH ROCM_ARCH METAL_ARCH)
        set(${sibling_}_${switch_} "${USEARCH_${switch_}}")
    endforeach ()
endforeach ()

# NumKong: the distance kits and tensors, header-only, as every unit inlines the kernels its CPU or GPU class runs.
# ForkUnion: the thread pools, the memory topology, and the ISA-specific waiters and atomic references. StringZilla: the
# string kits behind the sequence modality, from the Levenshtein steps to the overlaps. SmashTable: the Elias-Fano row
# directories of the NVMe indexes.
set(usearch_sibling_targets_)
usearch_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG numkong::header https://github.com/ashvardanian/NumKong
    8870e51d68b11d5a713ce5d647df98d8960f542d
)
usearch_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION forkunion::header https://github.com/ashvardanian/ForkUnion
    ce67be6511ce8cfc4b139b3e7266ec0e5edc372c
)
usearch_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA stringzilla::header https://github.com/ashvardanian/StringZilla
    ff22abca2e1f271f37bd8cb10174bc9dd37ed4e7
)
usearch_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE smashtable::header https://github.com/ashvardanian/SmashTable
    30e349dcd1e582896386d959d4a36badce061787
)
