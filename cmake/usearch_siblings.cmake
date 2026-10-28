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
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${name_}/CMakeLists.txt" AND NOT DEFINED ${override_})
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
    set(usearch_sibling_targets_
        ${usearch_sibling_targets_} ${sibling_target_}
        PARENT_SCOPE
    )

    target_include_directories(
        usearch_header SYSTEM INTERFACE $<TARGET_PROPERTY:${link_target_},INTERFACE_INCLUDE_DIRECTORIES>
    )
endfunction ()

# The siblings tune their own units for the CPU USearch is tuned for.
if (USEARCH_TARGET_ARCH)
    set(NUMKONG_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
    set(STRINGZILLA_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
    set(FORKUNION_TARGET_ARCH "${USEARCH_TARGET_ARCH}")
endif ()

# NumKong: the distance kits and tensors, header-only, as every unit inlines the kernels its CPU or GPU class runs.
# ForkUnion: the thread pools, the memory topology, and the ISA-specific waiters and atomic references. StringZilla: the
# string kits behind the sequence modality, from the Levenshtein steps to the overlaps. SmashTable: the Elias-Fano row
# directories of the NVMe indexes.
set(usearch_sibling_targets_)
usearch_sibling(
    numkong FETCHCONTENT_SOURCE_DIR_NUMKONG numkong::header https://github.com/ashvardanian/NumKong
    3fecca8d625ccc89ba2605e195e621e48446c795
)
usearch_sibling(
    forkunion FETCHCONTENT_SOURCE_DIR_FORKUNION forkunion::header https://github.com/ashvardanian/ForkUnion
    0b9296ebe0dc3886f4f850f0dc97d6ce382ac909
)
usearch_sibling(
    stringzilla FETCHCONTENT_SOURCE_DIR_STRINGZILLA stringzilla::header https://github.com/ashvardanian/StringZilla
    53fb6a67b188561ffcab506ee3d03901716b1b49
)
usearch_sibling(
    smashtable FETCHCONTENT_SOURCE_DIR_SMASHTABLE smashtable::header https://github.com/ashvardanian/SmashTable
    e45080c7ea61a5a9c7fc90ac8b9595b49b341574
)
