# The C interface lives in `c/usearch/*.h` and its implementation in `src/`, so this directory holds only the headers
# and the smoke test that proves the two agree. Both library flavors take the punned objects rather than compiling a
# boundary of their own.

if (USEARCH_BUILD_TEST AND USEARCH_BUILD_SHARED)
    add_executable(usearch_test_c test.c)
    usearch_setup_target(usearch_test_c)
    target_link_libraries(usearch_test_c PRIVATE ${USEARCH_PUNNED_LIBRARIES})
    include(CTest)
    enable_testing()
    add_test(NAME usearch_test_c COMMAND usearch_test_c)

    # Export the dynamic symbol table so `backtrace_symbols_fd` can resolve function names when the in-test crash
    # handler fires.
    set_target_properties(usearch_test_c PROPERTIES ENABLE_EXPORTS ON)
endif ()

# Both libraries are always declared and build only for the targets that link them; `USEARCH_BUILD_SHARED` puts them in
# the default build and the install.
if (NOT USEARCH_BUILD_SHARED)
    set(usearch_library_exclusion_ EXCLUDE_FROM_ALL)
endif ()

# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
#
# Both flavors are the punned objects under another name. The shared one links the object libraries and inherits their
# dependencies; the archive takes the objects themselves, since an archive links nothing - and linking a library instead
# would drop every symbol nothing inside referenced, which is the entire C interface.
#
# For the same reason the archive carries its dependencies' objects: one leaving yyjson, NumKong or ForkUnion undefined
# makes every consumer hunt more archives out of the dependency tree to link one of ours.
add_library(usearch_shared SHARED ${usearch_library_exclusion_})
target_link_libraries(usearch_shared PRIVATE ${USEARCH_PUNNED_LIBRARIES})
add_library(
    usearch_static STATIC ${usearch_library_exclusion_} ${USEARCH_PUNNED_OBJECTS} $<TARGET_OBJECTS:yyjson>
                          $<TARGET_OBJECTS:numkong_static> $<TARGET_OBJECTS:forkunion_static>
)
add_library(${PROJECT_NAME}::shared ALIAS usearch_shared)
add_library(${PROJECT_NAME}::static ALIAS usearch_static)
set_target_properties(usearch_shared usearch_static PROPERTIES LINKER_LANGUAGE CXX)

# Set the compile options for the static library
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(usearch_static PRIVATE -static-libstdc++)
    target_link_options(usearch_static PRIVATE -static-libstdc++)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(usearch_static PRIVATE -static)
    target_link_options(usearch_static PRIVATE -static)
endif ()

usearch_setup_target(usearch_shared)
usearch_setup_target(usearch_static)

# The shared flavor brings JeMalloc into the processes that load it, as `usearch_setup_target` does for executables; an
# archive links nothing.
if (USEARCH_WITH_JEMALLOC)
    target_link_libraries(usearch_shared PRIVATE ${USEARCH_JEMALLOC_LIBRARY})
endif ()

# Outside the export set: `USEARCH_INSTALL` governs the CMake config, not whether a built library lands in the prefix.
# One component per flavor, so a packager can take either.
if (USEARCH_BUILD_SHARED)
    install(
        TARGETS usearch_shared
        COMPONENT usearch_shared
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(
        TARGETS usearch_static
        COMPONENT usearch_static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif ()
