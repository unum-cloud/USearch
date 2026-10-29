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

# The objects compile with hidden visibility, so the shared flavor exports the `USEARCH_EXPORT` entry points alone, and
# the static siblings it links, ForkUnion and yyjson, stay inside it too.
set_target_properties(
    usearch_shared
    PROPERTIES C_VISIBILITY_PRESET hidden
               CXX_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
)
if (APPLE)
    target_link_options(usearch_shared PRIVATE "LINKER:-exported_symbol,_usearch_*")
elseif (CMAKE_SYSTEM_NAME MATCHES "Emscripten|WASI")
    # `wasm-ld` has no `--exclude-libs`.
elseif (CMAKE_SYSTEM_NAME MATCHES "^(Linux|Android|FreeBSD)$")
    # The standard library's inline definitions, like `std::bad_variant_access`, keep default visibility under any
    # preset, so only a version script keeps them in.
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/usearch_shared.map" "{ global: usearch_*; local: *; };\n")
    target_link_options(usearch_shared PRIVATE "LINKER:--version-script=${CMAKE_CURRENT_BINARY_DIR}/usearch_shared.map")
elseif (NOT MSVC)
    target_link_options(usearch_shared PRIVATE "LINKER:--exclude-libs,ALL")
endif ()

# The shared flavor brings JeMalloc into the processes that load it, as `usearch_setup_target` does for executables; an
# archive links nothing.
if (USEARCH_WITH_JEMALLOC)
    target_link_libraries(usearch_shared PRIVATE ${USEARCH_JEMALLOC_LIBRARY})
endif ()

# `SOVERSION` carries the major alone, the only break the C ABI admits.
set_target_properties(usearch_shared PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

# Each flavor joins an export set, so `find_package(usearch)` names them `usearch::shared` and `usearch::static`. The
# preview SDK takes the shared one; the archive installs as a component of its own, which the Rust crate links from.
if (USEARCH_BUILD_SHARED)
    set_target_properties(usearch_shared PROPERTIES EXPORT_NAME shared)
    set_target_properties(usearch_static PROPERTIES EXPORT_NAME static)
    install(
        TARGETS usearch_shared
        EXPORT ${PROJECT_NAME}Targets
        COMPONENT sdk
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    install(
        TARGETS usearch_static
        EXPORT ${PROJECT_NAME}StaticTargets
        COMPONENT static
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif ()
