# The C interface lives in `include/usearch/*.h` and its implementation in `src/`, so this directory holds only the
# smoke test that proves the two agree. Both library flavors take the punned objects rather than compiling a boundary of
# their own.

if (USEARCH_BUILD_TEST_C)
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

# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_LIB_C)
    # Both flavors are the punned objects under another name. The shared one links the object libraries and inherits
    # their dependencies; the archive takes the objects themselves, since an archive links nothing - and linking a
    # library instead would drop every symbol nothing inside referenced, which is the entire C interface.
    add_library(usearch_shared SHARED)
    target_link_libraries(usearch_shared PRIVATE ${USEARCH_PUNNED_LIBRARIES})
    add_library(usearch_static STATIC ${USEARCH_PUNNED_OBJECTS})
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

endif ()
