# The C interface lives in `include/usearch/*.h` and its implementation in `src/`, so this
# directory holds only the smoke test that proves the two agree. Both library flavors link
# `usearch_indexes` rather than compiling a boundary of their own.

if (USEARCH_BUILD_TEST_C)
    add_executable(test_c test.c)
    setup_target(test_c)
    target_link_libraries(test_c PRIVATE usearch_indexes)
    include(CTest)
    enable_testing()
    add_test(NAME test_c COMMAND test_c)

    # Export the dynamic symbol table so `backtrace_symbols_fd` can resolve
    # function names when the in-test crash handler fires.
    set_target_properties(test_c PROPERTIES ENABLE_EXPORTS ON)
endif ()

# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_LIB_C)
    # Both flavors are the instantiation grid under another name, so they take its objects
    # directly. Linking the archive instead would drop every symbol nothing inside it
    # referenced - which is the entire C interface. CMake 3.21+ allows `TARGET_OBJECTS` on a
    # static library, and the project floor is 3.28.
    add_library(usearch_c SHARED $<TARGET_OBJECTS:usearch_indexes>)
    add_library(usearch_static_c STATIC $<TARGET_OBJECTS:usearch_indexes>)
    set_target_properties(usearch_c usearch_static_c PROPERTIES LINKER_LANGUAGE CXX)

    # Set the compile options for the static library
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(usearch_static_c PRIVATE -static-libstdc++)
        target_link_options(usearch_static_c PRIVATE -static-libstdc++)
    elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(usearch_static_c PRIVATE -static)
        target_link_options(usearch_static_c PRIVATE -static)
    endif ()

    setup_target(usearch_c)
    setup_target(usearch_static_c)

endif ()
