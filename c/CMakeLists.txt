set(USEARCH_LIB_SOURCES lib.cpp)

if (USEARCH_BUILD_TEST_C)
    add_executable(test_c test.c ${USEARCH_LIB_SOURCES})
    setup_target(test_c)
    include(CTest)
    enable_testing()
    add_test(NAME test_c COMMAND test_c)
endif ()

# This article discusses a better way to allow building either static or shared libraries:
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html
if (USEARCH_BUILD_LIB_C)
    add_library(usearch_c SHARED ${USEARCH_LIB_SOURCES})
    add_library(usearch_static_c STATIC ${USEARCH_LIB_SOURCES})

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
