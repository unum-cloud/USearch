# The C++ wrappers over the C interface live in `cpp/usearch/*.hpp`, so this directory holds only the headers and the
# smoke test that drives them, linked against the same punned objects as the C one.

if (USEARCH_BUILD_TEST)
    add_executable(usearch_test_cpp test.cpp)
    usearch_setup_target(usearch_test_cpp)
    target_include_directories(usearch_test_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(usearch_test_cpp PRIVATE fmt::fmt ${USEARCH_PUNNED_LIBRARIES})
    add_test(NAME usearch_test_cpp COMMAND usearch_test_cpp)
endif ()
