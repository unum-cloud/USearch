# The JNI boundary of the Java SDK lives in `java/lib.cpp` and instantiates nothing: it links the punned objects the C
# interface is compiled from, exactly as the C library flavors in `c/` do.

find_package(JNI REQUIRED)

add_library(usearch_java SHARED lib.cpp)
usearch_setup_target(usearch_java)

target_link_libraries(usearch_java PRIVATE ${USEARCH_PUNNED_LIBRARIES})
target_include_directories(usearch_java PRIVATE ${JNI_INCLUDE_DIRS})

if (USEARCH_USE_JEMALLOC)
    find_library(JEMALLOC_LIBRARY jemalloc)
    if (JEMALLOC_LIBRARY)
        target_link_libraries(usearch_java PRIVATE ${JEMALLOC_LIBRARY})
    endif ()
endif ()
