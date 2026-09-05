# The JNI boundary of the Java SDK lives in `java/lib.cpp` and instantiates nothing: it links `usearch_indexes`, the
# pre-instantiated grid the C interface is compiled from, exactly as the C library flavors in `c/` do.

find_package(JNI REQUIRED)

add_library(usearch_jni SHARED lib.cpp)
usearch_setup_target(usearch_jni)

target_link_libraries(usearch_jni PRIVATE usearch_indexes)
target_include_directories(usearch_jni PRIVATE ${JNI_INCLUDE_DIRS})
set_target_properties(usearch_jni PROPERTIES OUTPUT_NAME "usearch_jni")

if (USEARCH_USE_JEMALLOC)
    find_library(JEMALLOC_LIBRARY jemalloc)
    if (JEMALLOC_LIBRARY)
        target_link_libraries(usearch_jni PRIVATE ${JEMALLOC_LIBRARY})
    endif ()
endif ()
