# The JNI boundary lives in `java/lib.c` and instantiates nothing: it links the C interface as an archive, so the JAR
# stages one shared object per platform rather than two that have to find each other at load time.

if (TARGET usearch_static)
    set(USEARCH_JAVA_LIBRARY usearch_static)
elseif (TARGET usearch_static_stub)
    set(USEARCH_JAVA_LIBRARY usearch_static_stub)
else ()
    message(FATAL_ERROR "The Java SDK links the C interface; configure with `-DUSEARCH_BUILD_LIB_C=1`.")
endif ()

find_package(JNI REQUIRED)

add_library(usearch_java SHARED lib.c)
usearch_setup_target(usearch_java)

target_link_libraries(usearch_java PRIVATE ${USEARCH_JAVA_LIBRARY})
target_include_directories(usearch_java PRIVATE ${JNI_INCLUDE_DIRS})
set_target_properties(usearch_java PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

if (USEARCH_USE_JEMALLOC)
    find_library(JEMALLOC_LIBRARY jemalloc)
    if (JEMALLOC_LIBRARY)
        target_link_libraries(usearch_java PRIVATE ${JEMALLOC_LIBRARY})
    endif ()
endif ()
