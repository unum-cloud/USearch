# The JNI boundary of the Java SDK lives in `java/lib.c` and instantiates nothing: it links `usearch_c`, the shared
# engine library the C interface is compiled into, which is the very library the Gradle build links against. Naming one
# target on both paths is what keeps a locally built tree and a CI artifact the same shared object.
#
# The brute-force stand-in defines the same symbols a different way, so the suite runs in seconds against the ABI when
# no engine was asked for - configure with `-DUSEARCH_BUILD_LIB_C_STUB=1` for that, and `-DUSEARCH_BUILD_LIB_C=1` for
# the engine itself. The engine wins whenever both are present.

if (TARGET usearch_c)
    set(USEARCH_JNI_LIBRARY usearch_c)
elseif (TARGET usearch_c_stub)
    set(USEARCH_JNI_LIBRARY usearch_c_stub)
else ()
    message(FATAL_ERROR "The Java SDK links the C interface; configure with `-DUSEARCH_BUILD_LIB_C=1`.")
endif ()

find_package(JNI REQUIRED)

add_library(usearch_jni SHARED lib.c)
usearch_setup_target(usearch_jni)

target_link_libraries(usearch_jni PRIVATE ${USEARCH_JNI_LIBRARY})
target_include_directories(usearch_jni PRIVATE ${JNI_INCLUDE_DIRS})
set_target_properties(
    usearch_jni
    PROPERTIES OUTPUT_NAME "usearch_jni"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)

if (USEARCH_USE_JEMALLOC)
    find_library(JEMALLOC_LIBRARY jemalloc)
    if (JEMALLOC_LIBRARY)
        target_link_libraries(usearch_jni PRIVATE ${JEMALLOC_LIBRARY})
    endif ()
endif ()
