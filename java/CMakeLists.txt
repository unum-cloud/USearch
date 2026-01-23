# JNI library build configuration for Java bindings

# Find JNI package
find_package(JNI REQUIRED)

# Define JNI library sources
set(USEARCH_JNI_SOURCES cloud/unum/usearch/cloud_unum_usearch_Index.cpp)

# Add SimSIMD source if enabled
if (USEARCH_USE_SIMSIMD)
    list(APPEND USEARCH_JNI_SOURCES ../simsimd/c/lib.c)
endif ()

# Build JNI shared library
add_library(usearch_jni SHARED ${USEARCH_JNI_SOURCES})

# Setup target with common USearch configurations
setup_target(usearch_jni)

# Add JNI include directories
target_include_directories(usearch_jni PRIVATE ${JNI_INCLUDE_DIRS})

# Set output name to match expected library name
set_target_properties(usearch_jni PROPERTIES OUTPUT_NAME "usearch_jni")

# Link libraries if needed
if (USEARCH_USE_JEMALLOC)
    find_library(JEMALLOC_LIBRARY jemalloc)
    if (JEMALLOC_LIBRARY)
        target_link_libraries(usearch_jni PRIVATE ${JEMALLOC_LIBRARY})
    endif ()
endif ()

if (USEARCH_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_link_libraries(usearch_jni PRIVATE OpenMP::OpenMP_CXX)
endif ()
