# The LibraryLink boundary lives in `wolfram/lib.c` and instantiates nothing: it links
# `usearch_indexes`, the pre-instantiated grid the C interface is compiled from, exactly as the JNI
# boundary in `java/` and the C library flavors in `c/` do.

if (NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/usr/local/Wolfram/Mathematica")
endif ()

# LibraryLink is a C API declared entirely by one header, so the SDK contributes an include
# directory and no library to link.
add_library(USearchWFM SHARED lib.c)
usearch_setup_target(USearchWFM)
target_include_directories(USearchWFM PRIVATE "${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
target_link_libraries(USearchWFM PRIVATE usearch_indexes)

# Wolfram Language resolves the library by its bare name, with no `lib` prefix.
set_target_properties(USearchWFM PROPERTIES PREFIX "" OUTPUT_NAME "USearchWFM")
