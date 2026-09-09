# The LibraryLink boundary lives in `wolfram/lib.c` and instantiates nothing: it links the C interface as an archive, so
# the shared object the Wolfram Language loads is one file rather than two.

if (NOT WOLFRAM_PATH)
    set(WOLFRAM_PATH "/usr/local/Wolfram/Mathematica")
endif ()

if (TARGET usearch_static)
    set(USEARCH_WOLFRAM_LIBRARY usearch_static)
elseif (TARGET usearch_static_stub)
    set(USEARCH_WOLFRAM_LIBRARY usearch_static_stub)
else ()
    message(FATAL_ERROR "The Wolfram SDK links the C interface; configure with `-DUSEARCH_BUILD_LIB_C=1`.")
endif ()

# LibraryLink is a C API declared entirely by one header, so the SDK contributes an include directory and no library to
# link.
add_library(usearch_wolfram SHARED lib.c)
usearch_setup_target(usearch_wolfram)
target_include_directories(usearch_wolfram PRIVATE "${WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
target_link_libraries(usearch_wolfram PRIVATE ${USEARCH_WOLFRAM_LIBRARY})

# The one target without a `lib` prefix: Wolfram finds a library in `LibraryResources/$SystemID` by its bare name.
set_target_properties(
    usearch_wolfram
    PROPERTIES PREFIX ""
               OUTPUT_NAME "usearch_wolfram"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)
