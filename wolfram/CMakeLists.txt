# The LibraryLink boundary lives in `wolfram/lib.c` and instantiates nothing: it links the C interface as an archive, so
# the shared object the Wolfram Language loads is one file rather than two.

set(USEARCH_WOLFRAM_PATH
    "/usr/local/Wolfram/Mathematica"
    CACHE PATH "Mathematica installation whose LibraryLink headers the Wolfram SDK compiles against"
)

# The C interface's archive is declared in `c/`, which the root adds after this directory, so the switch decides rather
# than the target's existence.
if (USEARCH_BUILD_LIB_C)
    set(usearch_wolfram_library_ usearch_static)
elseif (USEARCH_BUILD_LIB_C_STUB)
    set(usearch_wolfram_library_ usearch_static_stub)
else ()
    message(FATAL_ERROR "The Wolfram SDK links the C interface; configure with `-DUSEARCH_BUILD_LIB_C=1`.")
endif ()

# LibraryLink is a C API declared entirely by one header, so the SDK contributes an include directory and no library to
# link.
add_library(usearch_wolfram SHARED lib.c)
# Probed intrinsics use function target attributes, which Clang's builtin modules cannot represent.
set_source_files_properties(lib.c PROPERTIES COMPILE_OPTIONS "$<$<C_COMPILER_ID:Clang,AppleClang>:-fno-modules>")
usearch_setup_target(usearch_wolfram)
target_include_directories(usearch_wolfram SYSTEM PRIVATE "${USEARCH_WOLFRAM_PATH}/SystemFiles/IncludeFiles/C/")
target_link_libraries(usearch_wolfram PRIVATE ${usearch_wolfram_library_})

# The one target without a `lib` prefix: Wolfram finds a library in `LibraryResources/$SystemID` by its bare name.
set_target_properties(
    usearch_wolfram
    PROPERTIES PREFIX ""
               OUTPUT_NAME "usearch_wolfram"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)
