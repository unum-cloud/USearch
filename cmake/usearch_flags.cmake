# The warning flags USearch's own targets take through `usearch_warnings`, and the Debug sanitizers.

# Warnings apply to USearch's own targets only, so fetched dependencies build with their own flags. The C interface
# takes named arguments as partially designated structs, leaving the rest zeroed. CUDA sources take none, since NVCC
# rejects the host spellings.
add_library(usearch_warnings INTERFACE)
if (MSVC)
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE /WX)
    endif ()
else ()
    target_compile_options(
        usearch_warnings
        INTERFACE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wno-conversion;-Wno-unknown-pragmas;-Wno-missing-field-initializers>"
            "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
endif ()

# Debug builds take each sanitizer whose instrumented load the toolchain compiles and links; WASI links no sanitizer
# runtime at all.
set(usearch_sanitizers_ "")
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    include(CheckCXXSourceCompiles)
    include(CMakePushCheckState)
    cmake_push_check_state(RESET)
    foreach (usearch_sanitizer_ address leak alignment undefined)
        set(CMAKE_REQUIRED_FLAGS "-fsanitize=${usearch_sanitizer_}")
        check_cxx_source_compiles(
            "int main(int argc, char **argv) { return *(int *)(void *)argv[0] + argc; }"
            usearch_sanitizer_${usearch_sanitizer_}_compiles_
        )
        if (usearch_sanitizer_${usearch_sanitizer_}_compiles_)
            list(APPEND usearch_sanitizers_ "-fsanitize=${usearch_sanitizer_}")
        endif ()
    endforeach ()
    cmake_pop_check_state()
endif ()
