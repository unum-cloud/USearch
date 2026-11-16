# The warning flags and sanitizers USearch's own targets take through `usearch_warnings` and `usearch_sanitizers`.

# Warnings apply to USearch's own targets only, so fetched dependencies build with their own flags.
# Only C and C++ units take them, since nvcc rejects the host spellings.
# The C interface takes named arguments as partially designated structs, leaving the rest zeroed.
add_library(usearch_warnings INTERFACE)
if (MSVC)
    # `/W4` is the level `-Wall -Wextra` reach elsewhere, and C4068 is the unknown pragma the other compilers skip.
    # C4996 flags the portable C library calls MSVC deprecates in favour of its own `_s` spellings.
    target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/W4;/wd4068;/wd4996;/bigobj>")
    # The conformance switches MSVC leaves off by default, which every other supported compiler behaves as.
    target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/Zc:preprocessor;/utf-8>")
    # The C compiler rejects the two that exist only for C++.
    target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:/permissive-;/Zc:__cplusplus>")
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:/WX>")
    endif ()
else ()
    target_compile_options(
        usearch_warnings
        INTERFACE
            "$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic;-Wno-unknown-pragmas;-Wno-missing-field-initializers>"
            "$<$<AND:$<COMPILE_LANGUAGE:C,CXX>,$<CXX_COMPILER_ID:Clang,AppleClang>>:-Wdocumentation>"
    )
    if (USEARCH_WERROR)
        target_compile_options(usearch_warnings INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:-Werror>")
    endif ()
endif ()

# Sanitizers instrument USearch's own targets and the sibling headers they compile.
# Fetched dependencies keep their own flags, and `usearch_siblings.cmake` hands the choice to the siblings' switches.
# The build type alone picks the optimization level, so an instrumented RelWithDebInfo build stays comparable.
# The frame pointer only matters where the units compile, and MSVC links its sanitizer runtime by itself.
add_library(usearch_sanitizers INTERFACE)
if (USEARCH_BUILD_SANITIZERS STREQUAL "address")
    if (MSVC)
        set(usearch_sanitizer_flags_ /fsanitize=address)
    else ()
        set(usearch_sanitizer_flags_ -fsanitize=address,undefined)
    endif ()
elseif (USEARCH_BUILD_SANITIZERS STREQUAL "thread")
    if (MSVC)
        message(FATAL_ERROR "MSVC has no ThreadSanitizer, so USEARCH_BUILD_SANITIZERS=thread needs Clang or GCC")
    endif ()
    set(usearch_sanitizer_flags_ -fsanitize=thread)
elseif (NOT USEARCH_BUILD_SANITIZERS STREQUAL "off")
    message(FATAL_ERROR "USEARCH_BUILD_SANITIZERS is off, address or thread, not '${USEARCH_BUILD_SANITIZERS}'")
endif ()
if (usearch_sanitizer_flags_ AND MSVC)
    target_compile_options(usearch_sanitizers INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:${usearch_sanitizer_flags_}>")
elseif (usearch_sanitizer_flags_)
    target_compile_options(
        usearch_sanitizers INTERFACE "$<$<COMPILE_LANGUAGE:C,CXX>:${usearch_sanitizer_flags_};-fno-omit-frame-pointer>"
    )
    target_link_options(usearch_sanitizers INTERFACE "$<$<LINK_LANGUAGE:C,CXX>:${usearch_sanitizer_flags_}>")
endif ()
