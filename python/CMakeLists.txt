# The CPython boundary lives in `python/*.c` and instantiates nothing: it links the C interface as an archive, so the
# wheel carries one shared object. `scikit-build-core` drives this from `pyproject.toml`.

if (USEARCH_BUILD_STUB)
    set(usearch_python_library_ usearch_static_stub)
else ()
    set(usearch_python_library_ usearch_static)
endif ()

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

# Module glue first, then the shared ingestion, then one file per modality named after the header it mirrors - the same
# stems `rust/` and `c/usearch/` list.
python_add_library(
    usearch_python
    MODULE
    WITH_SOABI
    lib.c
    types.c
    dense_index.c
    sequence_index.c
    shapes_index.c
    geospatial_index.c
)

# `usearch/__init__.py` imports it as `usearch.lib`, so the file name is part of the interface.
set_target_properties(
    usearch_python
    PROPERTIES OUTPUT_NAME "lib"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)

target_include_directories(usearch_python PRIVATE ${PROJECT_SOURCE_DIR}/c ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(usearch_python PRIVATE ${usearch_python_library_})

# CPython's own symbols come from the interpreter, so they are absent at link time and the linker must not insist.
if (APPLE)
    target_link_options(usearch_python PRIVATE -undefined dynamic_lookup)
elseif (MSVC)
    target_link_options(usearch_python PRIVATE /FORCE)
else ()
    target_link_options(usearch_python PRIVATE -Wl,--unresolved-symbols=ignore-all)
endif ()

# Named, or every dependency's own install rules land in the wheel beside the extension.
install(
    TARGETS usearch_python
    DESTINATION usearch
    COMPONENT usearch_python
)
