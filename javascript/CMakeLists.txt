# The N-API boundary lives in `javascript/*.c` and instantiates nothing: it links the C interface as an archive, so
# `usearch.node` carries the engine inside it. `cmake-js` drives this from `package.json`.

if (USEARCH_BUILD_STUB)
    set(usearch_node_library_ usearch_static_stub)
else ()
    set(usearch_node_library_ usearch_static)
endif ()

if (NOT CMAKE_JS_INC)
    message(FATAL_ERROR "The Node addon is built by `cmake-js`, which supplies the headers and the import library.")
endif ()

# `CMAKE_JS_SRC` is the Windows delay-load hook and empty everywhere else.
add_library(usearch_node SHARED lib.c types.c dense_index.c ${CMAKE_JS_SRC})

# The loader resolves the addon by file name, so the `.node` suffix and the bare stem are part of the interface rather
# than a platform convention.
set_target_properties(
    usearch_node
    PROPERTIES PREFIX ""
               SUFFIX ".node"
               OUTPUT_NAME "usearch"
               C_STANDARD 11
               C_STANDARD_REQUIRED ON
)

target_include_directories(usearch_node PRIVATE ${CMAKE_JS_INC} ${PROJECT_SOURCE_DIR}/c ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(usearch_node PRIVATE NAPI_VERSION=8)
target_link_libraries(usearch_node PRIVATE ${CMAKE_JS_LIB} ${usearch_node_library_})

if (MSVC)
    target_compile_options(usearch_node PRIVATE /Zc:preprocessor)
endif ()
