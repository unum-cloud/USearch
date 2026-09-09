/**
 *  @brief Module glue of the USearch Node-API addon - the classes and functions it publishes,
 *      and the per-environment state its verbs type-check against.
 *  @file javascript/lib.c
 *  @author Ash Vardanian
 *
 *  Only @c <node_api.h> - no C++ wrapper, no exceptions, and nothing of the engine's own
 *  headers beyond its C surface, so one prebuilt shared library serves every Node release
 *  the ABI covers.
 */
#include <stdlib.h> // `malloc`, `free`

#include "lib.h"

#pragma region Registration

static void usearch_js_addon_finalize(napi_env env, void* data, void* hint) {
    usearch_js_addon_t* const addon = (usearch_js_addon_t*)data;
    (void)hint;
    if (!addon) return;
    if (addon->executor_class) napi_delete_reference(env, addon->executor_class);
    if (addon->index_class) napi_delete_reference(env, addon->index_class);
    free(addon);
}

NAPI_MODULE_INIT() {
    static napi_property_descriptor const module_properties[] = {
        {"version", NULL, usearch_js_version, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesCompiled", NULL, usearch_js_capabilities_compiled, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesAvailable", NULL, usearch_js_capabilities_available, NULL, NULL, NULL, napi_default, NULL},
    };

    napi_value const executor_class = usearch_js_executor_define(env);
    if (!executor_class) return NULL;
    napi_value const index_class = usearch_js_index_define(env);
    if (!index_class) return NULL;

    usearch_js_addon_t* const addon = (usearch_js_addon_t*)malloc(sizeof(usearch_js_addon_t));
    if (!addon) {
        usearch_js_throw(env, "out-of-memory", "Could not allocate the addon state");
        return NULL;
    }
    addon->executor_class = NULL;
    addon->index_class = NULL;
    if (napi_create_reference(env, executor_class, 1, &addon->executor_class) != napi_ok ||
        napi_create_reference(env, index_class, 1, &addon->index_class) != napi_ok ||
        napi_set_instance_data(env, addon, usearch_js_addon_finalize, NULL) != napi_ok) {
        usearch_js_addon_finalize(env, addon, NULL);
        usearch_js_throw(env, "internal", "Could not register the addon state");
        return NULL;
    }

    if (napi_define_properties(env, exports, usearch_js_table_size_(module_properties), module_properties) != napi_ok ||
        napi_set_named_property(env, exports, "Executor", executor_class) != napi_ok ||
        napi_set_named_property(env, exports, "CompiledIndex", index_class) != napi_ok) {
        usearch_js_throw(env, "internal", "Could not publish the addon exports");
        return NULL;
    }
    return exports;
}

#pragma endregion Registration
