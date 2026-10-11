/**
 *  @file javascript/lib.c
 *  @author Ash Vardanian
 *  @date September 9, 2026
 *  @brief Module glue of the USearch Node-API addon - the classes and functions it publishes, and
 *      the per-environment state its verbs type-check against.
 *
 *  Only @c <node_api.h> - no C++ wrapper, no exceptions, and nothing of the engine's own
 *  headers beyond its C surface, so one prebuilt shared library serves every Node release
 *  the ABI covers.
 */
#include <stdlib.h> // `calloc`, `free`

#include "lib.h"

#pragma region Registration

static void usearch_js_addon_finalize(napi_env env, void* data, void* hint) {
    usearch_js_addon_t* const addon = (usearch_js_addon_t*)data;
    (void)hint;
    if (!addon) return;
    if (addon->executor_class) napi_delete_reference(env, addon->executor_class);
    if (addon->index_class) napi_delete_reference(env, addon->index_class);
    if (addon->arena_class) napi_delete_reference(env, addon->arena_class);
    if (addon->deferred_class) napi_delete_reference(env, addon->deferred_class);
    if (addon->seal) napi_delete_reference(env, addon->seal);
    free(addon);
}

NAPI_MODULE_INIT() {
    static napi_property_descriptor const module_properties[] = {
        {"version", NULL, usearch_js_version, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesCompiled", NULL, usearch_js_capabilities_compiled, NULL, NULL, NULL, napi_default, NULL},
        {"capabilitiesAvailable", NULL, usearch_js_capabilities_available, NULL, NULL, NULL, napi_default, NULL},
    };

    // The state goes first: defining a class needs nothing, but instantiating one checks the seal.
    usearch_js_addon_t* const addon = (usearch_js_addon_t*)calloc(1, sizeof(usearch_js_addon_t));
    napi_value seal = NULL, scalars = NULL;
    if (!addon) {
        usearch_js_throw(env, usearch_status_out_of_memory_k, "Could not allocate the addon state");
        return NULL;
    }
    if (napi_set_instance_data(env, addon, usearch_js_addon_finalize, NULL) != napi_ok) {
        usearch_js_addon_finalize(env, addon, NULL);
        usearch_js_throw(env, usearch_status_internal_k, "Could not register the addon state");
        return NULL;
    }
    if (napi_create_external(env, addon, NULL, NULL, &seal) != napi_ok ||
        napi_create_reference(env, seal, 1, &addon->seal) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not create the addon seal");
        return NULL;
    }

    napi_value const executor_class = usearch_js_executor_define(env);
    if (!executor_class) return NULL;
    napi_value const index_class = usearch_js_index_define(env);
    if (!index_class) return NULL;
    napi_value const arena_class = usearch_js_arena_define(env);
    if (!arena_class) return NULL;
    napi_value const deferred_class = usearch_js_deferred_define(env);
    if (!deferred_class) return NULL;

    if (napi_create_reference(env, executor_class, 1, &addon->executor_class) != napi_ok ||
        napi_create_reference(env, index_class, 1, &addon->index_class) != napi_ok ||
        napi_create_reference(env, arena_class, 1, &addon->arena_class) != napi_ok ||
        napi_create_reference(env, deferred_class, 1, &addon->deferred_class) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not register the addon classes");
        return NULL;
    }

    if (napi_define_properties(env, exports, usearch_js_table_size_(module_properties), module_properties) != napi_ok ||
        napi_set_named_property(env, exports, "Executor", executor_class) != napi_ok ||
        napi_set_named_property(env, exports, "CompiledIndex", index_class) != napi_ok ||
        napi_set_named_property(env, exports, "Arena", arena_class) != napi_ok ||
        !(scalars = usearch_js_scalars(env)) || napi_set_named_property(env, exports, "scalars", scalars) != napi_ok) {
        usearch_js_throw(env, usearch_status_internal_k, "Could not publish the addon exports");
        return NULL;
    }
    return exports;
}

#pragma endregion Registration
