/**
 *  @file src/shapes_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The shapes modality's C surface: the handle over the picked shapes variant, and every
 *      exported entry point as one forward apiece.
 *
 *  The variants' bodies compile in the shapes class units, never here: a handle reaches its variant
 *  only through the residency table the ladder picked.
 */
#include "basic_index.cuh"

struct usearch_shapes_index_handle_t {
    using traits_t = shapes_traits_t;
    basic_handle<traits_t> basic;
};

struct usearch_shapes_cursor_handle_t {
    basic_cursor<usearch_shapes_index_handle_t> basic;
};

struct usearch_shapes_refinement_handle_t {
    basic_refinement<usearch_shapes_index_handle_t> basic;
};

extern "C" {

USEARCH_EXPORT void usearch_shapes_init_ram(usearch_shapes_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_shapes_init_vram(usearch_shapes_init_vram_t* arguments) { basic_init_vram(arguments); }
USEARCH_EXPORT void usearch_shapes_load_ram(usearch_shapes_load_ram_t* arguments) {
    basic_load_ram<usearch_shapes_load_ram>(arguments);
}
USEARCH_EXPORT void usearch_shapes_load_vram(usearch_shapes_load_vram_t* arguments) {
    basic_load_vram<usearch_shapes_load_vram>(arguments);
}
USEARCH_EXPORT void usearch_shapes_view_ram(usearch_shapes_view_ram_t* arguments) { basic_view_ram(arguments); }
USEARCH_EXPORT void usearch_shapes_view_nvme(usearch_shapes_view_nvme_t* arguments) {
    basic_view_nvme<usearch_shapes_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_shapes_free(usearch_shapes_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_shapes_reserve(usearch_shapes_reserve_t* arguments) {
    basic_reserve<usearch_shapes_reserve>(arguments);
}
USEARCH_EXPORT void usearch_shapes_clear(usearch_shapes_clear_t* arguments) {
    basic_clear<usearch_shapes_clear>(arguments);
}
USEARCH_EXPORT void usearch_shapes_describe(usearch_shapes_describe_t* arguments) {
    basic_describe<usearch_shapes_describe>(arguments);
}
USEARCH_EXPORT void usearch_shapes_count(usearch_shapes_count_t* arguments) {
    basic_count<usearch_shapes_count>(arguments);
}
USEARCH_EXPORT void usearch_shapes_insert(usearch_shapes_insert_t* arguments) {
    basic_insert<usearch_shapes_insert>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refine(usearch_shapes_refine_t* arguments) {
    basic_refine<usearch_shapes_refine>(arguments);
}
USEARCH_EXPORT void usearch_shapes_health(usearch_shapes_health_t* arguments) {
    basic_health<usearch_shapes_health>(arguments);
}
USEARCH_EXPORT void usearch_shapes_search(usearch_shapes_search_t* arguments) {
    basic_search<usearch_shapes_search>(arguments);
}
USEARCH_EXPORT void usearch_shapes_get(usearch_shapes_get_t* arguments) { basic_get<usearch_shapes_get>(arguments); }
USEARCH_EXPORT void usearch_shapes_cursor_init(usearch_shapes_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_shapes_cursor_next(usearch_shapes_cursor_next_t* arguments) {
    basic_cursor_next<usearch_shapes_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_shapes_cursor_free(usearch_shapes_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_shapes_sample(usearch_shapes_sample_t* arguments) {
    basic_sample<usearch_shapes_sample>(arguments);
}
USEARCH_EXPORT void usearch_shapes_absorb(usearch_shapes_absorb_t* arguments) {
    basic_absorb<usearch_shapes_absorb>(arguments);
}
USEARCH_EXPORT void usearch_shapes_split(usearch_shapes_split_t* arguments) {
    basic_split<usearch_shapes_split>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_init(usearch_shapes_refinement_init_t* arguments) {
    basic_refinement_init<usearch_shapes_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_next(usearch_shapes_refinement_next_t* arguments) {
    basic_refinement_next<usearch_shapes_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_shapes_refinement_finish(usearch_shapes_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_shapes_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_shapes_save(usearch_shapes_save_t* arguments) {
    basic_save<usearch_shapes_save>(arguments);
}
USEARCH_EXPORT void usearch_shapes_metadata_read(usearch_shapes_metadata_read_t* arguments) {
    basic_metadata_read<usearch_shapes_metadata_read, shapes_traits_t>(arguments);
}

} // extern "C"
