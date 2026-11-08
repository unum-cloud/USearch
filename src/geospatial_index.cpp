/**
 *  @file src/geospatial_index.cpp
 *  @author Ash Vardanian
 *  @date October 14, 2026
 *  @brief The geospatial modality's C surface: the handle over the picked geospatial variant, every
 *      exported entry point as one forward apiece, and the containment walk only this modality
 *      takes.
 *
 *  The variants' bodies compile in the geospatial class units, never here: a handle reaches its
 *  variant only through the residency table the ladder picked.
 */
#include "basic_index.cuh"

struct usearch_geospatial_index_handle_t {
    using traits_t = geospatial_traits_t;
    basic_handle<traits_t> basic;
};

struct usearch_geospatial_cursor_handle_t {
    basic_cursor<usearch_geospatial_index_handle_t> basic;
};

struct usearch_geospatial_refinement_handle_t {
    basic_refinement<usearch_geospatial_index_handle_t> basic;
};

extern "C" {

USEARCH_EXPORT void usearch_geospatial_init_ram(usearch_geospatial_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_geospatial_init_vram(usearch_geospatial_init_vram_t* arguments) {
    basic_init_vram(arguments);
}
USEARCH_EXPORT void usearch_geospatial_load_ram(usearch_geospatial_load_ram_t* arguments) {
    basic_load_ram<usearch_geospatial_load_ram>(arguments, as_text(arguments->path));
}
USEARCH_EXPORT void usearch_geospatial_load_ram_buffer(usearch_geospatial_load_ram_buffer_t* arguments) {
    basic_load_ram<usearch_geospatial_load_ram_buffer>(arguments, as_bytes(arguments->buffer));
}
USEARCH_EXPORT void usearch_geospatial_load_vram(usearch_geospatial_load_vram_t* arguments) {
    basic_load_vram<usearch_geospatial_load_vram>(arguments, as_text(arguments->path));
}
USEARCH_EXPORT void usearch_geospatial_load_vram_buffer(usearch_geospatial_load_vram_buffer_t* arguments) {
    basic_load_vram<usearch_geospatial_load_vram_buffer>(arguments, as_bytes(arguments->buffer));
}
USEARCH_EXPORT void usearch_geospatial_view_ram(usearch_geospatial_view_ram_t* arguments) { basic_view_ram(arguments); }
USEARCH_EXPORT void usearch_geospatial_view_nvme(usearch_geospatial_view_nvme_t* arguments) {
    basic_view_nvme<usearch_geospatial_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_free(usearch_geospatial_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_geospatial_reserve(usearch_geospatial_reserve_t* arguments) {
    basic_reserve<usearch_geospatial_reserve>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_clear(usearch_geospatial_clear_t* arguments) {
    basic_clear<usearch_geospatial_clear>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_describe(usearch_geospatial_describe_t* arguments) {
    basic_describe<usearch_geospatial_describe>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_count(usearch_geospatial_count_t* arguments) {
    basic_count<usearch_geospatial_count>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_insert(usearch_geospatial_insert_t* arguments) {
    basic_insert<usearch_geospatial_insert>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refine(usearch_geospatial_refine_t* arguments) {
    basic_refine<usearch_geospatial_refine>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_health(usearch_geospatial_health_t* arguments) {
    basic_health<usearch_geospatial_health>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_search(usearch_geospatial_search_t* arguments) {
    basic_search<usearch_geospatial_search>(arguments);
}

USEARCH_EXPORT void usearch_geospatial_within(usearch_geospatial_within_t* arguments) {
    basic_within<usearch_geospatial_within>(arguments);
}

USEARCH_EXPORT void usearch_geospatial_get(usearch_geospatial_get_t* arguments) {
    basic_get<usearch_geospatial_get>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_init(usearch_geospatial_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_next(usearch_geospatial_cursor_next_t* arguments) {
    basic_cursor_next<usearch_geospatial_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_cursor_free(usearch_geospatial_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_geospatial_sample(usearch_geospatial_sample_t* arguments) {
    basic_sample<usearch_geospatial_sample>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_absorb(usearch_geospatial_absorb_t* arguments) {
    basic_absorb<usearch_geospatial_absorb>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_split(usearch_geospatial_split_t* arguments) {
    basic_split<usearch_geospatial_split>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_init(usearch_geospatial_refinement_init_t* arguments) {
    basic_refinement_init<usearch_geospatial_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_next(usearch_geospatial_refinement_next_t* arguments) {
    basic_refinement_next<usearch_geospatial_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_refinement_finish(usearch_geospatial_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_geospatial_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_save(usearch_geospatial_save_t* arguments) {
    basic_save<usearch_geospatial_save>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_save_buffer(usearch_geospatial_save_buffer_t* arguments) {
    basic_save_buffer<usearch_geospatial_save_buffer>(arguments);
}
USEARCH_EXPORT void usearch_geospatial_metadata_read(usearch_geospatial_metadata_read_t* arguments) {
    basic_metadata_read<usearch_geospatial_metadata_read, geospatial_traits_t>(arguments, as_text(arguments->path));
}
USEARCH_EXPORT void usearch_geospatial_metadata_read_buffer(usearch_geospatial_metadata_read_buffer_t* arguments) {
    basic_metadata_read<usearch_geospatial_metadata_read_buffer, geospatial_traits_t>(arguments,
                                                                                      as_bytes(arguments->buffer));
}

} // extern "C"
