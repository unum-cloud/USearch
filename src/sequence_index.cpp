/**
 *  @file src/sequence_index.cpp
 *  @author Ash Vardanian
 *  @date October 13, 2026
 *  @brief The sequence modality's C surface: the handle over the picked sequence variant, and every
 *      exported entry point as one forward apiece.
 *
 *  The variants' bodies compile in the sequence class units, never here: a handle reaches its
 *  variant only through the residency table the ladder picked.
 */
#include "basic_index.cuh"

struct usearch_sequence_index_handle_t {
    using traits_t = sequence_traits_t;
    basic_handle<traits_t> basic;
};

struct usearch_sequence_cursor_handle_t {
    basic_cursor<usearch_sequence_index_handle_t> basic;
};

struct usearch_sequence_refinement_handle_t {
    basic_refinement<usearch_sequence_index_handle_t> basic;
};

extern "C" {

USEARCH_EXPORT char const* usearch_sequence_metric_name(usearch_sequence_metric_t metric) {
    std::string_view const token = token_of(metric);
    // Every token is a string literal, so its data ends in a terminator.
    return token.empty() ? "unknown" : token.data();
}
USEARCH_EXPORT usearch_sequence_metric_t usearch_sequence_metric_from_name(usearch_bytes_view_t name) {
    return sequence_metric_of(as_text(name));
}
USEARCH_EXPORT void usearch_sequence_init_ram(usearch_sequence_init_ram_t* arguments) { basic_init_ram(arguments); }
USEARCH_EXPORT void usearch_sequence_init_vram(usearch_sequence_init_vram_t* arguments) { basic_init_vram(arguments); }
USEARCH_EXPORT void usearch_sequence_load_ram(usearch_sequence_load_ram_t* arguments) {
    basic_load_ram<usearch_sequence_load_ram>(arguments, as_text(arguments->path));
}
USEARCH_EXPORT void usearch_sequence_load_ram_buffer(usearch_sequence_load_ram_buffer_t* arguments) {
    basic_load_ram<usearch_sequence_load_ram_buffer>(arguments, as_bytes(arguments->buffer));
}
USEARCH_EXPORT void usearch_sequence_load_vram(usearch_sequence_load_vram_t* arguments) {
    basic_load_vram<usearch_sequence_load_vram>(arguments);
}
USEARCH_EXPORT void usearch_sequence_view_ram(usearch_sequence_view_ram_t* arguments) { basic_view_ram(arguments); }
USEARCH_EXPORT void usearch_sequence_view_nvme(usearch_sequence_view_nvme_t* arguments) {
    basic_view_nvme<usearch_sequence_view_nvme>(arguments);
}
USEARCH_EXPORT void usearch_sequence_free(usearch_sequence_index_t index) { basic_free(index); }
USEARCH_EXPORT void usearch_sequence_reserve(usearch_sequence_reserve_t* arguments) {
    basic_reserve<usearch_sequence_reserve>(arguments);
}
USEARCH_EXPORT void usearch_sequence_clear(usearch_sequence_clear_t* arguments) {
    basic_clear<usearch_sequence_clear>(arguments);
}
USEARCH_EXPORT void usearch_sequence_describe(usearch_sequence_describe_t* arguments) {
    basic_describe<usearch_sequence_describe>(arguments);
}
USEARCH_EXPORT void usearch_sequence_count(usearch_sequence_count_t* arguments) {
    basic_count<usearch_sequence_count>(arguments);
}
USEARCH_EXPORT void usearch_sequence_insert(usearch_sequence_insert_t* arguments) {
    basic_insert<usearch_sequence_insert>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refine(usearch_sequence_refine_t* arguments) {
    basic_refine<usearch_sequence_refine>(arguments);
}
USEARCH_EXPORT void usearch_sequence_health(usearch_sequence_health_t* arguments) {
    basic_health<usearch_sequence_health>(arguments);
}
USEARCH_EXPORT void usearch_sequence_search(usearch_sequence_search_t* arguments) {
    basic_search<usearch_sequence_search>(arguments);
}
USEARCH_EXPORT void usearch_sequence_get(usearch_sequence_get_t* arguments) {
    basic_get<usearch_sequence_get>(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_init(usearch_sequence_cursor_init_t* arguments) {
    basic_cursor_init(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_next(usearch_sequence_cursor_next_t* arguments) {
    basic_cursor_next<usearch_sequence_cursor_next>(arguments);
}
USEARCH_EXPORT void usearch_sequence_cursor_free(usearch_sequence_cursor_t cursor) { basic_free(cursor); }
USEARCH_EXPORT void usearch_sequence_sample(usearch_sequence_sample_t* arguments) {
    basic_sample<usearch_sequence_sample>(arguments);
}
USEARCH_EXPORT void usearch_sequence_absorb(usearch_sequence_absorb_t* arguments) {
    basic_absorb<usearch_sequence_absorb>(arguments);
}
USEARCH_EXPORT void usearch_sequence_split(usearch_sequence_split_t* arguments) {
    basic_split<usearch_sequence_split>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_init(usearch_sequence_refinement_init_t* arguments) {
    basic_refinement_init<usearch_sequence_refinement_init>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_next(usearch_sequence_refinement_next_t* arguments) {
    basic_refinement_next<usearch_sequence_refinement_next>(arguments);
}
USEARCH_EXPORT void usearch_sequence_refinement_finish(usearch_sequence_refinement_finish_t* arguments) {
    basic_refinement_finish<usearch_sequence_refinement_finish>(arguments);
}
USEARCH_EXPORT void usearch_sequence_save(usearch_sequence_save_t* arguments) {
    basic_save<usearch_sequence_save>(arguments);
}
USEARCH_EXPORT void usearch_sequence_save_buffer(usearch_sequence_save_buffer_t* arguments) {
    basic_save_buffer<usearch_sequence_save_buffer>(arguments);
}
USEARCH_EXPORT void usearch_sequence_metadata_read(usearch_sequence_metadata_read_t* arguments) {
    basic_metadata_read<usearch_sequence_metadata_read, sequence_traits_t>(arguments, as_text(arguments->path));
}
USEARCH_EXPORT void usearch_sequence_metadata_read_buffer(usearch_sequence_metadata_read_buffer_t* arguments) {
    basic_metadata_read<usearch_sequence_metadata_read_buffer, sequence_traits_t>(arguments,
                                                                                  as_bytes(arguments->buffer));
}

} // extern "C"
