/**
 *  @file src/sequence.cpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief The trampoline of the sequence index library: runtime dispatch over the variable-length
 *      sequence indexes.
 *
 *  The sequence modality builds on the baseline tier: StringZilla-backed similarity
 *  measures - Levenshtein, Needleman-Wunsch, Smith-Waterman - over arbitrary-length
 *  values through @c basic_ram_index, with the per-dtype shards joining as the
 *  scoring kits land.
 */
#include <string_view> // `std::string_view` - the witness's checkpoint path

#include <usearch/basic/ram_index.hpp>
#include <usearch/sequence/ram_index.hpp>
#include <usearch/utilities/ram_device.hpp>

namespace us = unum::usearch;
namespace sz = ashvardanian::stringzilla;

/** Instantiates the sequence checkpoint verbs over the default index in both forms - the
 *  caller's bytes and a path this maps itself - so they compile here. */
[[maybe_unused]] static us::status_t sequence_checkpoint_witness(us::sequence_ram_index_t& index,
                                                                 us::sequence_codec_t const& codec,
                                                                 us::forkunion_executor_t& executor,
                                                                 us::bytes_span_t buffer,
                                                                 std::string_view path) noexcept {
    if (us::status_t const saved = us::sequence_save(buffer, index, codec, executor, us::save_config_t {},
                                                     "levenshtein");
        us::failed(saved))
        return saved;
    if (us::failed(us::sequence_metadata(us::bytes_view_t {buffer}).status)) return us::status_t::malformed_k;
    if (us::status_t const loaded = us::sequence_load(us::bytes_view_t {buffer}, index, codec, executor,
                                                      us::load_config_t {});
        us::failed(loaded))
        return loaded;

    if (us::status_t const saved = us::sequence_save(path, index, codec, executor, us::save_config_t {}, "levenshtein");
        us::failed(saved))
        return saved;
    if (us::failed(us::sequence_metadata(path).status)) return us::status_t::malformed_k;
    return us::sequence_load(path, index, codec, executor, us::load_config_t {});
}
