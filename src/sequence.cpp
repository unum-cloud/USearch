/**
 *  @file src/sequence.cpp
 *  @author Ash Vardanian
 *  @date July 17, 2026
 *  @brief The trampoline of the sequence index library: runtime dispatch over the variable-length
 *      sequence indexes.
 *
 *  The sequence modality builds on the baseline tier: StringZilla-backed similarity
 *  measures - Levenshtein, Needleman-Wunsch, Smith-Waterman - over arbitrary-length
 *  values through @c basic::ram_index, with the per-dtype shards joining as the
 *  scoring kits land.
 */
#include <string_view> // `std::string_view` - the witness's checkpoint path

#include <usearch/basic/ram_index.hpp>
#include <usearch/sequence/ram_index.hpp>
#include <usearch/utilities/ram_device.hpp>

namespace sequence = unum::usearch::sequence;
namespace sz = ashvardanian::stringzilla;

/** Instantiates the sequence checkpoint verbs over the default index in both forms - the
 *  caller's bytes and a path this maps itself - so they compile here. */
[[maybe_unused]] static sequence::status_t sequence_checkpoint_witness(sequence::ram_index_t& index,
                                                                       sequence::codec_t const& codec,
                                                                       sequence::forkunion_executor_t& executor,
                                                                       sequence::bytes_span_t buffer,
                                                                       std::string_view path) noexcept {
    if (sequence::status_t const saved = sequence::save(buffer, index, codec, executor, sequence::save_config_t {},
                                                        "levenshtein");
        sequence::failed(saved))
        return saved;
    if (sequence::failed(sequence::metadata(sequence::bytes_view_t {buffer}).status))
        return sequence::status_t::malformed_k;
    if (sequence::status_t const loaded = sequence::load(sequence::bytes_view_t {buffer}, index, codec, executor,
                                                         sequence::load_config_t {});
        sequence::failed(loaded))
        return loaded;

    if (sequence::status_t const saved = sequence::save(path, index, codec, executor, sequence::save_config_t {},
                                                        "levenshtein");
        sequence::failed(saved))
        return saved;
    if (sequence::failed(sequence::metadata(path).status)) return sequence::status_t::malformed_k;
    return sequence::load(path, index, codec, executor, sequence::load_config_t {});
}
