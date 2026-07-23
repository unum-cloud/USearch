/**
 *  @brief The dtype-free trampoline of the dense router library: runtime dispatch over
 *      the extern-declared per-dtype shards.
 *  @author Ash Vardanian
 *  @file cpp/dense_indexes.cpp
 *  @date July 17, 2026
 *
 *  This translation unit is the meeting point the language bindings link: it lowers a
 *  runtime (dtype, kernel) request onto the pre-instantiated shards - one translation
 *  unit per dtype, so the heavy instantiation grids build in parallel LLVM backends -
 *  and never instantiates an index template itself. The dispatch surface grows with the
 *  router facade in @c include/usearch/router.hpp; until then the shards carry the
 *  instantiations alone.
 */
#include <usearch/dense/ram_index.hpp>

namespace us = unum::usearch;
namespace fu = ashvardanian::forkunion;
namespace nk = ashvardanian::numkong;
