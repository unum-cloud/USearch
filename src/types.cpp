/**
 *  @brief The library-wide probes of the C interface - the twin of `include/usearch/types.h`.
 *  @author Ash Vardanian
 *  @file src/types.cpp
 *
 *  Three functions that take no handle and cannot fail: what version this is, which kernel
 *  families it was compiled with, and which of those the machine in front of it can run. They
 *  live apart from any modality because every modality reports the same answer.
 *
 *  A handle's own `capabilities` is a narrower question - what dispatch picked for the geometry
 *  that handle froze - and each modality answers that one for itself.
 */
#include <cstdio> // `std::snprintf`

#include <usearch/types.h>

#if USEARCH_USE_NUMKONG
#include <numkong/numkong.h>
#endif

namespace {

/** @brief Rendered once on first use, so the pointer the C interface hands out stays valid. */
char const* rendered_version() {
    static char rendered[32] = {0};
    if (!rendered[0])
        std::snprintf(rendered, sizeof(rendered), "%d.%d.%d", USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR,
                      USEARCH_VERSION_PATCH);
    return rendered;
}

#if USEARCH_USE_NUMKONG

/** @brief The kernel families a capability mask can name, newest class last. */
constexpr struct {
    nk_capability_t bit;
    char const* name;
} capability_names_k[] = {
    {nk_cap_serial_k, "serial"},       {nk_cap_neon_k, "neon"},         {nk_cap_neonsdot_k, "neonsdot"},
    {nk_cap_neonbfdot_k, "neonbfdot"}, {nk_cap_neonfhm_k, "neonfhm"},   {nk_cap_neonfp8_k, "neonfp8"},
    {nk_cap_haswell_k, "haswell"},     {nk_cap_skylake_k, "skylake"},   {nk_cap_icelake_k, "icelake"},
    {nk_cap_genoa_k, "genoa"},         {nk_cap_sapphire_k, "sapphire"}, {nk_cap_sapphireamx_k, "sapphireamx"},
};

/** @brief Composes @p mask into @p rendered as a comma-separated list, once. */
char const* render_capabilities(char* rendered, size_t capacity, nk_capability_t mask) {
    if (rendered[0]) return rendered;
    char* cursor = rendered;
    char* const sentinel = rendered + capacity - 1;
    for (auto const& named : capability_names_k) {
        if (!(mask & named.bit)) continue;
        if (cursor != rendered && cursor != sentinel) *cursor++ = ',';
        for (char const* name = named.name; *name && cursor != sentinel;) *cursor++ = *name++;
    }
    *cursor = '\0';
    return rendered;
}

#endif

} // namespace

extern "C" {

USEARCH_EXPORT char const* usearch_version(void) { return rendered_version(); }

USEARCH_EXPORT char const* usearch_capabilities_compiled(void) {
#if USEARCH_USE_NUMKONG
    static char rendered[256] = {0};
    return render_capabilities(rendered, sizeof(rendered), nk_capabilities_compiled());
#else
    return "serial";
#endif
}

USEARCH_EXPORT char const* usearch_capabilities_available(void) {
#if USEARCH_USE_NUMKONG
    static char rendered[256] = {0};
    return render_capabilities(rendered, sizeof(rendered), nk_capabilities_available());
#else
    return "serial";
#endif
}

} // extern "C"
