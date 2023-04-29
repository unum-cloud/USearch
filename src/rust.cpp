#include "rust.h"

#include "advanced.hpp"

using namespace unum::usearch;
using namespace unum;

class Index::impl {
    friend Index;
    downcasting_index_t native_;
};

Index::Index() : impl(new class Index::impl) {}

void Index::add(uint64_t label, rust::Slice<float const> vector) const {}
Results Index::search(rust::Slice<float const> vector, uint64_t count) const { return {}; }

uint64_t Index::dim() const { return impl->native_.dim(); }
uint64_t Index::connectivity() const { return impl->native_.connectivity(); }
uint64_t Index::size() const { return impl->native_.size(); }
uint64_t Index::capacity() const { return impl->native_.capacity(); }

std::unique_ptr<Index> new_index() { return std::make_unique<Index>(); }
