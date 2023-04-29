#pragma once
#include "rust/cxx.h"

struct Results;

class Index {
  public:
    Index();
    void add(uint64_t label, rust::Slice<float const> vector) const;
    Results search(rust::Slice<float const> vector, uint64_t count) const;

    uint64_t dim() const;
    uint64_t connectivity() const;
    uint64_t size() const;
    uint64_t capacity() const;

  private:
    class impl;
    std::shared_ptr<impl> impl;
};

std::unique_ptr<Index> new_index();
