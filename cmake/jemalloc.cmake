# JeMalloc from the system, linked into every executable and the C library, which `usearch_setup_target` and `c/` do.
find_library(JEMALLOC_LIBRARIES NAMES jemalloc REQUIRED)
