# JeMalloc from the system, linked into every executable and the C library, which `usearch_setup_target` and `c/` do.
find_library(USEARCH_JEMALLOC_LIBRARY NAMES jemalloc REQUIRED)
