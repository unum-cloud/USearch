# USearch for WebAssembly

The WebAssembly build is the C library from `c/usearch.h`, compiled with Emscripten.
`wasm/lib.cpp` only marks that API `EMSCRIPTEN_KEEPALIVE`. There is no second JavaScript or Wolfram binding in this directory.
The Node.js package is documented in `javascript/README.md`.

## Installation

Download `usearch_wasm_<version>.zip` from the [releases page](https://github.com/unum-cloud/USearch/releases).
It contains `usearch.h` and `libusearch_c.*` built with Emscripten 6, NumKong on, OpenMP and jemalloc off.

To rebuild it:

```sh
emcmake cmake -B build_wasm \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSEARCH_BUILD_LIB_C=ON \
  -DUSEARCH_USE_OPENMP=OFF \
  -DUSEARCH_USE_NUMKONG=ON \
  -DUSEARCH_USE_JEMALLOC=OFF
emmake cmake --build build_wasm --config Release
```

## Quickstart

Link your program against that library. The header in the release archive sits next to the library, so include `usearch.h` rather than `usearch/usearch.h`.

```sh
emcc app.c -I. -L. -lusearch_c -o app.js
node app.js
```

```c
#include <usearch.h>

int main(void) {
    usearch_error_t error = NULL;
    usearch_init_options_t opts = {
        .metric_kind = usearch_metric_cos_k,
        .quantization = usearch_scalar_f32_k,
        .dimensions = 3,
    };
    usearch_index_t index = usearch_init(&opts, &error);
    float vector[3] = {0.2f, 0.6f, 0.4f};

    usearch_reserve(index, 10, &error);
    usearch_add(index, 42, vector, usearch_scalar_f32_k, &error);

    usearch_key_t keys[1];
    usearch_distance_t distances[1];
    usearch_search(index, vector, usearch_scalar_f32_k, 1, keys, distances, &error);

    usearch_free(index, &error);
    return error ? 1 : 0;
}
```

## Buffers, not the host filesystem

`usearch_save`, `usearch_load`, and `usearch_view` use the Emscripten filesystem, which is empty unless you mount one.
In a browser, serialize through memory instead:

```c
size_t bytes = usearch_serialized_length(index, &error);
void *buffer = malloc(bytes);
usearch_save_buffer(index, buffer, bytes, &error);
usearch_load_buffer(index, buffer, bytes, &error);
usearch_view_buffer(index, buffer, bytes, &error); /* buffer must outlive the view */
```

Copy `buffer` out with the Emscripten `FS` or `HEAPU8` if the page needs to keep the bytes.
`usearch_view_buffer` does not copy the index, so freeing that allocation while a view is open is a use-after-free.

