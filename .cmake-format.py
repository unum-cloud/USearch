# Options affecting formatting.
with section("format"):
    # How wide to allow formatted cmake files
    line_width = 120

    # How many spaces to tab for indent
    tab_size = 4

    # If true, separate flow control names from their parentheses with a space
    separate_ctrl_name_with_space = True

    # If true, separate function names from parentheses with a space
    separate_fn_name_with_space = False

    # If a statement is wrapped to more than one line, than dangle the closing
    # parenthesis on its own line.
    dangle_parens = True

# The project's own functions, so a keyword stays on the line of the items it brings.
with section("parse"):
    additional_commands = {
        "usearch_amd_class": {"pargs": "2+"},
        "usearch_apple_class": {"pargs": 3},
        "usearch_cpu_class": {"pargs": "2+", "kwargs": {"OVER": 1}},
        "usearch_cuda_codes": {"pargs": 2},
        "usearch_cuda_names": {"pargs": "1+"},
        "usearch_dense_dtype_sources": {
            "pargs": "1+",
            "kwargs": {
                "BF16": "*",
                "U1": "*",
                "I8": "*",
                "E4M3": "*",
                "F16": "*",
                "U8": "*",
                "E5M2": "*",
                "I4": "*",
                "U4": "*",
                "E3M2": "*",
                "E2M3": "*",
            },
        },
        "usearch_gpu_bundle": {"pargs": "4+", "kwargs": {"DEVICE": 1}},
        "usearch_gpu_host_only": {"pargs": 2},
        "usearch_nvidia_class": {"pargs": "2+"},
        "usearch_object_library": {"pargs": 1, "kwargs": {"SOURCES": "*", "DEFINITIONS": "*", "LIBRARIES": "*"}},
        "usearch_rocm_codes": {"pargs": "1+"},
        "usearch_setup_target": {"pargs": 1, "flags": ["HOST"]},
        "usearch_sibling": {"pargs": 4},
        "usearch_test_vram_shards": {
            "pargs": 2,
            "kwargs": {"ENTRY": 1, "DEVICE": 1, "SOURCES": "*", "DEFINITIONS": "*", "CODES": "*"},
        },
        "usearch_union_isa_flags": {"pargs": "1+"},
        "usearch_variants": {
            "pargs": "3+",
            "kwargs": {"GATE": 1, "DEFINITIONS": "*", "LIBRARIES": "*", "KERNELS": "*"},
        },
    }

with section("markup"):
    # Never reflow hand-authored comment blocks: a file-level docstring can carry multi-line apt,
    # uv or cmake usage that a reflow would restitch into an unreadable run. Disabling markup keeps
    # every comment verbatim, so the docstring survives without a `cmake-format: off` guard.
    enable_markup = False
    first_comment_is_literal = True
