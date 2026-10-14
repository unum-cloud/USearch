# -----------------------------
# Options effecting formatting.
# -----------------------------
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

# The project's own functions, so a dtype keyword stays on the line of the shards it brings.
with section("parse"):
    additional_commands = {
        name: {
            "pargs": positional,
            "kwargs": {
                dtype: "*" for dtype in ("BF16", "U1", "I8", "E4M3", "F16", "U8", "E5M2", "I4", "U4", "E3M2", "E2M3")
            },
        }
        for name, positional in (("usearch_dense_dtype_sources", 1), ("usearch_dense_vram_bench", 2))
    }
