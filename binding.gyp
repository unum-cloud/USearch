{
    "variables": {
        # Where the CMake build dropped `libusearch_c`; override with
        # `npm_config_usearch_library_dir` or `--usearch_library_dir=...`. Absolute, since gyp
        # rewrites a relative `-L` against its generator output directory.
        "usearch_library_path%": "<!(node -p \"require('path').resolve(process.env.npm_config_usearch_library_dir || 'build/Release')\")",
    },
    "targets": [
        {
            "target_name": "usearch",
            "sources": [
                "javascript/lib.c",
                "javascript/types.c",
                "javascript/dense_index.c",
            ],
            "cflags": [
                "-std=c11",
                "-Wno-unknown-pragmas",
            ],
            "include_dirs": ["include", "javascript"],
            "libraries": [
                "-L<(usearch_library_path)",
                "-lusearch_c",
                "-Wl,-rpath,<(usearch_library_path)",
            ],
            "xcode_settings": {
                "MACOSX_DEPLOYMENT_TARGET": "11.0",
                "OTHER_CFLAGS": ["-std=c11"],
            },
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "AdditionalOptions": ["/Zc:preprocessor"],
                }
            },
        }
    ]
}
