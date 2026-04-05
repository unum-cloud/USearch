{
    "variables": {
        "numkong_root": "./numkong",
    },
    "includes": ["./numkong/numkong.gypi"],
    "targets": [
        {
            "target_name": "usearch",
            "sources": ["javascript/lib.cpp"],
            "dependencies": [
                "<!(node -p \"require('node-addon-api').gyp\")",
                "numkong_lib",
            ],
            "cflags": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
            ],
            "cflags_cc": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-std=c++17",
            ],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "include",
            ],
            "defines": [
                "USEARCH_USE_NUMKONG=1",
                "NK_DYNAMIC_DISPATCH=1",
            ],
            "xcode_settings": {
                "GCC_ENABLE_CPP_EXCEPTIONS": "YES",
                "CLANG_CXX_LIBRARY": "libc++",
                "MACOSX_DEPLOYMENT_TARGET": "11.0",
                "OTHER_CFLAGS": ["-arch arm64", "-arch x86_64"],
                "OTHER_LDFLAGS": ["-arch arm64", "-arch x86_64"],
            },
            "msvs_settings": {
                "VCCLCompilerTool": {
                    "ExceptionHandling": 1,
                    "AdditionalOptions": ["-std:c++17"],
                }
            },
            "conditions": [
                [
                    'OS=="linux"',
                    {
                        "cflags_cc": [
                            '<!(if [ "$USEARCH_USE_OPENMP" = "1" ]; then echo \'-fopenmp\'; fi)',
                        ],
                        "ldflags": ['<!(if [ "$USEARCH_USE_OPENMP" = "1" ]; then echo \'-lgomp\'; fi)'],
                        "defines": [
                            "USEARCH_USE_OPENMP=<!(echo ${USEARCH_USE_OPENMP:-0})",
                        ],
                    },
                ],
                [
                    'OS=="mac"',
                    {
                        "defines": [
                            "USEARCH_USE_OPENMP=0",
                        ],
                    },
                ],
                [
                    'OS=="win"',
                    {
                        "defines": [
                            "USEARCH_USE_OPENMP=0",
                        ],
                    },
                ],
            ],
        }
    ]
}
