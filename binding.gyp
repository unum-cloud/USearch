{
    "targets": [
        {
            "target_name": "usearch",
            "sources": ["javascript/lib.cpp", "simsimd/c/lib.c"],
            "cflags": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wpass=transform-warning"
            ],
            "cflags_cc": [
                "-fexceptions",
                "-Wno-unknown-pragmas",
                "-Wno-maybe-uninitialized",
                "-Wpass=transform-warning",
                "-std=c++17",
            ],
            "include_dirs": [
                "<!@(node -p \"require('node-addon-api').include\")",
                "include",
                "fp16/include",
                "simsimd/include",
            ],
            "dependencies": ["<!(node -p \"require('node-addon-api').gyp\")"],
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
                            "USEARCH_USE_SIMSIMD=<!(echo ${USEARCH_USE_SIMSIMD:-1})",
                            "USEARCH_USE_FP16LIB=<!(echo ${USEARCH_USE_FP16LIB:-1})",
                            "SIMSIMD_TARGET_NEON=<!(echo ${SIMSIMD_TARGET_NEON:-1})",
                            "SIMSIMD_TARGET_NEON_BF16=<!(echo ${SIMSIMD_TARGET_NEON_BF16:-0})",
                            "SIMSIMD_TARGET_SVE=<!(echo ${SIMSIMD_TARGET_SVE:-1})",
                            "SIMSIMD_TARGET_SVE_BF16=<!(echo ${SIMSIMD_TARGET_SVE_BF16:-0})",
                            "SIMSIMD_TARGET_HASWELL=<!(echo ${SIMSIMD_TARGET_HASWELL:-1})",
                            "SIMSIMD_TARGET_SKYLAKE=<!(echo ${SIMSIMD_TARGET_SKYLAKE:-1})",
                            "SIMSIMD_TARGET_ICE=<!(echo ${SIMSIMD_TARGET_ICE:-1})",
                            "SIMSIMD_TARGET_GENOA=<!(echo ${SIMSIMD_TARGET_GENOA:-1})",
                            "SIMSIMD_TARGET_SAPPHIRE=<!(echo ${SIMSIMD_TARGET_SAPPHIRE:-1})",
                        ],
                    },
                ],
                [
                    'OS=="mac"',
                    {
                        "defines": [
                            "USEARCH_USE_OPENMP=<!(echo ${USEARCH_USE_OPENMP:-0})",
                            "USEARCH_USE_SIMSIMD=<!(echo ${USEARCH_USE_SIMSIMD:-0})",
                            "USEARCH_USE_FP16LIB=<!(echo ${USEARCH_USE_FP16LIB:-1})",
                            "SIMSIMD_TARGET_NEON=<!(echo ${SIMSIMD_TARGET_NEON:-1})",
                            "SIMSIMD_TARGET_NEON_BF16=<!(echo ${SIMSIMD_TARGET_NEON_BF16:-0})",
                            "SIMSIMD_TARGET_SVE=<!(echo ${SIMSIMD_TARGET_SVE:-0})",
                            "SIMSIMD_TARGET_HASWELL=<!(echo ${SIMSIMD_TARGET_HASWELL:-1})",
                            "SIMSIMD_TARGET_SKYLAKE=<!(echo ${SIMSIMD_TARGET_SKYLAKE:-0})",
                            "SIMSIMD_TARGET_ICE=<!(echo ${SIMSIMD_TARGET_ICE:-0})",
                            "SIMSIMD_TARGET_GENOA=<!(echo ${SIMSIMD_TARGET_GENOA:-0})",
                            "SIMSIMD_TARGET_SAPPHIRE=<!(echo ${SIMSIMD_TARGET_SAPPHIRE:-0})",
                        ],
                    },
                ],
                [
                    'OS=="win"',
                    {
                        "defines": [
                            "USEARCH_USE_OPENMP=0",
                            "USEARCH_USE_SIMSIMD=0",
                            "USEARCH_USE_FP16LIB=1",
                            "SIMSIMD_TARGET_SKYLAKE=0",
                            "SIMSIMD_TARGET_ICE=0",
                            "SIMSIMD_TARGET_SAPPHIRE=0",
                            "SIMSIMD_TARGET_GENOA=0",
                            "SIMSIMD_TARGET_SVE=0",
                            "SIMSIMD_TARGET_HASWELL=0",
                            "SIMSIMD_TARGET_NEON=0",
                            "SIMSIMD_TARGET_NEON_BF16=0",
                        ],
                    },
                ],
            ],
        }
    ]
}
