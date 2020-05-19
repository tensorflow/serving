"""Builds ICU library."""

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # Apache 2.0

exports_files([
    "icu4c/LICENSE",
    "icu4j/main/shared/licenses/LICENSE",
])

cc_library(
    name = "headers",
    hdrs = glob(["icu4c/source/common/unicode/*.h"]),
    includes = [
        "icu4c/source/common",
    ],
    deps = [
    ],
)

cc_library(
    name = "common",
    hdrs = glob(["icu4c/source/common/unicode/*.h"]),
    includes = [
        "icu4c/source/common",
    ],
    deps = [
        ":icuuc",
    ],
)

alias(
    name = "nfkc",
    actual = ":common",
)

alias(
    name = "nfkc_cf",
    actual = ":common",
)

cc_library(
    name = "icuuc",
    srcs = glob(
        [
            "icu4c/source/common/*.c",
            "icu4c/source/common/*.cpp",
            "icu4c/source/stubdata/*.cpp",
        ],
    ),
    hdrs = glob([
        "icu4c/source/common/*.h",
    ]),
    copts = [
        "-DU_COMMON_IMPLEMENTATION",
        "-ffunction-sections",
        "-fdata-sections",
    ] + select({
        ":android": [
            "-DU_HAVE_NL_LANGINFO_CODESET=0",
            "-Wno-deprecated-declarations",
        ],
        ":apple": [
            "-Wno-shorten-64-to-32",
            "-Wno-unused-variable",
        ],
        ":windows": [
            "/utf-8",
            "/DLOCALE_ALLOW_NEUTRAL_NAMES=0",
        ],
        "//conditions:default": [],
    }),
    tags = ["requires-rtti"],
    visibility = [
        "//visibility:private",
    ],
    deps = [
        ":headers",
    ],
)

config_setting(
    name = "android",
    values = {"crosstool_top": "//external:android/crosstool"},
)

config_setting(
    name = "apple",
    values = {"cpu": "darwin"},
)

config_setting(
    name = "windows",
    values = {"cpu": "x64_windows"},
)
