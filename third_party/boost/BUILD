# Build Boost library.
# https://github.com/boostorg/boost

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Boost Software License

cc_library(
    name = "boost",
    srcs = glob([
        "libs/*/include/**/*.hpp",
        "libs/*/include/**/*.h",
        "libs/*/*/include/**/*.hpp",
        "libs/*/*/include/**/*.h",
    ]),
    includes = glob(
        [
            "libs/*/include",
            "libs/*/*/include",
        ],
        exclude_directories = 0,
    ),
)
