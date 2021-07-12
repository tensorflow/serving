"""Utilities for the compilation of tensorflow code."""

load("@org_tensorflow//tensorflow:tensorflow.bzl", "check_deps", "tf_binary_additional_srcs", "tf_cc_shared_object", "tf_copts", "tf_custom_op_library_additional_deps")

def _make_search_paths(prefix, search_level):
    return "-rpath,%s/%s/external/org_tensorflow/tensorflow" % (prefix, "/".join([".."] * search_level))

def rpath_linkopts_to_tensorflow(name):
    """Create a rpath linkopts flag to include tensorflow .so's directory."""

    levels_to_root = native.package_name().count("/") + name.count("/")
    return select({
        "@org_tensorflow//tensorflow:macos": [
            "-Wl,%s" % (_make_search_paths("@loader_path", levels_to_root),),
            "-Wl,-rename_section,__TEXT,text_env,__TEXT,__text",
        ],
        "@org_tensorflow//tensorflow:windows": [],
        "//conditions:default": [
            "-Wl,%s" % (_make_search_paths("$$ORIGIN", levels_to_root + 1),),
        ],
    })

def tf_custom_op_library_external(name, srcs = [], deps = [], linkopts = [], copts = [], **kwargs):
    """Helper to build a dynamic library (.so) from the sources containing implementations of custom ops and kernels.

    Similar to "tf_custom_op_library" in tensorflow, but also work for external libraries and windows.
      """

    # Rely on the TF in "pywrap_tensorflow_import_lib".
    deps = deps + tf_custom_op_library_additional_deps()

    check_deps(
        name = name + "_check_deps",
        disallowed_deps = [
            "@org_tensorflow//tensorflow/core:framework",
            "@org_tensorflow//tensorflow/core:lib",
        ],
        deps = deps,
    )

    tf_cc_shared_object(
        name = name,
        srcs = srcs,
        deps = deps,
        framework_so = tf_binary_additional_srcs() + [
            # Rely on the TF in "tensorflow.dll".
            #"@org_tensorflow//tensorflow:tensorflow_dll_import_lib",
            # Rely on the TF in "tensorflow_cc.dll".
            #"@org_tensorflow//tensorflow:tensorflow_cc_dll_import_lib",
        ],
        copts = copts + tf_copts(is_external = True),
        linkopts = linkopts + select({
            "//conditions:default": ["-lm"],
            "@org_tensorflow//tensorflow:windows": [],
            "@org_tensorflow//tensorflow:macos": [],
        }),
        **kwargs
    )
