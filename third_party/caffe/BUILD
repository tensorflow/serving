package(
    default_visibility = ["//visibility:public"],
)

#
# Caffe config settings
#

# enable pycaffe
config_setting(
    name = "caffe_python_layer",
    values = { "define": "caffe_python_layer=ON" },
)
# standard caffe
config_setting(
    name = "use_caffe_bvlc",
    values = { "define": "caffe_flavour=bvlc" },
)

#
# Shims/preludes
#

cc_library(
    name = "openblas_prelude",
    hdrs = ["openblas_prelude.h"],
)

cc_library(
    name = "python_prelude",
    hdrs = ["python_prelude.h"],
)

filegroup(
    name = "pycaffe_overrides",
    srcs = ["__init__.py"]
)
