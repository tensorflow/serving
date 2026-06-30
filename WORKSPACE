workspace(name = "tf_serving")

# TODO(b/269515133): We temporarily remove remote_predict from our builds for
# 2.12 due to a breakage caused by
# github.com/tensorflow/tensorflow/commit/6147c03eb9af1e5d2ae155045b33e909ef96944e
# This will be removed in a subsequent release.
local_repository(
    name = "ignore_remote_predict",
    path = "tensorflow_serving/experimental/tensorflow/ops/remote_predict/",
)

# ===== TensorFlow dependency =====
#
# TensorFlow is imported here instead of in tf_serving_workspace() because
# existing automation scripts that bump the TF commit hash expect it here.
#
# To update TensorFlow to a new revision.
# 1. Update the 'git_commit' args below to include the new git hash.
# 2. Get the sha256 hash of the archive with a command such as...
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the 'sha256' arg with the result.
# 3. Request the new archive to be mirrored on mirror.bazel.build for more
#    reliable downloads.
load("//tensorflow_serving:repo.bzl", "tensorflow_http_archive")
tensorflow_http_archive(
    name = "org_tensorflow",
    sha256 = "558416ce432ca229b464fa81512fdb7142e357480c6dfe34cf25b1c3a599d937",
    git_commit = "64b09cf6de68caffa69c5bfcd0e37909db45892d",
    patch = "//third_party/tensorflow:tensorflow.patch",
    patch_cmds = [
        "sed -i '/cc_library = _cc_library/d' tensorflow/core/platform/rules_cc.bzl",
        "echo -e \"\\ndef cc_library_oss(deps=[], **kwargs):\\n    if kwargs.get(\\\"name\\\") == \\\"lib_internal_impl\\\" or \\\"protobuf\\\" in kwargs.get(\\\"name\\\", \\\"\\\"):\\n        _cc_library(deps = deps, **kwargs)\\n        return\\n    if type(deps) == \\\"list\\\":\\n        if \\\"@com_google_protobuf//:protobuf\\\" not in deps:\\n            deps = deps + [\\\"@com_google_protobuf//:protobuf\\\"]\\n    else:\\n        deps = deps + [\\\"@com_google_protobuf//:protobuf\\\"]\\n    _cc_library(deps = deps, **kwargs)\\ncc_library = cc_library_oss\" >> tensorflow/core/platform/rules_cc.bzl",
        "sed -i 's#deps = \\[op_gen\\] + deps#deps = [op_gen] + deps + [clean_dep(\"//tensorflow/core/framework:kernel_shape_util\"), clean_dep(\"//tensorflow/core/framework:full_type_util\")]#' tensorflow/tensorflow.bzl",
        "sed -i '/name = \"kernel_shape_util\",/a \\    visibility = [\"//visibility:public\"],' tensorflow/core/framework/BUILD",
        "echo -e '\\nalias(name = \"tensorflow_libtensorflow_framework\", actual = \"//tensorflow/core:tensorflow\", visibility = [\"//visibility:public\"])' >> BUILD",
        "echo -e '\\nalias(name = \"tensorflow_tf_header_lib\", actual = \"//tensorflow/core:tensorflow\", visibility = [\"//visibility:public\"])' >> BUILD",
        "sed -i '/name = \"env\",/,/deps = \\[/ s#deps = \\[#deps = [\":status\", \":statusor\", \":context\", \":tracing\", \"//xla/tsl/profiler/backends/cpu:threadpool_listener_state\", \"//xla/tsl/platform:byte_order\", #' third_party/xla/xla/tsl/platform/default/BUILD",
        "sed -i '/name = \"tracing\",/,/deps = \\[/ s#deps = \\[#deps = [\"//xla/tsl/platform:logging\", #' third_party/xla/xla/tsl/platform/default/BUILD",
        "sed -i '/name = \"tf_runtime\",/a \\        repo_mapping = {\"@xla\": \"@local_xla\", \"@tsl\": \"@local_tsl\"},' third_party/tf_runtime/workspace.bzl",
        "sed -i '/name = \"error_util\",/,/deps = \\[/ s#deps = \\[#deps = [\"@xla//xla/tsl/concurrency:async_value\", \"@xla//xla/tsl/concurrency:concurrent_vector\", \"@xla//xla/tsl/concurrency:executor\", \"@xla//xla/tsl/concurrency:ref_count\", \"@xla//xla/tsl/util:safe_reinterpret_cast\", \"@tsl//tsl/platform:context\", #' tensorflow/core/tfrt/utils/BUILD",
        "sed -i '/name = \"work_queue_interface\",/,/deps = \\[/ s#deps = \\[#deps = [\"@xla//xla/tsl/concurrency:ref_count\", #' tensorflow/core/tfrt/runtime/BUILD",
        "sed -i '/name = \"execute\",/,/deps = \\[/ s#deps = \\[#deps = [\"@xla//xla/tsl/platform:macros\", \"@xla//xla/tsl/platform:types\", \"@xla//xla/tsl/profiler/utils:no_init\", \"@tsl//tsl/profiler/lib:traceme_encode\", \"@xla//xla/tsl/profiler/utils:traceme_global_flags\", \"@xla//xla/tsl/profiler/backends/cpu:traceme_recorder\", \"@tsl//tsl/platform:bfloat16\", \"@tsl//tsl/platform:ml_dtypes\", \"@tsl//tsl/platform:tstring\", \"@tsl//tsl/platform:cord\", \"@tsl//tsl/platform:refcount\", \"@tsl//tsl/platform:thread_annotations\", \"@tsl//tsl/platform:stringpiece\", \"@xla//xla/tsl/profiler/utils:time_utils\", \"@xla//xla/tsl/profiler/utils:math_utils\", #' tensorflow/core/tfrt/mlrt/interpreter/BUILD",
        "sed -i '/tf_vendored(name = \"xla\",/s/)/, repo_mapping = {\"@xla\": \"@local_xla\", \"@tsl\": \"@local_tsl\"})/' tensorflow/workspace3.bzl",
        "sed -i '/tf_vendored(name = \"tsl\",/s/)/, repo_mapping = {\"@xla\": \"@local_xla\", \"@tsl\": \"@local_tsl\"})/' tensorflow/workspace3.bzl",
        """python3 -c 'import re, glob
for p in glob.glob("third_party/xla/**/BUILD*", recursive=True):
    s = open(p).read(); blocks = s.split("cc_library(");
    for i in range(1, len(blocks)):
        b = blocks[i]; m_th = re.search(r"textual_hdrs\\s*=\\s*(\\[[^\\]]+\\]),?\\n?", b);
        if m_th:
            th = m_th.group(1); b = b.replace(m_th.group(0), ""); m_h = re.search(r"hdrs\\s*=\\s*(\\[[^\\]]+\\])", b);
            if m_h: h = m_h.group(1); merged = h[:-1] + ", " + th[1:]; b = b.replace(m_h.group(0), "hdrs = " + merged);
            else: b = "\\n    hdrs = " + th + "," + b;
            blocks[i] = b
    open(p, "w").write("cc_library(".join(blocks))'""",
        "echo -e '\\ndiff --git a/WORKSPACE b/WORKSPACE\\n--- a/WORKSPACE\\n+++ b/WORKSPACE\\n@@ -184,25 +184,2 @@\\n sass_repositories()\\n \\n-http_archive(\\n-    name = \"xla\",\\n-    patch_args = [\"-p1\"],\\n-    patches = [\\n-        \"//third_party:xla.patch\",\\n-        \"//third_party:xla_add_grpc_cares_darwin_arm64_support.patch\",\\n-    ],\\n-    sha256 = \"ba80ef58f89ca11bc5652e936cf856cdeae91e6b723ce6750e9ce0202cab51ac\",\\n-    strip_prefix = \"xla-f094066398e2c884e994711fd677f68864324614\",\\n-    urls = [\\n-        \"https://github.com/openxla/xla/archive/f094066398e2c884e994711fd677f68864324614.zip\",\\n-    ],\\n-)\\n-\\n-http_archive(\\n-    name = \"tsl\",\\n-    sha256 = \"8cf1e1285c7b1843a7f5f787465c1ef80304b3400ed837870bc76d74ce04f5af\",\\n-    strip_prefix = \"tsl-d71df2f7612583617d359c36243695097dd63726\",\\n-    urls = [\\n-        \"https://github.com/google/tsl/archive/d71df2f7612583617d359c36243695097dd63726.zip\",\\n-    ],\\n-)\\n-\\n load(\"@xla//tools/toolchains/python:python_repo.bzl\", \"python_repository\")' >> third_party/xprof/xprof.patch",
    ],
    repo_mapping = {
        "@local_xla": "@local_xla",
        "@local_tsl": "@local_tsl",
        "@org_tensorflow": "@org_tensorflow",
        "@xla": "@local_xla",
        "@tsl": "@local_tsl",
    },
)

# Import all of TensorFlow Serving's external dependencies.
# Downstream projects (projects importing TensorFlow Serving) need to
# duplicate all code below in their WORKSPACE file in order to also initialize
# those external dependencies.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("7.4.1")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "bazel_skylib",
    sha256 = "97e709db2e97b646263b5c5e83e3b00de48c1ae55b9e421e3b5e3f9467d02a3a",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/1.8.0/bazel-skylib-1.8.0.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.8.0/bazel-skylib-1.8.0.tar.gz",
    ],
)

http_archive(
    name = "rules_cc",
    sha256 = "b8b918a85f9144c01f6cfe0f45e4f2838c7413961a8ff23bc0c6cdf8bb07a3b6",
    strip_prefix = "rules_cc-0.1.5",
    url = "https://github.com/bazelbuild/rules_cc/releases/download/0.1.5/rules_cc-0.1.5.tar.gz",
)

http_archive(
    name = "rules_python",
    sha256 = "8964aa1e7525fea5244ba737458694a057ada1be96a92998a41caa1983562d00",
    strip_prefix = "rules_python-1.8.5",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_python/releases/download/1.8.5/rules_python-1.8.5.tar.gz",
        "https://github.com/bazelbuild/rules_python/releases/download/1.8.5/rules_python-1.8.5.tar.gz",
    ],
    patches = [
        "@rules_ml_toolchain//third_party/rules_python:rules_python_scope.patch",
        "@rules_ml_toolchain//third_party/rules_python:rules_python_freethreaded.patch",
        "@rules_ml_toolchain//third_party/rules_python:rules_python_versions.patch",
        "@rules_ml_toolchain//third_party/rules_python:rules_python_pip_version.patch",
    ],
    patch_args = ["-p1"],
)

# Toolchains for ML projects hermetic builds.
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
http_archive(
    name = "rules_ml_toolchain",
    sha256 = "0b42f693a60c6050d87db1e0a0eaeb84ab3f54191fce094d86334faedc807da0",
    strip_prefix = "rules_ml_toolchain-398d613aea7a4c294da49b79a6d6f3f8732bd84c",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google-ml-infra/rules_ml_toolchain/archive/398d613aea7a4c294da49b79a6d6f3f8732bd84c.tar.gz",
        "https://github.com/google-ml-infra/rules_ml_toolchain/archive/398d613aea7a4c294da49b79a6d6f3f8732bd84c.tar.gz",
    ],
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")
register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")
# register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")
# register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64_cuda")

# Initialize hermetic Python
load("@org_tensorflow//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@org_tensorflow//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    default_python_version = "system",
    requirements = {
        "3.10": "@org_tensorflow//:requirements_lock_3_10.txt",
        "3.11": "@org_tensorflow//:requirements_lock_3_11.txt",
        "3.12": "@org_tensorflow//:requirements_lock_3_12.txt",
        "3.13": "@org_tensorflow//:requirements_lock_3_13.txt",
        "3.14": "@org_tensorflow//:requirements_lock_3_14.txt",
    },
)

load("@org_tensorflow//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@org_tensorflow//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("//tensorflow_serving:repo.bzl", "tf_serving_vendored")

tf_serving_vendored(
    name = "local_xla",
    path = "third_party/xla",
    repo_mapping = {
        "@local_xla": "@local_xla",
        "@local_tsl": "@local_tsl",
        "@org_tensorflow": "@org_tensorflow",
        "@xla": "@local_xla",
        "@tsl": "@local_tsl",
    },
    root = "@org_tensorflow//:unused",
)

tf_serving_vendored(
    name = "local_tsl",
    path = "third_party/xla/third_party/tsl",
    repo_mapping = {
        "@local_xla": "@local_xla",
        "@local_tsl": "@local_tsl",
        "@org_tensorflow": "@org_tensorflow",
        "@xla": "@local_xla",
        "@tsl": "@local_tsl",
    },
    root = "@org_tensorflow//:unused",
)
load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

http_archive(
    name = "rules_shell",
    sha256 = "0d0c56d01c3c40420bf7bf14d73113f8a92fbd9f5cd13205a3b89f72078f0321",
    strip_prefix = "rules_shell-0.1.1",
    urls = [
        "https://github.com/bazelbuild/rules_shell/releases/download/v0.1.1/rules_shell-v0.1.1.tar.gz",
    ],
)

load("@rules_proto//proto:repositories.bzl", "rules_proto_dependencies")

rules_proto_dependencies()

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(name = "tf_wheel_version_suffix")

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)

# nvshmem_configure removed in newer rules_ml_toolchain

