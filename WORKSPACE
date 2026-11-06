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
    # Pinned to TF 2.22.0-rc0 release commit; git_commit is placed before sha256 to
    # prevent ml-serving-oss-releaser (update_workspace_commit.py) from
    # overwriting this pin with continuous_tf_head.
    git_commit = "160ece53b82229862412699ff770232d2c3e1313",
    sha256 = "2edbc9f88eae860371cb19ec785315ba3eed90d0ec8db35b1d666f155d34d213",
    patch = "//third_party/tensorflow:tensorflow.patch",
    patch_cmds = [
        """python3 -c 'import re, glob
for p in glob.glob("third_party/xla/**/BUILD*", recursive=True):
    s = open(p).read(); parts = s.split("cc_library("); new_parts = [parts[0]]
    for part in parts[1:]:
        depth = 1; idx = 0
        while idx < len(part) and depth > 0:
            if part[idx] == "(": depth += 1
            elif part[idx] == ")": depth -= 1
            idx += 1
        b = part[:idx]; rest = part[idx:]
        m_th = re.search(r"textual_hdrs\\s*=\\s*(\\[[^\\]]*\\]),?\\s*\\n?", b, re.DOTALL)
        if m_th:
            th = m_th.group(1); b_no = b[:m_th.start()] + b[m_th.end():]; m_h = re.search(r"hdrs\\s*=\\s*(\\[[^\\]]*\\])", b_no, re.DOTALL)
            if m_h:
                th_items = [x.strip() for x in th.lstrip("[").rstrip("]").split(",") if x.strip()]
                h_items = [x.strip() for x in m_h.group(1).lstrip("[").rstrip("]").split(",") if x.strip()]
                merged = h_items + [x for x in th_items if x not in h_items]
                b = b_no[:m_h.start()] + "hdrs = [" + ", ".join(merged) + "]" + b_no[m_h.end():]
            else: b = b[:m_th.start()] + "hdrs = " + th + ",\\n" + b[m_th.end():]
        new_parts.append(b + rest)
    open(p, "w").write("cc_library(".join(new_parts))'""",
        "find . -name \"cub_scan_kernel_cuda_impl.cu.cc\" -exec python3 -c 'import sys, re; f=sys.argv[1]; c=open(f).read(); c=re.sub(r\"using MaxPolicyT = typename cub::detail::scan::policy_hub<.*?>::MaxPolicy;\", \"using MaxPolicyT = typename cub::DeviceScanPolicy<T, ScanOpT>::MaxPolicy;\", c, flags=re.DOTALL); c=c.replace(\"auto* kernel = BlockScanKernel<T, ScanOpT>;\", \"void (*kernel)(const T*, T*, int64_t) = BlockScanKernel<T, ScanOpT>;\"); open(f, \"w\").write(c)' {} \\;",
        "find tensorflow third_party/xla -name 'workspace*.bzl' -exec sed -i 's/native.register_/# native.register_/g' {} +",
        "echo -e '\\ndiff --git a/WORKSPACE b/WORKSPACE\\n--- a/WORKSPACE\\n+++ b/WORKSPACE\\n@@ -184,25 +184,2 @@\\n sass_repositories()\\n \\n-http_archive(\\n-    name = \"xla\",\\n-    patch_args = [\"-p1\"],\\n-    patches = [\\n-        \"//third_party:xla.patch\",\\n-        \"//third_party:xla_add_grpc_cares_darwin_arm64_support.patch\",\\n-    ],\\n-    sha256 = \"ba80ef58f89ca11bc5652e936cf856cdeae91e6b723ce6750e9ce0202cab51ac\",\\n-    strip_prefix = \"xla-f094066398e2c884e994711fd677f68864324614\",\\n-    urls = [\\n-        \"https://github.com/openxla/xla/archive/f094066398e2c884e994711fd677f68864324614.zip\",\\n-    ],\\n-)\\n-\\n-http_archive(\\n-    name = \"tsl\",\\n-    sha256 = \"8cf1e1285c7b1843a7f5f787465c1ef80304b3400ed837870bc76d74ce04f5af\",\\n-    strip_prefix = \"tsl-d71df2f7612583617d359c36243695097dd63726\",\\n-    urls = [\\n-        \"https://github.com/google/tsl/archive/d71df2f7612583617d359c36243695097dd63726.zip\",\\n-    ],\\n-)\\n-\\n load(\"@xla//tools/toolchains/python:python_repo.bzl\", \"python_repository\")' >> third_party/xprof/xprof.patch",
    ],
    repo_mapping = {
        "@local_xla": "@local_xla",
        "@local_tsl": "@local_tsl",
        "@org_tensorflow": "@org_tensorflow",
        "@xla": "@local_xla",
        "@tsl": "@local_tsl",
        "@grpc": "@com_github_grpc_grpc",
    },
)

# Import all of TensorFlow Serving's external dependencies.
# Downstream projects (projects importing TensorFlow Serving) need to
# duplicate all code below in their WORKSPACE file in order to also initialize
# those external dependencies.
load("//tensorflow_serving:workspace.bzl", "tf_serving_workspace")
tf_serving_workspace()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_shell",
    sha256 = "bc61ef94facc78e20a645726f64756e5e285a045037c7a61f65af2941f4c25e1",
    strip_prefix = "rules_shell-0.4.1",
    urls = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
        "https://github.com/bazelbuild/rules_shell/releases/download/v0.4.1/rules_shell-v0.4.1.tar.gz",
    ],
)

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
        "@grpc": "@com_github_grpc_grpc",
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
        "@grpc": "@com_github_grpc_grpc",
    },
    root = "@org_tensorflow//:unused",
)

# Check bazel version requirement, which is stricter than TensorFlow's.
load("@bazel_skylib//lib:versions.bzl", "versions")
versions.check("7.4.1")

load("@bazel_features//:deps.bzl", "bazel_features_deps")
bazel_features_deps()

load("@rules_cc//cc:extensions.bzl", "compatibility_proxy_repo")
compatibility_proxy_repo()

load("@rules_shell//shell:repositories.bzl", "rules_shell_dependencies", "rules_shell_toolchains")
rules_shell_dependencies()
rules_shell_toolchains()

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

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()
load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()
load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

# Initialize bazel package rules' external dependencies.
load("@rules_pkg//:deps.bzl", "rules_pkg_dependencies")
rules_pkg_dependencies()

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
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_versions.bzl",
    "REDIST_VERSIONS_TO_BUILD_TEMPLATES",
)
load("@xla//third_party/cccl:workspace.bzl", "CCCL_2_8_5_DIST_DICT", "CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES")

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS | CCCL_2_8_5_DIST_DICT,
    redist_versions_to_build_templates = REDIST_VERSIONS_TO_BUILD_TEMPLATES | CCCL_GITHUB_VERSIONS_TO_BUILD_TEMPLATES,
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

nccl_redist_init_repository(patches = ["@org_tensorflow//third_party/nccl:nccl_wheel.patch"])

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

