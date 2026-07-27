"""Provides a macro to import all TensorFlow Serving dependencies.

Some of the external dependencies need to be initialized. To do this, duplicate
the initialization code from TensorFlow Serving's WORKSPACE file.
"""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def tf_serving_workspace():
    """All TensorFlow Serving external dependencies."""

    # ===== Abseil C++ dependency with Clang 18 C++17 public iterator alias patch =====
    http_archive(
        name = "com_google_absl",
        sha256 = "6e1aee535473414164bf83e4ebc40240dec71a4701f8a642d906e95bea1aea0c",
        strip_prefix = "abseil-cpp-20260526.0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/20260526.0.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/20260526.0.tar.gz",
        ],
        patch_cmds = [
            'python3 -c \'import glob, re\\nfor p in glob.glob("**/btree.h", recursive=True):\\n    s = open(p).read()\\n    if "friend class btree_iterator;" not in s:\\n        s = re.sub(r"std::is_same<btree_iterator<N,\\\\s*R,\\\\s*P>,\\\\s*iterator>::value\\\\s*&&\\\\s*std::is_same<btree_iterator,\\\\s*const_iterator>::value", r"std::is_same<std::remove_const_t<N>, std::remove_const_t<node_type>>::value && std::is_same<btree_iterator, const_iterator>::value", s)\\n        s = re.sub(r"std::is_same<btree_iterator<N,\\\\s*R,\\\\s*P>,\\\\s*const_iterator>::value\\\\s*&&\\\\s*std::is_same<btree_iterator,\\\\s*iterator>::value", r"std::is_same<std::remove_const_t<N>, normal_node>::value && std::is_same<btree_iterator, iterator>::value && !std::is_same<iterator, const_iterator>::value", s)\\n        s = re.sub(r"(\\\\s*)using iterator = std::conditional_t<", r"\\1public:\\n\\1using iterator = std::conditional_t<", s)\\n        s = re.sub(r"(class btree_iterator : private btree_iterator_generation_info \\{)", r"\\1\\n public:\\n  template <typename N2, typename R2, typename P2> friend class btree_iterator;", s)\\n        s = re.sub(r"(bool operator==\\(const const_iterator &other\\) const \\{)", r"template <typename N, typename R, typename P> bool operator==(const btree_iterator<N, R, P> &other) const { return node_ == other.node_ && position_ == other.position_; }\\n  \\1", s)\\n        s = re.sub(r"(bool operator!=\\(const const_iterator &other\\) const \\{)", r"template <typename N, typename R, typename P> bool operator!=(const btree_iterator<N, R, P> &other) const { return node_ != other.node_ || position_ != other.position_; }\\n  \\1", s)\\n        s = re.sub(r"(const_iterator internal_end\\(const_iterator iter\\) const \\{)", r"const_iterator internal_end(iterator iter) const { return iter.node_ != nullptr ? iter : end(); }\\n  \\1", s)\\n        s = s.replace("class btree_iterator_generation_info_disabled {", "class btree_iterator_generation_info_disabled {\\n public:")\\n        s = s.replace("class btree_iterator_generation_info_enabled {", "class btree_iterator_generation_info_enabled {\\n public:")\\n        s = s.replace("btree_iterator(const btree_iterator<N, R, P> other)  // NOLINT", "btree_iterator(const btree_iterator<N, R, P> &other)  // NOLINT")\\n        s = s.replace("explicit btree_iterator(const btree_iterator<N, R, P> other)", "explicit btree_iterator(const btree_iterator<N, R, P> &other)")\\n        s = s.replace("btree_iterator_generation_info(other),", "btree_iterator_generation_info(other.generation()),")\\n        s = s.replace("node_(other.node_),\\n        position_(other.position_) {}", "node_(const_cast<node_type*>(other.node_)),\\n        position_(other.position_) {}")\\n        open(p, "w").write(s)\'',
            'python3 -c \'import glob, re; [(lambda s=open(p).read(): (" public:\\n  // Alias used for heterogeneous lookup functions." not in s and open(p, "w").write(re.sub(r"\\\\s*protected:\\\\s*(// Alias used for heterogeneous lookup functions\\.)", r"\\n public:\\n  \\1", s))))() for p in glob.glob("**/btree_container.h", recursive=True)]\'',
        ]
        
