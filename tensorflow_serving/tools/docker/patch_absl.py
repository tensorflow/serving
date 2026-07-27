import glob
import re
import sys

def patch_btree(p):
    s = open(p).read()
    if "friend class btree_iterator;" not in s:
        s = re.sub(r"(class btree_iterator : private btree_iterator_generation_info \{)", r"\1\n  template <typename, typename, typename> friend class btree_iterator;", s)
    if "bool operator==(const btree_iterator<N, R, P>" not in s:
        s = re.sub(r"(bool operator==\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator==(const btree_iterator<N, R, P> &other) const { return node_ == other.node_ && position_ == other.position_; }\n  \1", s)
    if "bool operator!=(const btree_iterator<N, R, P>" not in s:
        s = re.sub(r"(bool operator!=\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator!=(const btree_iterator<N, R, P> &other) const { return node_ != other.node_ || position_ != other.position_; }\n  \1", s)
    if "internal_end(iterator iter) const" not in s:
        s = re.sub(r"(const_iterator internal_end\(const_iterator iter\) const \{)", r"const_iterator internal_end(iterator iter) const { return iter.node_ != nullptr ? iter : end(); }\n  \1", s)
    if "using iterator = btree_iterator<node_type, reference, pointer>;" not in s:
        s = re.sub(r"using iterator\s*=\s*typename btree_iterator<node_type, reference, pointer>::iterator;", r"using iterator = btree_iterator<node_type, reference, pointer>;", s)
    s = re.sub(r"std::is_same<btree_iterator<N, R, P>,\s*iterator>::value\s*&&\s*std::is_same<btree_iterator,\s*const_iterator>::value", r"std::is_same<btree_iterator, const_iterator>::value && !std::is_same<btree_iterator<N, R, P>, const_iterator>::value", s)
    s = s.replace("btree_iterator(const btree_iterator<N, R, P> other)  // NOLINT", "btree_iterator(const btree_iterator<N, R, P> &other)  // NOLINT")
    s = s.replace("node_(other.node_),\n        position_(other.position_) {}", "node_(const_cast<node_type*>(other.node_)),\n        position_(other.position_) {}")
    open(p, "w").write(s)

def patch_container(p):
    s = open(p).read()
    if " public:\n  // Alias used for heterogeneous lookup functions." not in s:
        s = re.sub(r"\s*protected:\s*(// Alias used for heterogeneous lookup functions\.)", r"\n public:\n  \1", s)
    open(p, "w").write(s)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for p in glob.glob(f"{target_dir}/**/btree.h", recursive=True):
        patch_btree(p)
    for p in glob.glob(f"{target_dir}/**/btree_container.h", recursive=True):
        patch_container(p)
    print("Successfully patched Abseil btree headers in:", target_dir)
