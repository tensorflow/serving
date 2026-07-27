import glob
import re
import sys

def patch_btree(p):
    s = open(p).read()
    s = re.sub(r"(class btree_iterator : private btree_iterator_generation_info \{)", r"\1\n  template <typename, typename, typename> friend class btree_iterator;", s)
    s = re.sub(r"(bool operator==\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator==(const btree_iterator<N, R, P> &other) const { return node_ == other.node_ && position_ == other.position_; }\n  \1", s)
    s = re.sub(r"(bool operator!=\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator!=(const btree_iterator<N, R, P> &other) const { return node_ != other.node_ || position_ != other.position_; }\n  \1", s)
    s = re.sub(r"(const_iterator internal_end\(const_iterator iter\) const \{)", r"const_iterator internal_end(iterator iter) const { return iter.node_ != nullptr ? iter : end(); }\n  \1", s)
    s = re.sub(r"using iterator\s*=\s*typename btree_iterator<node_type, reference, pointer>::iterator;", r"using iterator = btree_iterator<node_type, reference, pointer>;", s)
    open(p, "w").write(s)

def patch_container(p):
    s = open(p).read()
    s = re.sub(r"\s*protected:\s*(// Alias used for heterogeneous lookup functions\.)", r"\n public:\n  \1", s)
    open(p, "w").write(s)

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    for p in glob.glob(f"{target_dir}/**/btree.h", recursive=True):
        patch_btree(p)
    for p in glob.glob(f"{target_dir}/**/btree_container.h", recursive=True):
        patch_container(p)
    print("Successfully patched Abseil btree headers in:", target_dir)
