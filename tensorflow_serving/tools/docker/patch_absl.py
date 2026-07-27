import glob
import re
import sys

def patch_btree(p):
    s = open(p).read()
    if "template <typename N2, typename R2, typename P2> friend class btree_iterator;" not in s:
        s = re.sub(r"(class btree_iterator : private btree_iterator_generation_info \{)", r"\1\n public:\n  template <typename N2, typename R2, typename P2> friend class btree_iterator;", s)
    
    # Make iterator and const_iterator aliases public in btree_iterator
    s = re.sub(r"(\s*)using iterator = std::conditional_t<", r"\1public:\n\1using iterator = std::conditional_t<", s)
    
    s = s.replace("class btree_iterator_generation_info_disabled {", "class btree_iterator_generation_info_disabled {\n public:")
    s = s.replace("class btree_iterator_generation_info_enabled {", "class btree_iterator_generation_info_enabled {\n public:")
    if "bool operator==(const btree_iterator<N, R, P>" not in s:
        s = re.sub(r"(bool operator==\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator==(const btree_iterator<N, R, P> &other) const { return node_ == other.node_ && position_ == other.position_; }\n  \1", s)
    if "bool operator!=(const btree_iterator<N, R, P>" not in s:
        s = re.sub(r"(bool operator!=\(const const_iterator &other\) const \{)", r"template <typename N, typename R, typename P> bool operator!=(const btree_iterator<N, R, P> &other) const { return node_ != other.node_ || position_ != other.position_; }\n  \1", s)
    if "internal_end(iterator iter) const" not in s:
        s = re.sub(r"(const_iterator internal_end\(const_iterator iter\) const \{)", r"const_iterator internal_end(iterator iter) const { return iter.node_ != nullptr ? iter : end(); }\n  \1", s)
    
    # Replace explicit converting constructor (m2) FIRST
    m2 = re.search(r"std::is_same<btree_iterator<N,\s*R,\s*P>,\s*const_iterator>::value\s*&&\s*std::is_same<btree_iterator,\s*iterator>::value", s)
    if m2:
        s = s[:m2.start()] + "std::is_same<std::remove_const_t<N>, normal_node>::value && std::is_same<btree_iterator, iterator>::value && !std::is_same<iterator, const_iterator>::value" + s[m2.end():]
    
    # Replace implicit converting constructor (m1) SECOND
    m1 = re.search(r"std::is_same<btree_iterator<N,\s*R,\s*P>,\s*iterator>::value\s*&&\s*std::is_same<btree_iterator,\s*const_iterator>::value", s)
    if m1:
        s = s[:m1.start()] + "std::is_same<std::remove_const_t<N>, std::remove_const_t<node_type>>::value && std::is_same<btree_iterator, const_iterator>::value" + s[m1.end():]
    
    s = s.replace("btree_iterator(const btree_iterator<N, R, P> other)  // NOLINT", "btree_iterator(const btree_iterator<N, R, P> &other)  // NOLINT")
    s = s.replace("explicit btree_iterator(const btree_iterator<N, R, P> other)", "explicit btree_iterator(const btree_iterator<N, R, P> &other)")
    
    # Fix both initializers to use other.generation()
    s = s.replace("btree_iterator_generation_info(other),", "btree_iterator_generation_info(other.generation()),")
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
