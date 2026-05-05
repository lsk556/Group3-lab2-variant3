from __future__ import annotations

from typing import Any
import itertools
import pytest
from hypothesis import given
from hypothesis.strategies import SearchStrategy
import hypothesis.strategies as st
from binary_tree import (
    BinaryTreeSet,
    concat,
    cons,
    empty,
    filter as bts_filter,
    find,
    from_list,
    intersection,
    iterator,
    length,
    map as bts_map,
    member,
    reduce,
    remove,
    to_list,
)


# ---------- Strategy for generating BinaryTreeSet with arbitrary comparab
def btree() -> SearchStrategy[BinaryTreeSet[Any]]:
    return st.builds(from_list, st.lists(st.one_of(st.none(), st.integers())))


# ---------- Variant 3 API test (from lab description) ----------
def test_api() -> None:
    empty: BinaryTreeSet[Any] = BinaryTreeSet()
    assert str(cons(None, empty)) == "{None}"
    l1 = cons(None, cons(1, empty))
    l2 = cons(1, cons(None, empty))
    assert str(empty) == "{}"
    assert str(l1) == "{None, 1}" or str(l1) == "{1, None}"
    assert empty != l1
    assert empty != l2
    assert l1 == l2
    assert l1 == cons(None, cons(1, l1))

    assert length(empty) == 0
    assert length(l1) == 2
    assert length(l2) == 2

    assert str(remove(l1, None)) == "{1}"
    assert str(remove(l1, 1)) == "{None}"

    assert not member(None, empty)
    assert member(None, l1)
    assert member(1, l1)
    assert not member(2, l1)

    assert intersection(l1, l2) == l1
    assert intersection(l1, l2) == l2
    assert intersection(l1, empty) == empty
    assert intersection(l1, cons(None, empty)) == cons(None, empty)

    assert to_list(l1) == [None, 1] or to_list(l1) == [1, None]
    assert l1 == from_list([None, 1])
    assert l1 == from_list([1, None, 1])

    assert concat(l1, l2) == from_list([None, 1, 1, None])

    buf = []
    for e in l1:
        buf.append(e)
    assert buf in map(list, itertools.permutations([1, None]))

    lst = to_list(l1) + to_list(l2)
    for e in l1:
        lst.remove(e)
    for e in l2:
        lst.remove(e)
    assert lst == []


# ---------- Immutability tests ----------
def test_immutable_cons() -> None:
    t: BinaryTreeSet[int] = empty()
    t2 = cons(1, t)
    assert t == empty()
    assert member(1, t2)
    assert not member(1, t)


def test_immutable_remove() -> None:
    t = from_list([1, 2, 3])
    t2 = remove(t, 2)
    assert member(2, t)
    assert not member(2, t2)
    assert to_list(t) == [1, 2, 3]


@given(btree())
def test_immutable_all_operations(tree: BinaryTreeSet[Any]) -> None:
    """Property-based test that no operation modifies the original tree."""
    orig_list = to_list(tree)
    cons(999, tree)
    assert to_list(tree) == orig_list
    if orig_list:
        elem = orig_list[0]
        remove(tree, elem)
        assert to_list(tree) == orig_list
    _ = bts_filter(tree, lambda x: True)
    assert to_list(tree) == orig_list
    _ = bts_map(tree, lambda x: x)
    assert to_list(tree) == orig_list
    _ = concat(tree, empty())
    assert to_list(tree) == orig_list
    _ = intersection(tree, empty())
    assert to_list(tree) == orig_list


# ---------- Basic function tests ----------
def test_length() -> None:
    t: BinaryTreeSet[int] = empty()
    assert length(t) == 0
    t = cons(1, t)
    assert length(t) == 1
    t = cons(1, t)
    assert length(t) == 1
    t = cons(2, t)
    assert length(t) == 2


def test_member() -> None:
    t = from_list([3, 1, 4, 1, 5])
    assert member(1, t)
    assert member(3, t)
    assert member(5, t)
    assert not member(2, t)


def test_to_list() -> None:
    t = from_list([3, 1, 2])
    assert to_list(t) == [1, 2, 3]


def test_from_list_duplicates() -> None:
    t = from_list([2, 2, 2])
    assert to_list(t) == [2]
    assert length(t) == 1


def test_remove() -> None:
    t = from_list([2, 1, 3])
    t = remove(t, 1)
    assert to_list(t) == [2, 3]
    t = remove(t, 3)
    assert to_list(t) == [2]
    t = remove(t, 2)
    assert t == empty()


def test_concat() -> None:
    a = from_list([1, 2])
    b = from_list([3, 4])
    c = concat(a, b)
    assert to_list(c) == [1, 2, 3, 4]
    assert to_list(a) == [1, 2]
    assert to_list(b) == [3, 4]


def test_intersection() -> None:
    a = from_list([1, 2, 3])
    b = from_list([2, 3, 4])
    c = intersection(a, b)
    assert to_list(c) == [2, 3]
    e: BinaryTreeSet[int] = empty()
    assert intersection(a, e) == e


def test_filter() -> None:
    t = from_list([1, 2, 3, 4, 5])
    res = bts_filter(t, lambda x: x % 2 == 0)
    assert to_list(res) == [2, 4]
    assert to_list(t) == [1, 2, 3, 4, 5]


def test_map() -> None:
    t = from_list([1, 2, 3])
    res = bts_map(t, lambda x: x * 2)
    assert to_list(res) == [2, 4, 6]


def test_map_with_none() -> None:
    t = from_list([1, 2, 3])
    res = bts_map(t, lambda x: None if x == 2 else x)
    assert to_list(res) == [None, 1, 3]
    assert set(to_list(res)) == {None, 1, 3}


def test_reduce() -> None:
    t: BinaryTreeSet[int] = empty()
    assert reduce(t, lambda acc, x: acc + x, 0) == 0
    t = from_list([1, 2, 3, 4])
    assert reduce(t, lambda acc, x: acc + x, 0) == 10
    assert reduce(t, lambda acc, x: acc * x, 1) == 24


def test_find() -> None:
    t = from_list([1, 2, 3, 4, 5])
    assert find(t, lambda x: x > 3) == 4
    assert find(t, lambda x: x > 10) is None


def test_iterator() -> None:
    t = from_list([3, 1, 2])
    assert list(iterator(t)) == [1, 2, 3]
    e: BinaryTreeSet[int] = empty()
    with pytest.raises(StopIteration):
        next(iter(e))


# ---------- Monoid / structure tests ----------
def test_empty_identity() -> None:
    e: BinaryTreeSet[int] = empty()
    assert e == empty()
    assert str(e) == "{}"
    assert length(e) == 0


def test_monoid_associativity_fixed() -> None:
    a = from_list([1, 2])
    b = from_list([3, 4])
    c = from_list([5, 6])
    left = concat(concat(a, b), c)
    right = concat(a, concat(b, c))
    assert left == right


def test_monoid_empty_left() -> None:
    a = from_list([1, 2])
    e: BinaryTreeSet[int] = empty()
    assert concat(e, a) == a


def test_monoid_empty_right() -> None:
    a = from_list([1, 2])
    e: BinaryTreeSet[int] = empty()
    assert concat(a, e) == a


# ---------- Property-Based Tests ----------
@given(btree())
def test_from_list_to_list_roundtrip(tree: BinaryTreeSet[Any]) -> None:
    lst = to_list(tree)
    tree2 = from_list(lst)
    assert tree == tree2


@given(btree())
def test_size_equals_len_of_to_list(tree: BinaryTreeSet[Any]) -> None:
    assert length(tree) == len(to_list(tree))


@given(btree(), st.one_of(st.none(), st.integers()))
def test_remove_removes_element(tree: BinaryTreeSet[Any], elem: Any) -> None:
    before = length(tree)
    had = member(elem, tree)
    tree2 = remove(tree, elem)
    after = length(tree2)
    if had:
        assert after == before - 1
        assert not member(elem, tree2)
    else:
        assert after == before


@given(btree(), btree(), btree())
def test_monoid_associativity(
    a: BinaryTreeSet[Any], b: BinaryTreeSet[Any], c: BinaryTreeSet[Any]
) -> None:
    left = concat(concat(a, b), c)
    right = concat(a, concat(b, c))
    assert left == right


@given(btree())
def test_empty_identity_pbt(tree: BinaryTreeSet[Any]) -> None:
    e: BinaryTreeSet[Any] = empty()
    assert concat(tree, e) == tree
    assert concat(e, tree) == tree


@given(btree(), st.one_of(st.none(), st.integers()))
def test_member_after_cons(tree: BinaryTreeSet[Any], elem: Any) -> None:
    new_tree = cons(elem, tree)
    assert member(elem, new_tree)
    if elem is not None:
        assert member(elem, new_tree) is True
    else:
        assert member(None, new_tree) is True


@given(btree(), st.one_of(st.none(), st.integers()))
def test_find_property(tree: BinaryTreeSet[Any], elem: Any) -> None:
    new_tree = cons(elem, tree)
    found = find(new_tree, lambda x: x == elem)
    assert found == elem
    assert find(tree, lambda x: False) is None


@given(btree())
def test_map_identity(tree: BinaryTreeSet[Any]) -> None:
    mapped = bts_map(tree, lambda x: x)
    assert mapped == tree


@given(btree())
def test_filter_property(tree: BinaryTreeSet[Any]) -> None:
    filtered_true = bts_filter(tree, lambda x: True)
    assert filtered_true == tree
    filtered_false = bts_filter(tree, lambda x: False)
    assert filtered_false == empty()

    def pred(x: Any) -> bool:
        return x is not None and isinstance(x, int) and x % 2 == 0

    filtered = bts_filter(tree, pred)
    for elem in to_list(tree):
        if pred(elem):
            assert member(elem, filtered)
        else:
            assert not member(elem, filtered)


@given(btree())
def test_iterator_matches_to_list(tree: BinaryTreeSet[Any]) -> None:
    assert list(iterator(tree)) == to_list(tree)
