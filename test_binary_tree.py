from __future__ import annotations

from typing import Any, Optional
import itertools
import pytest
from hypothesis import given
from hypothesis.strategies import composite, SearchStrategy
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


@composite
def btree(
    draw: Any,
    elements: SearchStrategy[int] = st.integers(),
) -> BinaryTreeSet[int]:
    lst: list[int] = draw(st.lists(elements))
    return from_list(lst)


# ---------- Variant 3 API test (from lab description) ----------
def test_api() -> None:
    e: BinaryTreeSet[Optional[int]] = empty()
    assert str(cons(None, e)) == "{None}"
    l1 = cons(None, cons(1, e))
    l2 = cons(1, cons(None, e))
    assert str(e) == "{}"
    assert str(l1) == "{None,1}" or str(l1) == "{1,None}"
    assert e != l1
    assert e != l2
    assert l1 == l2
    assert l1 == cons(None, cons(1, l1))
    assert length(e) == 0
    assert length(l1) == 2
    assert length(l2) == 2
    assert str(remove(l1, None)) == "{1}"
    assert str(remove(l1, 1)) == "{None}"
    assert not member(None, e)
    assert member(None, l1)
    assert member(1, l1)
    assert not member(2, l1)
    assert intersection(l1, l2) == l1
    assert intersection(l1, l2) == l2
    assert intersection(l1, e) == e
    assert intersection(l1, cons(None, e)) == cons(None, e)
    assert to_list(l1) == [None, 1] or to_list(l1) == [1, None]
    assert l1 == from_list([None, 1])
    assert l1 == from_list([1, None, 1])
    assert concat(l1, l2) == from_list([None, 1, 1, None])
    buf = []
    for elem in l1:
        buf.append(elem)
    assert buf in map(list, itertools.permutations([1, None]))
    lst = to_list(l1) + to_list(l2)
    for elem in l1:
        lst.remove(elem)
    for elem in l2:
        lst.remove(elem)
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
@given(btree(st.integers()))
def test_from_list_to_list_roundtrip(
    tree: BinaryTreeSet[int],
) -> None:
    lst = to_list(tree)
    tree2 = from_list(lst)
    assert tree == tree2


@given(btree(st.integers()))
def test_size_equals_len_of_to_list(
    tree: BinaryTreeSet[int],
) -> None:
    assert length(tree) == len(to_list(tree))


@given(btree(st.integers()), st.integers())
def test_remove_removes_element(
    tree: BinaryTreeSet[int], b: int
) -> None:
    before = length(tree)
    had_b = member(b, tree)
    tree2 = remove(tree, b)
    after = length(tree2)
    if had_b:
        assert after == before - 1
        assert not member(b, tree2)
    else:
        assert after == before


@given(
    btree(st.integers()),
    btree(st.integers()),
    btree(st.integers()),
)
def test_monoid_associativity(
    a: BinaryTreeSet[int],
    b: BinaryTreeSet[int],
    c: BinaryTreeSet[int],
) -> None:
    left = concat(concat(a, b), c)
    right = concat(a, concat(b, c))
    assert left == right


@given(btree(st.integers()))
def test_empty_identity_pbt(
    tree: BinaryTreeSet[int],
) -> None:
    expected = to_list(tree)
    e: BinaryTreeSet[int] = empty()
    assert to_list(concat(tree, e)) == expected
    assert to_list(concat(e, tree)) == expected
