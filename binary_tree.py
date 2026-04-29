from __future__ import annotations

from typing import (
    Callable,
    Generic,
    Iterator,
    Optional,
    Protocol,
    TypeVar,
    cast,
)

T = TypeVar("T")
U = TypeVar("U")


class _SupportsLt(Protocol):
    def __lt__(self, other: object) -> bool:
        ...


def _lt(a: object, b: object) -> bool:
    if a is None and b is None:
        return False
    if a is None:
        return True
    if b is None:
        return False
    return cast(_SupportsLt, a) < cast(_SupportsLt, b)


class _Node(Generic[T]):
    __slots__ = ("value", "left", "right")

    def __init__(
        self,
        value: T,
        left: Optional[_Node[T]],
        right: Optional[_Node[T]],
    ) -> None:
        self.value: T = value
        self.left: Optional[_Node[T]] = left
        self.right: Optional[_Node[T]] = right


class BinaryTreeSet(Generic[T]):
    def __init__(self, root: Optional[_Node[T]] = None) -> None:
        self._root: Optional[_Node[T]] = root

    def __iter__(self) -> Iterator[T]:
        return _iterate(self._root)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BinaryTreeSet):
            return False
        return to_list(self) == to_list(other)

    def __str__(self) -> str:
        items = ",".join(str(v) for v in to_list(self))
        return "{" + items + "}"


def _iterate(node: Optional[_Node[T]]) -> Iterator[T]:
    if node is None:
        return
    yield from _iterate(node.left)
    yield node.value
    yield from _iterate(node.right)


def _insert(node: Optional[_Node[T]], value: T) -> _Node[T]:
    if node is None:
        return _Node(value, None, None)
    if value == node.value:
        return node
    if _lt(value, node.value):
        return _Node(node.value, _insert(node.left, value), node.right)
    return _Node(node.value, node.left, _insert(node.right, value))


def _find_min(node: _Node[T]) -> _Node[T]:
    if node.left is None:
        return node
    return _find_min(node.left)


def _remove(node: Optional[_Node[T]], value: T) -> Optional[_Node[T]]:
    if node is None:
        return None
    if value == node.value:
        if node.left is None:
            return node.right
        if node.right is None:
            return node.left
        succ = _find_min(node.right)
        return _Node(
            succ.value,
            node.left,
            _remove(node.right, succ.value),
        )
    if _lt(value, node.value):
        return _Node(node.value, _remove(node.left, value), node.right)
    return _Node(node.value, node.left, _remove(node.right, value))


def _member(node: Optional[_Node[T]], value: T) -> bool:
    if node is None:
        return False
    if value == node.value:
        return True
    if _lt(value, node.value):
        return _member(node.left, value)
    return _member(node.right, value)


def _length(node: Optional[_Node[T]]) -> int:
    if node is None:
        return 0
    return 1 + _length(node.left) + _length(node.right)


def _to_list(node: Optional[_Node[T]]) -> list[T]:
    if node is None:
        return []
    return _to_list(node.left) + [node.value] + _to_list(node.right)


def _filter_list(
    node: Optional[_Node[T]], predicate: Callable[[T], bool]
) -> list[T]:
    if node is None:
        return []
    left = _filter_list(node.left, predicate)
    right = _filter_list(node.right, predicate)
    if predicate(node.value):
        return left + [node.value] + right
    return left + right


def _map_list(node: Optional[_Node[T]], func: Callable[[T], U]) -> list[U]:
    if node is None:
        return []
    return (
        _map_list(node.left, func)
        + [func(node.value)]
        + _map_list(node.right, func)
    )


def _reduce(
    node: Optional[_Node[T]], func: Callable[[U, T], U], acc: U
) -> U:
    if node is None:
        return acc
    acc = _reduce(node.left, func, acc)
    acc = func(acc, node.value)
    return _reduce(node.right, func, acc)


def _intersection(
    node: Optional[_Node[T]], tree: BinaryTreeSet[T]
) -> list[T]:
    if node is None:
        return []
    left = _intersection(node.left, tree)
    right = _intersection(node.right, tree)
    if member(node.value, tree):
        return left + [node.value] + right
    return left + right


def _find(
    node: Optional[_Node[T]], predicate: Callable[[T], bool]
) -> Optional[T]:
    if node is None:
        return None
    left = _find(node.left, predicate)
    if left is not None:
        return left
    if predicate(node.value):
        return node.value
    return _find(node.right, predicate)


def empty() -> BinaryTreeSet[T]:
    return BinaryTreeSet()


def cons(element: T, tree: BinaryTreeSet[T]) -> BinaryTreeSet[T]:
    return BinaryTreeSet(_insert(tree._root, element))


def remove(tree: BinaryTreeSet[T], element: T) -> BinaryTreeSet[T]:
    return BinaryTreeSet(_remove(tree._root, element))


def length(tree: BinaryTreeSet[T]) -> int:
    return _length(tree._root)


def member(element: T, tree: BinaryTreeSet[T]) -> bool:
    return _member(tree._root, element)


def intersection(
    tree1: BinaryTreeSet[T], tree2: BinaryTreeSet[T]
) -> BinaryTreeSet[T]:
    return from_list(_intersection(tree1._root, tree2))


def to_list(tree: BinaryTreeSet[T]) -> list[T]:
    return _to_list(tree._root)


def from_list(lst: list[T]) -> BinaryTreeSet[T]:
    if not lst:
        return BinaryTreeSet()
    return cons(lst[0], from_list(lst[1:]))


def _concat_from_list(
    tree: BinaryTreeSet[T], lst: list[T]
) -> BinaryTreeSet[T]:
    if not lst:
        return tree
    return cons(lst[0], _concat_from_list(tree, lst[1:]))


def concat(
    tree1: BinaryTreeSet[T], tree2: BinaryTreeSet[T]
) -> BinaryTreeSet[T]:
    return _concat_from_list(tree1, to_list(tree2))


def filter(
    tree: BinaryTreeSet[T], predicate: Callable[[T], bool]
) -> BinaryTreeSet[T]:
    return from_list(_filter_list(tree._root, predicate))


def map(
    tree: BinaryTreeSet[T], func: Callable[[T], U]
) -> BinaryTreeSet[U]:
    return from_list(_map_list(tree._root, func))


def reduce(
    tree: BinaryTreeSet[T], func: Callable[[U, T], U], initial: U
) -> U:
    return _reduce(tree._root, func, initial)


def find(
    tree: BinaryTreeSet[T], predicate: Callable[[T], bool]
) -> Optional[T]:
    return _find(tree._root, predicate)


def iterator(tree: BinaryTreeSet[T]) -> Iterator[T]:
    return _iterate(tree._root)
