# BST immutable collection - lab 2 - variant 3

This is an immutable collection implementation based on binary search tree (BST).  

All operations will return a new structure without operating on the original structure.  

According to the requirements of the experiment, it also includes
Unit testing, attribute-based testing (hypothesis), and mandatory API testing

## Project structure

- `binary_tree.py` -- immutable BST set with all required API functions

- `test_binary_tree.py` -- unit tests, property‑based tests, and the variant‑3
  API test (copied from the lab PDF).

## Features

- **Immutability**: No operation changes the original set; every “update”
  returns a new set with structural sharing.
- **Recursion**: All traversal and transformation functions are implemented
  recursively (no loops).
- **Monoid**: The set forms a monoid with `empty()` as identity and `concat()`
  as binary operation. Associativity and identity laws are tested.
- **Support for `None`**: The set can contain `None` values alongside
  comparable types (e.g., integers) using a custom `_lt` comparator.
- **Property‑based tests**: Hypothesis ensures invariants (round‑trip,
  length consistency, remove semantics, monoid laws, etc.) hold for many
  random inputs.
- **Full API test**: The variant‑3 test from the lab description is included
  and passes.

## Contribution

- Xia Jiale (<1436172989@qq.com>) -- finish the test_binary_tree.py.

## Changelog

- 2025‑05‑05 – 1
  - Final finish the test_binary_tree.py.

## Design notes

- **Handling `None` comparisons**  
  Python does not allow comparing `None` with integers (`<` raises `TypeError`).
  We defined a helper `_lt(a, b)` that treats `None` as smaller than any other
  value. This preserves total ordering and allows `None` to be stored in the
  tree.

- **`__str__` format**  
  According to the variant‑3 API test, the string representation must be:
  - `"{}"` for empty set,
  - `"{elem}"` for a single element (no space inside),
  - `"{elem1, elem2}"` for two or more elements (comma + space).
  Our implementation follows exactly that.

- **`concat` implementation**  
  To keep the code simple, `concat(a, b)` is implemented as
  `from_list(to_list(a) + to_list(b))`. This does not reuse the internal
  nodes of `b` and is not the most efficient, but it is correct and sufficient
  for this lab. A more efficient version would recursively insert all elements
  of `b` into `a`, but it would not change the asymptotic complexity.
