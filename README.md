# Group-3 - lab 2 - variant 3

This project implements an immutable binary search tree (BST) collection.
All operations return a new structure without modifying the original one.
The implementation includes unit tests, property-based tests (Hypothesis),
and the mandatory variant-3 API test.

## Project Structure

- binary_tree.py — Immutable BST set with all required API functions
- test_binary_tree.py — Unit tests, property-based tests, and the
  variant-3 API test (copied from the lab PDF)

## Core API Functions

- cons — Add a new element
- remove — Delete an element by value
- member — Check if an element exists
- length — Get element count
- to_list — Convert to sorted list
- from_list — Build set from list

## Advanced API Functions

- filter — Keep elements matching a predicate
- map — Transform all elements
- reduce — Aggregate values
- empty — Create an empty set
- concat — Merge two sets
- find — Find element by specific predicate
- iterator — Return an iterator over the set

## Contribution Log

### 29.04.2026 — Lin Shengkai

- Implemented immutable BST data structure
- Wrote basic tests

### 05.05.2026 — Xia Jiale

- Finished test_binary_tree.py

### 05.05.2026 — Lin Shengkai

- Updated README
- Adjusted code style

## Design Notes

This binary search tree maintains the BST property in an immutable style:

- Left child is smaller
- Right child is larger
- Automatic duplicate removal
- Supports None values
- Full test coverage including property-based testing

- **Immutability** — No operation changes the original set; every update
  returns a new set with structural sharing.
- **Recursion** — All traversal and transformation functions are implemented
  recursively (no loops).
- **Monoid** — The set forms a monoid with `empty()` as identity and
  `concat()` as binary operation. Associativity and identity laws are tested.
- **Handling `None` comparisons** — Python does not allow comparing `None`
  with integers (`&lt;` raises `TypeError`). We defined a helper `_lt(a, b)`
  that treats `None` as smaller than any other value. This preserves total
  ordering and allows `None` to be stored in the tree.
- **`__str__` format** — According to the variant-3 API test, the string
  representation must be `"{}"` for empty set, `"{elem}"` for a single
  element, and `"{elem1, elem2}"` for two or more elements.
- **`concat` implementation** — Implemented by inserting all elements of
  the second set into the first. This is correct and sufficient for this lab.