"""Structural (AST) assertions for tests that must check CODE SHAPE rather than
behaviour or drawn artists.

WHY THIS EXISTS
Some invariants have no runtime signal to observe: the LR warmup converts epochs to
optimiser steps as `lr_warmup_epochs * len(train_loader)` (a decaying schedule or a
mid-warmup rollback is the only thing a behavioural test could distinguish, and the
runs that matter finish warmup long before either), and the rollback path must rebuild
the scheduler alongside the optimizer. These were guarded by `"literal string" in src`
-- which breaks on any reformat (line-wrap, whitespace, a rename that moves the code)
with no bug, and passes on a near-miss that happens to contain the substring.

These helpers parse the source to an AST and assert on NODES instead of text: "there
is a call to make_lr_warmup", "some `a * b` multiplies lr_warmup_epochs by something
mentioning train_loader". That survives formatting and refactors while still catching
the real mistake (bare `total_iters=lr_warmup_epochs`, a missing scheduler rebuild).

Use these ONLY where the invariant is code-shape with no observable: prefer a
behavioural test (call it, check the result) or _figure_artifacts (inspect what was
drawn) whenever one is possible. Requires Python 3.9+ for ast.unparse.
"""

import ast
from pathlib import Path


def parse_fragment(text: str) -> ast.AST:
    """Parse an INDENTED code fragment sliced out of a function body (e.g. a rollback
    block located by string index). Such a slice has its first line flush but the rest
    still indented, which bare ast.parse rejects -- so dedent to a common base and wrap
    in `if True:` to give any residual nesting a valid parent. For asserting call
    structure WITHIN a specific block, where whole-module parsing would lose the
    scoping (a call made elsewhere would falsely satisfy the check)."""
    import textwrap
    lines = text.split("\n")
    rest = [l for l in lines[1:] if l.strip()]
    base = min((len(l) - len(l.lstrip()) for l in rest), default=0)
    dedented = [lines[0].lstrip()] + [l[base:] if len(l) >= base else l for l in lines[1:]]
    return ast.parse("if True:\n" + textwrap.indent("\n".join(dedented), "    "))


def parse_module(path) -> ast.Module:
    """Parse a source file to an AST module."""
    return ast.parse(Path(path).read_text(encoding="utf-8"))


def parse_function(func) -> ast.AST:
    """Parse a single live function object's source to an AST (via inspect)."""
    import inspect
    import textwrap
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _callee_name(call: ast.Call):
    f = call.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


def calls_to(tree: ast.AST, name: str) -> list[ast.Call]:
    """Every Call node invoking a function named `name` -- matches both a bare
    `name(...)` and an attribute `obj.name(...)`. Empty list means it is never
    called (structural equivalent of `"name(" in src`, but not fooled by the name
    appearing in a comment, a string, or a different context)."""
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Call) and _callee_name(n) == name]


def call_kwarg(call: ast.Call, name: str):
    """The AST value node for keyword argument `name` in a Call, or None."""
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _operand_mentions(node: ast.AST, name: str) -> bool:
    """True if `name` appears as an actual Name node anywhere in an operand's subtree
    (so `train_loader` matches `len(train_loader)` but NOT the unrelated identifier
    `train_loader_y`). Falls back to substring only for operands with no Name nodes
    (e.g. a pure literal), which the callers here never rely on."""
    names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    return name in names


def multiplies(tree: ast.AST, a_name: str, b_name: str) -> bool:
    """True if some `X * Y` (either operand order) has the identifier `a_name` as a
    Name in one operand and `b_name` in the other. Matches on AST Name nodes, NOT
    source substrings -- so multiplies(t, 'lr_warmup_epochs', 'train_loader') matches
    `lr_warmup_epochs * len(train_loader)` however it is formatted, does NOT match a
    bare `total_iters=lr_warmup_epochs` (no multiplication -- the batch-units bug),
    and does NOT false-match a similarly-named `lr_warmup_epochs_x * train_loader_y`."""
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if (_operand_mentions(node.left, a_name) and _operand_mentions(node.right, b_name)) or \
               (_operand_mentions(node.right, a_name) and _operand_mentions(node.left, b_name)):
                return True
    return False


def names_bound(tree: ast.AST) -> set[str]:
    """All names bound by assignment or annotated-assignment anywhere in the tree --
    for asserting that e.g. `lr_scheduler` is (re)assigned in a rollback block."""
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
    return out
