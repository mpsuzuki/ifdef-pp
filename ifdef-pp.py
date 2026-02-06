#!/usr/bin/env python3
import re
import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, List

# ------------------------------------------------------------
# CondAtom: represents a single macro condition (pending)
# ------------------------------------------------------------

class CondType(Enum):
    DEFINE  = "D"
    UNDEF   = "U"
    NEUTRAL = "N"
    COMPLEX = "X"
    CONST_BOOLEAN = "B"

@dataclass
class CondAtom:
    kind: CondType
    macro: Optional[str]

    @classmethod
    def define(cls, macro):
        return cls(CondType.DEFINE, macro)

    @classmethod
    def undef(cls, macro):
        return cls(CondType.UNDEF, macro)

    @classmethod
    def neutral(cls, macro):
        return cls(CondType.NEUTRAL, macro)

    @classmethod
    def complex(cls):
        return cls(CondType.COMPLEX, None)

    def __repr__(self):
        macro = self.macro if self.macro else "*"
        return f"{self.kind.value}:{macro}"

class TrueAtom(CondAtom):
    def __init__(self):
        super().__init__(CondType.CONST_BOOLEAN, None)
        self.value = True

    def __repr__(self):
        return "<True>"

class FalseAtom(CondAtom):
    def __init__(self):
        super().__init__(CondType.CONST_BOOLEAN, None)
        self.value = False

    def __repr__(self):
        return "<False>"

TRUE_ATOM = TrueAtom()
FALSE_ATOM = FalseAtom()

# ------------------------------------------------------------
# CondExpr: boolean expression tree
# ------------------------------------------------------------

class CondExprKind(Enum):
    ATOM = auto()
    UNKNOWN = auto()
    NOT  = auto()
    AND  = auto()
    OR   = auto()


@dataclass
class CondExpr:
    kind: CondExprKind
    atom: Optional[CondAtom] = None
    left: Optional["CondExpr"] = None
    right: Optional["CondExpr"] = None

    @classmethod
    def atom_expr(cls, atom):
        return cls(CondExprKind.ATOM, atom=atom)

    @classmethod
    def true(cls):
        return cls.atom_expr(TRUE_ATOM)

    @classmethod
    def false(cls):
        return cls.atom_expr(FALSE_ATOM)

    @classmethod
    def And(cls, a, b):
        return cls(CondExprKind.AND, left=a, right=b)

    @classmethod
    def Or(cls, a, b):
        return cls(CondExprKind.OR, left=a, right=b)

    @classmethod
    def Not(cls, a):
        return cls(CondExprKind.NOT, left=a)

# ------------------------------------------------------------
# TriValue: 3-valued logic
# ------------------------------------------------------------

class TriValue(Enum):
    TRUE = auto()
    FALSE = auto()
    PENDING = auto()

# ------------------------------------------------------------
# LineObj: represents one line of source
# ------------------------------------------------------------

class DirectiveKind(Enum):
    NONE     = auto()
    PP_MISC  = auto()
    IF       = auto()
    IFDEF    = auto()
    IFNDEF   = auto()
    ELIF     = auto()
    ELSE     = auto()
    ENDIF    = auto()

@dataclass
class LineObj:
    text: str
    directive: DirectiveKind = DirectiveKind.NONE
    related_if: Optional[int] = None
    local_cond: Optional[CondExpr] = None
    effective_cond: Optional[CondExpr] = None

    def is_directive_if(self): return self.directive == DirectiveKind.IF
    def is_directive_ifdef(self): return self.directive == DirectiveKind.IFDEF
    def is_directive_ifndef(self): return self.directive == DirectiveKind.IFNDEF
    def is_directive_elif(self): return self.directive == DirectiveKind.ELIF
    def is_directive_else(self): return self.directive == DirectiveKind.ELSE
    def is_directive_endif(self): return self.directive == DirectiveKind.ENDIF
    def is_directive_none(self): return self.directive == DirectiveKind.NONE
    def is_directive_pp_misc(self): return self.directive == DirectiveKind.PP_MISC

# ------------------------------------------------------------
# IfFrame: used during propagation
# ------------------------------------------------------------

@dataclass
class IfFrame:
    parent_effective: CondExpr
    taken: CondExpr

# ------------------------------------------------------------
# Regex for parsing directives
# ------------------------------------------------------------

regex_ifdef  = re.compile(r'^\s*#\s*ifdef\s+(\w+)\b')
regex_ifndef = re.compile(r'^\s*#\s*ifndef\s+(\w+)\b')
regex_if     = re.compile(r'^\s*#\s*if\b(.*)')
regex_elif_defined     = re.compile(r'^\s*#\s*elif\s+defined\s*\(\s*(\w+)\s*\)')
regex_elif_not_defined = re.compile(r'^\s*#\s*elif\s+!\s*defined\s*\(\s*(\w+)\s*\)')
regex_elif   = re.compile(r'^\s*#\s*elif\b(.*)')
regex_else   = re.compile(r'^\s*#\s*else\b')
regex_endif  = re.compile(r'^\s*#\s*endif\b')
regex_misc   = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')

def resolve_parent_if(lo, if_stack, objs, idx):
    if not if_stack:
        raise SyntaxError(f"Unmatched directive at line {idx+1}: {lo.text}")
    lo.related_if = if_stack[-1]
    return

# ------------------------------------------------------------
# parse_lines
# ------------------------------------------------------------

def parse_lines(lines):
    objs = []
    if_stack = []

    for idx, line in enumerate(lines):
        lo = LineObj(text=line)
        objs.append(lo)

        if m := regex_ifdef.match(line):
            lo.directive = DirectiveKind.IFDEF
            lo.local_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            lo.related_if = idx
            if_stack.append(idx)
            continue

        elif m := regex_ifndef.match(line):
            lo.directive = DirectiveKind.IFNDEF
            lo.local_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            lo.related_if = idx
            if_stack.append(idx)
            continue

        elif m := regex_if.match(line):
            lo.directive = DirectiveKind.IF
            lo.local_cond = CondExpr.atom_expr(CondAtom.complex())
            lo.related_if = idx
            if_stack.append(idx)
            continue

        elif m := regex_elif_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            resolve_parent_if(lo, if_stack, objs, idx)
            lo.local_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            continue

        elif m := regex_elif_not_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            resolve_parent_if(lo, if_stack, objs, idx)
            macro = m.group(1)
            lo.local_cond = CondExpr.Not(
                CondExpr.atom_expr(CondAtom.neutral(macro))
            )
            continue

        elif m := regex_elif.match(line):
            lo.directive = DirectiveKind.ELIF
            resolve_parent_if(lo, if_stack, objs, idx)
            lo.local_cond = CondExpr.atom_expr(CondAtom.complex())
            continue

        elif regex_else.match(line):
            lo.directive = DirectiveKind.ELSE
            resolve_parent_if(lo, if_stack, objs, idx)
            lo.local_cond = CondExpr.true()
            continue

        elif regex_endif.match(line):
            lo.directive = DirectiveKind.ENDIF
            resolve_parent_if(lo, if_stack, objs, idx)
            lo.local_cond = CondExpr.true()
            if_stack.pop()
            continue

        elif regex_misc.match(line):
            lo.directive = DirectiveKind.PP_MISC
            continue

        else:
            lo.directive = DirectiveKind.NONE

    if if_stack:
        raise SyntaxError("Unclosed #if block(s)")

    return objs

# ------------------------------------------------------------
# propagate_effective_conds
# ------------------------------------------------------------

def propagate_effective_conds(objs):
    current_effective = CondExpr.true()
    stack = []

    for lo in objs:

        if lo.is_directive_none() or lo.is_directive_pp_misc():
            lo.effective_cond = current_effective
            continue

        if lo.is_directive_if() or lo.is_directive_ifdef() or lo.is_directive_ifndef():
            cond_if = lo.local_cond or CondExpr.true()
            lo.effective_cond = CondExpr.And(current_effective, cond_if)

            frame = IfFrame(parent_effective=current_effective, taken=cond_if)
            stack.append(frame)

            current_effective = lo.effective_cond
            continue

        if lo.is_directive_elif():
            frame = stack[-1]
            cond_elif = lo.local_cond or CondExpr.true()

            local = CondExpr.And(cond_elif, CondExpr.Not(frame.taken))
            lo.effective_cond = CondExpr.And(frame.parent_effective, local)

            frame.taken = CondExpr.Or(frame.taken, cond_elif)
            current_effective = lo.effective_cond
            continue

        if lo.is_directive_else():
            frame = stack[-1]
            local = CondExpr.Not(frame.taken)
            lo.effective_cond = CondExpr.And(frame.parent_effective, local)
            current_effective = lo.effective_cond
            continue

        if lo.is_directive_endif():
            frame = stack.pop()
            lo.effective_cond = frame.parent_effective
            current_effective = frame.parent_effective
            continue

# ------------------------------------------------------------
# eval_atom
# ------------------------------------------------------------

def eval_atom(atom, defined_set, undefined_set):
    macro = atom.macro

    if atom.kind == CondType.COMPLEX:
        return TriValue.PENDING

    if atom.kind == CondType.CONST_BOOLEAN:
        return TriValue.TRUE if atom is TRUE_ATOM else TriValue.FALSE

    if atom.kind == CondType.NEUTRAL:
        if macro in defined_set:
            return TriValue.TRUE
        if macro in undefined_set:
            return TriValue.FALSE
        return TriValue.PENDING

    if atom.kind == CondType.DEFINE:
        if macro in defined_set:
            return TriValue.TRUE
        if macro in undefined_set:
            return TriValue.FALSE
        return TriValue.PENDING

    if atom.kind == CondType.UNDEF:
        if macro in defined_set:
            return TriValue.FALSE
        if macro in undefined_set:
            return TriValue.TRUE
        return TriValue.PENDING

    return TriValue.PENDING

# ------------------------------------------------------------
# eval_expr
# ------------------------------------------------------------

def eval_expr(expr, defined_set, undefined_set):
    if expr.kind == CondExprKind.ATOM:
        return eval_atom(expr.atom, defined_set, undefined_set)

    if expr.kind == CondExprKind.NOT:
        v = eval_expr(expr.left, defined_set, undefined_set)
        if v == TriValue.TRUE: return TriValue.FALSE
        if v == TriValue.FALSE: return TriValue.TRUE
        return TriValue.PENDING

    if expr.kind == CondExprKind.AND:
        v1 = eval_expr(expr.left, defined_set, undefined_set)
        v2 = eval_expr(expr.right, defined_set, undefined_set)
        if v1 == TriValue.FALSE or v2 == TriValue.FALSE:
            return TriValue.FALSE
        if v1 == TriValue.TRUE and v2 == TriValue.TRUE:
            return TriValue.TRUE
        return TriValue.PENDING

    if expr.kind == CondExprKind.OR:
        v1 = eval_expr(expr.left, defined_set, undefined_set)
        v2 = eval_expr(expr.right, defined_set, undefined_set)
        if v1 == TriValue.TRUE or v2 == TriValue.TRUE:
            return TriValue.TRUE
        if v1 == TriValue.FALSE and v2 == TriValue.FALSE:
            return TriValue.FALSE
        return TriValue.PENDING

    return TriValue.PENDING

# ------------------------------------------------------------
# expr_to_if (minimal)
# ------------------------------------------------------------

def expr_to_if(expr):
    if expr.kind == CondExprKind.ATOM:
        atom = expr.atom
        if atom.kind == CondType.DEFINE:
            return f"defined({atom.macro})"
        if atom.kind == CondType.UNDEF:
            return f"!defined({atom.macro})"
        if atom.kind == CondType.NEUTRAL:
            return f"defined({atom.macro}) /* pending */"
        if atom.kind == CondType.COMPLEX:
            return "/* complex */"
        if atom.kind == CondType.CONST_BOOLEAN:
            return "1" if atom is TRUE_ATOM else "0"

    if expr.kind == CondExprKind.NOT:
        return f"!({expr_to_if(expr.left)})"

    if expr.kind == CondExprKind.AND:
        return f"({expr_to_if(expr.left)} && {expr_to_if(expr.right)})"

    if expr.kind == CondExprKind.OR:
        return f"({expr_to_if(expr.left)} || {expr_to_if(expr.right)})"

    return "/* unknown */"

# ------------------------------------------------------------
# filter_output_lines
# ------------------------------------------------------------

def collect_if_chains(objs):
    """
    return:
        chains: dict
            if_idx -> {
                "branches": [branch_start_idx, ...],
                "end": endif_idx
            }
    """
    chains = {}
    stack = []

    for idx, lo in enumerate(objs):
        if lo.is_directive_if() or lo.is_directive_ifdef() or lo.is_directive_ifndef():
            chains[idx] = {"branches": [idx], "end": None}
            stack.append(idx)

        elif lo.is_directive_elif() or lo.is_directive_else():
            parent = lo.related_if
            if parent in chains:
                chains[parent]["branches"].append(idx)

        elif lo.is_directive_endif():
            parent = lo.related_if
            if parent in chains:
                chains[parent]["end"] = idx
            if stack:
                stack.pop()

    return chains

def collapse_fully_resolved_if_chains(objs, defined_set, undefined_set):
    to_remove = set()
    chains = collect_if_chains(objs)

    for if_idx, info in chains.items():
        branches = info["branches"]
        end = info["end"]
        if end is None:
            continue

        results = []
        for b in branches:
            expr = objs[b].effective_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            results.append((b, v))

        true_branches = [b for b, v in results if v == TriValue.TRUE]
        pending = any(v == TriValue.PENDING for _, v in results)

        if len(true_branches) == 1 and not pending:
            # remove directives only
            for i in range(if_idx, end + 1):
                lo = objs[i]
                if lo.related_if == if_idx:
                    to_remove.add(i)

    return to_remove

def remove_false_branches(objs, defined_set, undefined_set):
    to_remove = set()
    chains = collect_if_chains(objs)

    for if_idx, info in chains.items():
        branches = info["branches"]
        end = info["end"]
        if end is None:
            continue

        # build branch ranges
        ranges = []
        for i, start in enumerate(branches):
            if i + 1 < len(branches):
                stop = branches[i+1] - 1
            else:
                stop = end - 1
            ranges.append((start, stop))

        # evaluate and remove
        for start, stop in ranges:
            expr = objs[start].effective_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            if v == TriValue.FALSE:
                for i in range(start, stop + 1):
                    to_remove.add(i)

    return to_remove

def compute_if_chain_pending(objs, defined_set, undefined_set, idx_to_remove):
    if_chain_pending = {}

    for idx, lo in enumerate(objs):
        if idx in idx_to_remove:
            continue

        if lo.is_directive_if() or lo.is_directive_ifdef() or lo.is_directive_ifndef():
            expr = lo.effective_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            if_chain_pending[idx] = (v == TriValue.PENDING)

        elif lo.is_directive_elif() or lo.is_directive_else():
            parent_idx = lo.related_if
            if parent_idx is not None and parent_idx in if_chain_pending:
                expr = lo.effective_cond or CondExpr.true()
                v = eval_expr(expr, defined_set, undefined_set)
                if v == TriValue.PENDING:
                    if_chain_pending[parent_idx] = True
    return if_chain_pending

def remove_inactive_lines(objs, defined_set, undefined_set, if_chain_pending, idx_to_remove):
    for idx, lo in enumerate(objs):
        if idx in idx_to_remove:
            continue

        # Directives use local_cond; normal lines use effective_cond
        if (
            lo.is_directive_if()
            or lo.is_directive_ifdef()
            or lo.is_directive_ifndef()
            or lo.is_directive_elif()
            or lo.is_directive_else()
        ):
            expr = lo.local_cond or CondExpr.true()
        else:
            expr = lo.effective_cond or CondExpr.true()

        v = eval_expr(expr, defined_set, undefined_set)

        # Check if parent if-chain is pending
        parent_pending = False
        if lo.related_if is not None and lo.related_if in if_chain_pending:
            if if_chain_pending[lo.related_if]:
                parent_pending = True

        # 1) Remove inactive lines (only when parent is not pending)
        if v == TriValue.FALSE and not parent_pending:
            idx_to_remove.add(idx)
            continue

        # 2) Remove directive lines that are fully TRUE (except #endif)
        if v == TriValue.TRUE and not parent_pending and (
            lo.is_directive_if()
            or lo.is_directive_ifdef()
            or lo.is_directive_ifndef()
            or lo.is_directive_elif()
            or lo.is_directive_else()
        ):
            idx_to_remove.add(idx)
            continue

        # 3) Special handling for #endif
        if lo.is_directive_endif():
            parent_idx = lo.related_if
            if parent_idx is not None and parent_idx in if_chain_pending:
                if if_chain_pending[parent_idx]:
                    continue
                else:
                    idx_to_remove.add(idx)
                    continue

def filter_output_lines(objs, defined_set, undefined_set, apple_libc_blocks=[]):
    if "--debug" in sys.argv:
        print("===== OBJS DUMP =====")
        for i, lo in enumerate(objs):
            print(f"[{i:03}] {lo.text.rstrip()}   dir={lo.directive}   related_if={lo.related_if}")
        print("======================")

    idx_to_remove = set()

    # Step 1: collapse fully resolved if-chains
    collapse = collapse_fully_resolved_if_chains(objs, defined_set, undefined_set)
    idx_to_remove |= collapse

    # Step 2: detect pending status of each if-chain
    if_chain_pending = compute_if_chain_pending(objs, defined_set, undefined_set, idx_to_remove)

    # Step 2.5: remove FALSE branches entirely (only when -U is used)
    false_branch_remove = remove_false_branches(objs, defined_set, undefined_set)
    idx_to_remove |= false_branch_remove

    # Step 3: normal filtering
    remove_inactive_lines(objs, defined_set, undefined_set, if_chain_pending, idx_to_remove)

    return postprocess_repair_structure(objs, idx_to_remove)

# ------------------------------------------------------------
# postprocess_repair_structure
# ------------------------------------------------------------

def postprocess_repair_structure(objs, removed):
    new_lines = []

    for idx, lo in enumerate(objs):
        if idx in removed:
            continue

        if lo.is_directive_elif():
            parent = lo.related_if
            if parent in removed:
                new_lines.append(f"#if {expr_to_if(lo.local_cond)}")
                continue

        if lo.is_directive_else():
            parent = lo.related_if
            if parent in removed:
                new_lines.append("#else")
                continue

        new_lines.append(lo.text)

    return new_lines

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", action="append", default=[])
    parser.add_argument("-U", action="append", default=[])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    lines = sys.stdin.read().splitlines()

    objs = parse_lines(lines)
    propagate_effective_conds(objs)

    if args.debug:
        for lo in objs:
            print(f"[DEBUG] {lo.text}  →  {lo.effective_cond}")

    output = filter_output_lines(objs, set(args.D), set(args.U))

    for line in output:
        print(line)

if __name__ == "__main__":
    main()

