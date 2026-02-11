#!/usr/bin/env python3
import re
import sys
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Any

# ------------------------------------------------------------
# CondAtom: represents a single macro condition (pending)
# ------------------------------------------------------------

class CondType(Enum):
    DEFINE  = "D"
    UNDEF   = "U"
    NEUTRAL = "N"
    CONST_BOOLEAN = "B"

@dataclass
class CondAtom:
    kind: CondType
    macro: Optional[str]

    def is_const(self):
        return (self.kind == CondType.CONST_BOOLEAN)

    @classmethod
    def define(cls, macro):
        return cls(CondType.DEFINE, macro)

    @classmethod
    def undef(cls, macro):
        return cls(CondType.UNDEF, macro)

    @classmethod
    def neutral(cls, macro):
        return cls(CondType.NEUTRAL, macro)

    def __repr__(self):
        macro = self.macro if self.macro else "*"
        return f"{self.kind.value}:{macro}"

class TrueAtom(CondAtom):
    def __init__(self):
        super().__init__(CondType.CONST_BOOLEAN, None)
        self.value = True

    def __repr__(self):
        return "T"

class FalseAtom(CondAtom):
    def __init__(self):
        super().__init__(CondType.CONST_BOOLEAN, None)
        self.value = False

    def __repr__(self):
        return "F"

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

    def is_atom(self): return self == CondExprKind.ATOM
    def is_unknown(self): return self == CondExprKind.UNKNOWN
    def is_op_not(self): return self == CondExprKind.NOT
    def is_op_and(self): return self == CondExprKind.AND
    def is_op_or(self): return self == CondExprKind.OR

regex_T_and_x = re.compile(r'^\(T\&\&(.+)\)$')
regex_F_and_x = re.compile(r'^\(F\&\&(.+)\)$')
regex_x_and_T = re.compile(r'^\((.+)\&\&T\)$')
regex_x_and_F = re.compile(r'^\((.+)\&\&F\)$')
regex_T_or_x = re.compile(r'^\(T\|\|(.+)\)$')
regex_F_or_x = re.compile(r'^\(F\|\|(.+)\)$')
regex_x_or_T = re.compile(r'^\((.+)\|\|T\)$')
regex_x_or_F = re.compile(r'^\((.+)\|\|F\)$')
def trim_const_boolean(str_expr):
    s = str_expr.strip()
    if m := regex_T_and_x.fullmatch(s):
        return m.group(1)
    elif m := regex_x_and_T.fullmatch(s):
        return m.group(1)
    elif regex_F_and_x.fullmatch(s) or regex_x_and_F.fullmatch(s):
        return "F"
    elif regex_T_or_x.fullmatch(s) or regex_x_or_T.fullmatch(s):
        return "T"
    elif m := regex_F_or_x.fullmatch(s):
        return m.group(1)
    elif m := regex_x_or_F.fullmatch(s):
        return m.group(1)
    else:
        return s
    

@dataclass
class CondExpr:
    kind: CondExprKind
    atom: Optional[CondAtom] = None
    args: Optional[List[Any]] = None

    def macros(self):
        macros = []
        if self.kind.is_unknown():
            pass
        elif self.kind.is_atom() and not self.atom.is_const():
            macros.append(self.atom.macro)
        else:
            for a in self.args:
                if instanceof(a, CondExpr):
                    macros.extend(a.macros())
                elif instanceof(a, CondAtom) and not a.is_const():
                    macros.append(a.macro)
        return macros

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
    def Not(cls, a):
        return cls(CondExprKind.NOT, args = [a])

    @classmethod
    def And(cls, a, b):
        return cls(CondExprKind.AND, args = [a, b])

    @classmethod
    def Or(cls, a, b):
        return cls(CondExprKind.OR, args = [a, b])

    @classmethod
    def Unknown(cls, text):
        return cls(CondExprKind.UNKNOWN, args = [text])

    def __repr__(self):
        if self.kind.is_unknown():
            return f"?:{self.args[0]}"
        if self.kind.is_atom():
            return repr(self.atom)
        if self.kind.is_op_not():
            return f"!{repr(self.args[0])}"
        if self.kind.is_op_and():
            return trim_const_boolean(
                f"({repr(self.args[0])}&&{repr(self.args[1])})"
            )
        if self.kind.is_op_or():
            return trim_const_boolean(
                f"({repr(self.args[0])}||{repr(self.args[1])})"
            )

# ------------------------------------------------------------
# TriValue: 3-valued logic
# ------------------------------------------------------------

class TriValue(Enum):
    TRUE = auto()
    FALSE = auto()
    PENDING = auto()

    def is_true(self): return self == TriValue.TRUE
    def is_false(self): return self == TriValue.FALSE
    def is_pending(self): return self == TriValue.PENDING

regex_color_hex = re.compile(r"^#[0-9A-Fa-f]{6}$")

def parse_hex_color(s):
    if not regex_color_hex.match(s):
        raise SyntaxError(f"{s} does not fit to hexadecimal color code")
    s = s[1:]
    if len(s) != 6:
        raise ValueError("hex color must be 6 hex digits")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)

def color_hex_to_ansi_256(s):
    r, g, b = parse_hex_color(s)
    n = 16 + 36 * (r // 51) + 6 * (g // 51) + (b // 51)
    return f"\033[38;5;{n:03d}m"

def color_hex_to_ansi_rgb(s):
    r, g, b = parse_hex_color(s)
    return f"\033[38;2;{r:03d};{g:03d};{b:03d}m"

# ------------------------------------------------------------
# LineObj: represents one line of source
# ------------------------------------------------------------

class DirectiveKind(Enum):
    NONE     = auto()	# normal content, not pp-directive
    DISABLED = auto()	# pp-directive but do not process (like header guard)
    PP_MISC  = auto()	# pp-directive but unknown/unsupported currently
    IF       = auto()	# if
    IFDEF    = auto()	# ifdef
    IFNDEF   = auto()	# ifndef
    ELIF     = auto()	# elif
    ELSE     = auto()	# else
    ENDIF    = auto()	# endif

    def __repr__(self):
        if self == DirectiveKind.NONE:     return "<none>"
        if self == DirectiveKind.DISABLED: return "<disabled>"
        if self == DirectiveKind.PP_MISC:  return "#pp_misc"
        if self == DirectiveKind.IF:     return "#if"
        if self == DirectiveKind.IFDEF:  return "#ifdef"
        if self == DirectiveKind.IFNDEF: return "#ifndef"
        if self == DirectiveKind.ELIF:   return "#elif"
        if self == DirectiveKind.ELSE:   return "#else"
        if self == DirectiveKind.ENDIF:  return "#endif"
        return "<unknown>"

DEFAULT_COLORS = {
    None:                   "\033[0m",
    DirectiveKind.NONE:     "#000000",
    DirectiveKind.DISABLED: "#333300",
    DirectiveKind.PP_MISC:  "#CCCC99",
    DirectiveKind.IF:       "#00CC99",
    DirectiveKind.IFDEF:    "#009999",
    DirectiveKind.IFNDEF:   "#009966",
    DirectiveKind.ELIF:     "#99CC33",
    DirectiveKind.ELSE:     "#FF00FF",
    DirectiveKind.ENDIF:    "#CC3366"
}

def ppdir_color256(pp_dir, color_dic = DEFAULT_COLORS):
    if pp_dir is None:
        return DEFAULT_COLORS[None]
    elif pp_dir in color_dic:
        return color_hex_to_ansi_256(color_dic[pp_dir])
    else:
        return DEFAULT_COLORS[None]

@dataclass
class LineObj:
    text: str
    debug: dict = field(default_factory=dict)
    directive: DirectiveKind = DirectiveKind.NONE
    blk_hdr_idx: Optional[int] = None
    br_hdr_cond: Optional[CondExpr] = None
    acc_cond: Optional[CondExpr] = None

    def is_directive_if(self): return self.directive == DirectiveKind.IF
    def is_directive_ifdef(self): return self.directive == DirectiveKind.IFDEF
    def is_directive_ifndef(self): return self.directive == DirectiveKind.IFNDEF
    def is_directive_iflike(self):
        return self.directive in ( DirectiveKind.IF,
                                   DirectiveKind.IFDEF,
                                   DirectiveKind.IFNDEF)
    def is_directive_elif(self): return self.directive == DirectiveKind.ELIF
    def is_directive_else(self): return self.directive == DirectiveKind.ELSE
    def is_directive_elselike(self):
        return self.directive in ( DirectiveKind.ELIF,
                                   DirectiveKind.ELSE )
    def is_directive_conditional_entry(self):
        return self.is_directive_iflike() or self.is_directive_elselike()
    def is_directive_endif(self): return self.directive == DirectiveKind.ENDIF
    def is_directive_none(self): return self.directive == DirectiveKind.NONE
    def is_directive_pp_misc(self): return self.directive == DirectiveKind.PP_MISC

    def directive_prefix(self):
        m = regex_directive_prefix.match(self.text)
        return m.group(0) if m else None

# ------------------------------------------------------------
# IfCtx: used during propagation
# ------------------------------------------------------------

@dataclass
class IfCtx:
    outer_acc: CondExpr
    blk_consumed_cond: CondExpr

# ------------------------------------------------------------
# Regex for parsing directives
# ------------------------------------------------------------

regex_directive_prefix = re.compile(r'^\s*#\s*')
regex_defined = re.compile(r'defined\s*\(\s*(\w+)\s*\)')
regex_not_defined = re.compile(r'\!\s*defined\s*\(\s*(\w+)\s*\)')
regex_ifdef  = re.compile(r'^\s*#\s*ifdef\s+(\w+)\b')
regex_ifndef = re.compile(r'^\s*#\s*ifndef\s+(\w+)\b')
regex_if     = re.compile(r'^\s*#\s*if\b(.*)')
regex_elif_defined     = re.compile(r'^\s*#\s*elif\s+defined\s*\(\s*(\w+)\s*\)')
regex_elif_not_defined = re.compile(r'^\s*#\s*elif\s+!\s*defined\s*\(\s*(\w+)\s*\)')
regex_elif   = re.compile(r'^\s*#\s*elif\b(.*)')
regex_else   = re.compile(r'^\s*#\s*else\b')
regex_endif  = re.compile(r'^\s*#\s*endif\b')
regex_misc   = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')

def assign_blk_hdr_idx(lo, if_stack, objs, idx):
    if not if_stack:
        raise SyntaxError(f"Unmatched directive at line {idx+1}: {lo.text}")
    lo.blk_hdr_idx = if_stack[-1]
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
            lo.br_hdr_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            lo.blk_hdr_idx = idx
            if_stack.append(idx)
            continue

        elif m := regex_ifndef.match(line):
            lo.directive = DirectiveKind.IFNDEF
            lo.br_hdr_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            lo.blk_hdr_idx = idx
            if_stack.append(idx)
            continue

        elif m := regex_if.match(line):
            lo.directive = DirectiveKind.IF
            expr_after_if = m.group(1).strip()
            if m2 := regex_defined.fullmatch(expr_after_if):
                macro = m2.group(1)
                lo.br_hdr_cond = CondExpr.atom_expr(CondAtom.neutral(macro))
            elif m2 := regex_not_defined.fullmatch(expr_after_if):
                macro = m2.group(1)
                lo.br_hdr_cond = CondExpr.Not(
                    CondExpr.atom_expr(CondAtom.neutral(macro))
                )
            else:
                lo.br_hdr_cond = CondExpr.Unknown(expr_after_if)

            lo.blk_hdr_idx = idx
            if_stack.append(idx)
            continue

        elif m := regex_elif_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            assign_blk_hdr_idx(lo, if_stack, objs, idx)
            lo.br_hdr_cond = CondExpr.atom_expr(CondAtom.neutral(m.group(1)))
            continue

        elif m := regex_elif_not_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            assign_blk_hdr_idx(lo, if_stack, objs, idx)
            macro = m.group(1)
            lo.br_hdr_cond = CondExpr.Not(
                CondExpr.atom_expr(CondAtom.neutral(macro))
            )
            continue

        elif m := regex_elif.match(line):
            lo.directive = DirectiveKind.ELIF
            assign_blk_hdr_idx(lo, if_stack, objs, idx)
            lo.br_hdr_cond = CondExpr.Unknown(m.group(1).strip())
            continue

        elif regex_else.match(line):
            lo.directive = DirectiveKind.ELSE
            assign_blk_hdr_idx(lo, if_stack, objs, idx)
            lo.br_hdr_cond = CondExpr.true()
            continue

        elif regex_endif.match(line):
            lo.directive = DirectiveKind.ENDIF
            assign_blk_hdr_idx(lo, if_stack, objs, idx)
            lo.br_hdr_cond = CondExpr.true()
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
# propagate_acc_cond
# ------------------------------------------------------------

def propagate_acc_cond(objs):
    current_acc = CondExpr.true()
    ctx_stack = []

    for lo in objs:

        if lo.is_directive_none() or lo.is_directive_pp_misc():
            lo.acc_cond = current_acc

        elif lo.is_directive_iflike():
            cond_if = lo.br_hdr_cond or CondExpr.true()
            lo.acc_cond = CondExpr.And(current_acc, cond_if)

            if_ctx = IfCtx(outer_acc = current_acc, blk_consumed_cond = cond_if)
            ctx_stack.append(if_ctx)

            current_acc = lo.acc_cond

        elif lo.is_directive_elif():
            if_ctx = ctx_stack[-1]
            cond_elif = lo.br_hdr_cond or CondExpr.true()

            br_hdr_cond = CondExpr.And(cond_elif, CondExpr.Not(if_ctx.blk_consumed_cond))
            lo.acc_cond = CondExpr.And(if_ctx.outer_acc, br_hdr_cond)

            if_ctx.blk_consumed_cond = CondExpr.Or(if_ctx.blk_consumed_cond, cond_elif)
            current_acc = lo.acc_cond

        elif lo.is_directive_else():
            if_ctx = ctx_stack[-1]
            br_hdr_cond = CondExpr.Not(if_ctx.blk_consumed_cond)
            lo.acc_cond = CondExpr.And(if_ctx.outer_acc, br_hdr_cond)
            current_acc = lo.acc_cond

        elif lo.is_directive_endif():
            if_ctx = ctx_stack.pop()
            lo.acc_cond = if_ctx.outer_acc
            current_acc = if_ctx.outer_acc

        if lo.acc_cond is None:
            lo.debug["acc_cond"] = ""
        else:
            lo.debug["acc_cond"] = repr(lo.acc_cond)


# ------------------------------------------------------------
# eval_atom
# ------------------------------------------------------------

def eval_atom(atom, defined_set, undefined_set):
    macro = atom.macro

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
    if expr.kind.is_unknown(): return TriValue.PENDING
    if expr.kind.is_atom():
        return eval_atom(expr.atom, defined_set, undefined_set)

    vs = [
        eval_expr(a, defined_set, undefined_set)
        for a in expr.args
    ]

    if expr.kind.is_op_not():
        if vs[0].is_true(): return TriValue.FALSE
        if vs[0].is_false(): return TriValue.TRUE
        return TriValue.PENDING

    if expr.kind.is_op_and():
        if any(v.is_false() for v in vs): return TriValue.FALSE
        if all(v.is_true() for v in vs): return TriValue.TRUE
        return TriValue.PENDING

    if expr.kind.is_op_or():
        if any(v.is_true() for v in vs): return TriValue.TRUE
        if all(v.is_false() for v in vs): return TriValue.FALSE
        return TriValue.PENDING


# ------------------------------------------------------------
# expr_to_if (minimal)
# ------------------------------------------------------------

def expr_to_if(expr):
    if expr.kind.is_atom():
        atom = expr.atom
        if atom.kind == CondType.DEFINE:
            return f"defined({atom.macro})"
        if atom.kind == CondType.UNDEF:
            return f"!defined({atom.macro})"
        if atom.kind == CondType.NEUTRAL:
            return f"defined({atom.macro}) /* pending */"
        if atom.kind == CondType.CONST_BOOLEAN:
            return "1" if atom is TRUE_ATOM else "0"

    if expr.kind.is_op_not():
        return f"!({expr_to_if(expr.args[0])})"

    if expr.kind.is_op_and():
        return f"({expr_to_if(expr.args[0])} && {expr_to_if(expr.args[1])})"

    if expr.kind.is_op_or():
        return f"({expr_to_if(expr.args[0])} || {expr_to_if(expr.args[1])})"

    # here, assume expr.kind.is_unknown()
    return f"/* {expr.args[0]} */"

# ------------------------------------------------------------
# filter_output_lines
# ------------------------------------------------------------

def collect_if_blocks(objs):
    """
    return:
        if_blocks: dict
            if_idx -> {
                "branches": [branch_start_idx, ...],
                "end": endif_idx
            }
    """
    if_blocks = {}
    stack = []

    for idx, lo in enumerate(objs):
        if lo.is_directive_iflike():
            if_blocks[idx] = {"branches": [idx], "end": None}
            stack.append(idx)

        elif lo.is_directive_elselike():
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx in if_blocks:
                if_blocks[blk_hdr_idx]["branches"].append(idx)

        elif lo.is_directive_endif():
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx in if_blocks:
                if_blocks[blk_hdr_idx]["end"] = idx
            if stack:
                stack.pop()

    return if_blocks

def collapse_fully_resolved_if_blocks(objs, defined_set, undefined_set):
    to_remove = set()
    if_blocks = collect_if_blocks(objs)

    for if_idx, info in if_blocks.items():
        branches = info["branches"]
        end = info["end"]
        if end is None:
            continue

        results = []
        for b in branches:
            expr = objs[b].acc_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            results.append((b, v))

        true_branches = [b for b, v in results if v.is_true()]
        pending = any(v == TriValue.PENDING for _, v in results)

        if len(true_branches) == 1 and not pending:
            # remove directives only
            for i in range(if_idx, end + 1):
                lo = objs[i]
                if lo.blk_hdr_idx == if_idx:
                    to_remove.add(i)

    return to_remove

def remove_false_branches(objs, defined_set, undefined_set):
    to_remove = set()
    if_blocks = collect_if_blocks(objs)

    for if_idx, info in if_blocks.items():
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
            expr = objs[start].acc_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            if v == TriValue.FALSE:
                for i in range(start, stop + 1):
                    to_remove.add(i)

    for idx, lo in enumerate(objs):
        lo.debug["false_branch"] = "T" if idx in to_remove else "F"

    return to_remove

def compute_if_block_pending(objs, defined_set, undefined_set, idx_to_remove):
    if_block_pending = {}

    for idx, lo in enumerate(objs):
        if idx in idx_to_remove:
            continue

        if lo.is_directive_iflike():
            expr = lo.acc_cond or CondExpr.true()
            v = eval_expr(expr, defined_set, undefined_set)
            if_block_pending[idx] = (v == TriValue.PENDING)

        elif lo.is_directive_elselike():
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx is not None and blk_hdr_idx in if_block_pending:
                expr = lo.acc_cond or CondExpr.true()
                v = eval_expr(expr, defined_set, undefined_set)
                if v == TriValue.PENDING:
                    if_block_pending[blk_hdr_idx] = True

    for idx, lo in enumerate(objs):
        if not idx in if_block_pending:
            lo.debug["pending"] = "_"
        elif if_block_pending[idx] is True:
            lo.debug["pending"] = "T"
        elif if_block_pending[idx] is False:
            lo.debug["pending"] = "F"
        else:
            lo.debug["pending"] = "?"

    return if_block_pending

def remove_inactive_lines(objs, defined_set, undefined_set, if_block_pending, idx_to_remove):
    for idx, lo in enumerate(objs):
        if idx in idx_to_remove:
            continue

        # Directives use br_hdr_cond; normal lines use acc_cond
        if lo.is_directive_conditional_entry():
            expr = lo.br_hdr_cond or CondExpr.true()
        else:
            expr = lo.acc_cond or CondExpr.true()

        v = eval_expr(expr, defined_set, undefined_set)

        # Check if current if-block is pending
        cur_if_block_pending = False
        if lo.blk_hdr_idx is not None and lo.blk_hdr_idx in if_block_pending:
            if if_block_pending[lo.blk_hdr_idx]:
                cur_if_block_pending = True

        # 1) Remove inactive lines (only when current block is not pending)
        if v == TriValue.FALSE and not cur_if_block_pending:
            idx_to_remove.add(idx)
            continue

        # 2) Remove directive lines that are fully TRUE (except #endif)
        if v.is_true() and not cur_if_block_pending and lo.is_directive_conditional_entry():
            idx_to_remove.add(idx)
            continue

        # 3) Special handling for #endif
        if lo.is_directive_endif():
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx is not None and blk_hdr_idx in if_block_pending:
                if if_block_pending[blk_hdr_idx]:
                    continue
                else:
                    idx_to_remove.add(idx)
                    continue

    for idx, lo in enumerate(objs):
        lo.debug["inactive_branch"] = "T" if idx in idx_to_remove else "F"

def max_width(itr):
    return max(len(s) for s in itr)

def debug_column_width(objs, k):
    return max_width(lo.debug[k] for lo in objs)

def directive_column_width(objs):
    return max_width(repr(lo.directive) for lo in objs)

def filter_output_lines(objs, defined_set, undefined_set, apple_libc_blocks=[], debug = False):
    if debug:
        print("===== OBJS DUMP =====")
        max_width_dir = directive_column_width(objs)
        s_clr256_rst = ppdir_color256(None)
        for i, lo in enumerate(objs):
            s_blk_hdr_idx = f"{lo.blk_hdr_idx:03}" if lo.blk_hdr_idx else "_"
            s_clr256 = ppdir_color256(lo.directive)
            print(f"[{i:03}] "
                  f"dir={s_clr256}{repr(lo.directive):{max_width_dir}s}{s_clr256_rst} "
                  f"blk_hdr={s_blk_hdr_idx} "
                  f"    {lo.text.rstrip()}")
        print("======================")

    idx_to_remove = set()

    # Step 1: collapse fully resolved if-blocks
    collapse = collapse_fully_resolved_if_blocks(objs, defined_set, undefined_set)
    idx_to_remove |= collapse

    # Step 2: detect pending status of each if-block
    if_block_pending = compute_if_block_pending(objs, defined_set, undefined_set, idx_to_remove)

    # Step 2.5: remove FALSE branches entirely (only when -U is used)
    false_branch_remove = remove_false_branches(objs, defined_set, undefined_set)
    idx_to_remove |= false_branch_remove

    # Step 3: normal filtering
    remove_inactive_lines(objs, defined_set, undefined_set, if_block_pending, idx_to_remove)

    if debug:
        max_width_acc = debug_column_width(objs, "acc_cond")
        for lo in objs:
            if lo.debug['inactive_branch'] == 'T':
                str_act_color = '\033[31mx\033[0m'
            else:
                str_act_color = '\033[32mO\033[0m'
            print(f"[DEBUG] acc={lo.debug['acc_cond']:{max_width_acc}} "
                  f"pnd={lo.debug['pending']} "
                  f"act={str_act_color} "
                  f"→ {lo.text}")

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
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx in removed:
                new_lines.append(f"{lo.directive_prefix()}if {expr_to_if(lo.br_hdr_cond)}")
                continue

        if lo.is_directive_else():
            blk_hdr_idx = lo.blk_hdr_idx
            if blk_hdr_idx in removed:
                new_lines.append(f"{lo.directive_prefix()}else")
                continue

        new_lines.append(lo.text)

    return new_lines

# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def detect_header_guard(objs, debug = False):
    if debug:
        print("start detect_header_guard()")

    # lookup last endif
    endif_idx = None
    for i in reversed(range(len(objs))):
        if objs[i].is_directive_endif():
            endif_idx = i
            break

    if endif_idx is None:
        return None

    if debug:
        print(f"detect_header_guard: last endif is at {endif_idx}")

    endif_obj = objs[endif_idx]

    # if no blk_hdr_idx for the last endif, the source is broken
    if endif_obj.blk_hdr_idx is None:
        return None

    if_idx = endif_obj.blk_hdr_idx
    if_obj = objs[if_idx]

    if debug:
        print(f"detect_header_guard: condition starts at {if_idx}")

    # confirm related pp is '#ifndef X' or '#if !defined(X)'
    if if_obj.directive not in (DirectiveKind.IFNDEF, DirectiveKind.IF):
        return None

    if debug:
        print(f"detect_header_guard: {if_idx} is '#ifndef' or '#if defined'")

    # extract a macro used for header guard
    #   confirm br_hdr_cond has one macro only
    br_hdr_macros = if_obj.br_hdr_cond.macros()
    if len(br_hdr_macros) != 1:
        return None

    macro = br_hdr_macros[0]

    if debug:
        print(f"detect_header_guard: {macro} is header guard macro")

    # lookup next pp after first #ifndef or #if !defined()
    regex_cpp_directive = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')

    j = if_idx + 1
    while j < len(objs) and not regex_cpp_directive.match(objs[j].text):
        j += 1

    if j >= len(objs):
        return None

    if debug:
        print(f"detect_header_guard: {macro} might be redefined at {j}")

    # confirm the 2nd pp defines the macro used for header guard
    regex_define_macro = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)')
    m = regex_define_macro.match(objs[j].text)
    if m is None:
        return None

    if m.group(1) != macro:
        return None

    define_idx = j

    if debug:
        print(f"detect_header_guard: {macro} is confirmed to be redefined at {j}")

    # confirm first #ifndef or #if !defined(X) is the first pp.
    for k in range(if_idx):
        if objs[k].directive is not DirectiveKind.NONE:
            # if there is earlier pp, this might not be header guard.
            if debug:
                print(f"detect_header_guard: first cpp directive is found at {k}")
            return None

    if debug:
        print(f"detect_header_guard: {if_idx} is confirmed to be the first cpp directive")

    # decide this is header guard.
    return (if_idx, define_idx, endif_idx, macro)

def get_words_from_file(path_file):
    words = []
    with open(path_file, "r") as fh:
        for line in fh.read().splitlines():
            tok = re.sub(r"#.*", "", line).strip()
            if tok:
                words.append(tok)
    return words

def open_fh_to_write(do_mkdir, str_dest_dir, str_path, file_suffix):
    if str_path in [None, "-", 0, "stdout", "/dev/stdout", sys.stdout]:
        return ("/dev/stdout", sys.stdout)
    elif file_suffix:
        pth = Path( str_path + file_suffix )
    else:
        pth = Path( str_path )

    if str_dest_dir:
        pth = Path( str_dest_dir ) / pth

    if do_mkdir:
        pth.parent.mkdir( parents = True, exist_ok = True )

    fh = open( pth, "w" )
    print(f"# write {pth}", file = sys.stderr)
    return ( str(pth), fh )

def collect_apple_libc_blocks(lines):
    regex_begin_libc = re.compile(r"^\s*//\s*Begin-Libc\s*$")
    regex_end_libc   = re.compile(r"^\s*//\s*End-Libc\s*$")
    blocks = []
    for idx, line in enumerate(lines):
        if regex_begin_libc.match(line):
            blocks.append( AppleLibcBlock(idx, None, True) )
        elif regex_end_libc.match(line):
            if blocks[-1].end_libc:
                blocks.append( AppleLibcBlock(None, None, True) )
            blocks[-1].end_libc = idx
    blocks = [blk for blk in blocks if blk.begin_libc and blk.end_libc]
    return blocks

def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawTextHelpFormatter
    )
    parser.add_argument("-D", action = "append", default = [],
                        help = "Define macro")
    parser.add_argument("-U", action = "append", default = [],
                        help = "Undefine macro")
    parser.add_argument("-o", type = str, default = None,
                        help = "Write output to <pathname>")
    parser.add_argument("--dest-dir", "--destdir", type = str, default = None,
        help = "Set DESTDIR to emit output files")
    parser.add_argument("--debug", action = "store_true",
        help = "Debug mode")
    parser.add_argument("--patch", action = "store_true",
        help = "Emit as an unified patch")
    parser.add_argument("--patch-output", type = str,
        help = "Emit unified patch output to file")
    parser.add_argument("--patch-suffix", type = str, default = ".diff",
        help = "Set suffix for patch output")
    parser.add_argument("--no-mkdir", action = "store_true",
        help = "Do not make parental directories during output")
    parser.add_argument("--analyze-header-guard", action = "store_true",
        help = "Treat header guards (#ifndef X, #define X, #endif) as normal conditions.\n"
               "By default, macros used in header guards cannot be defined by -D.")
    parser.add_argument("--list-macros-define", type = str,
        help = "Read file of symbols to be defined")
    parser.add_argument("--list-macros-undefine", type = str,
        help = "Read file of symbols to be undefined")

    args, rest = parser.parse_known_args()
    fh_in = None
    path_in = "/dev/stdin"
    for tok in rest:
        if tok.startswith("-D") and len(tok) > 2:
            args.D.append(tok[2:])
        elif tok.startswith("-U") and len(tok) > 2:
            args.U.append(tok[2:])
        elif os.path.exists(tok) and os.access(tok, os.R_OK):
            if fh_in is None:
                path_in = tok
                fh_in = open(tok, "r")
            else:
                sys.stderr.write("Cannot accept multiple input files")
                return -1

    if args.list_macros_define:
        args.D += get_words_from_file(args.list_macros_define)

    if args.list_macros_undefine:
        args.U += get_words_from_file(args.list_macros_undefine)

    if fh_in is None:
        fh_in = sys.stdin


    if args.patch:
        path_patch, fh_patch = open_fh_to_write( not(args.no_mkdir), args.dest_dir,
                                                 args.patch_output if args.patch_output else args.o,
                                                 None if args.patch_output else args.patch_suffix )
    else:
        path_out, fh_out = open_fh_to_write( not(args.no_mkdir), args.dest_dir, args.o, None )

    lines = fh_in.read().splitlines()
    apple_libc_blocks = collect_apple_libc_blocks(lines)
    objs = parse_lines(lines)

    if not args.analyze_header_guard:
        guard = detect_header_guard(objs, args.debug)
        if guard is not None:
            guard_start, guard_define, guard_end, guard_macro = guard
            for idx in [guard_start, guard_define, guard_end]:
                objs[idx].directive   = DirectiveKind.DISABLED
                objs[idx].br_hdr_cond = None

    propagate_acc_cond(objs)

    source_processed = filter_output_lines(objs, args.D, args.U, apple_libc_blocks, args.debug)

    if args.patch:
        source_original = [lo.text for lo in objs]
        for line in difflib.unified_diff( source_original,
                                          source_processed,
                                          fromfile = path_in,
                                          tofile = path_in if args.o is None else args.o,
                                          lineterm = "" ):
            print(line, file = fh_patch)
        if fh_patch != sys.stdout:
            fh_patch.close()
    else:
        for line in source_processed:
            print(line, file = fh_out)
        if fh_out != sys.stdout:
            fh_out.close()

if __name__ == "__main__":
    main()

