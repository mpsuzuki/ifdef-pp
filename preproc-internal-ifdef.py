#!/usr/bin/env python3
import argparse
import sys
import re
import difflib
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

class DirectiveKind(Enum):
    NONE = auto()
    DISABLED = auto()
    PP_MISC = auto()
    IF = auto()
    IFDEF = auto()
    IFNDEF = auto()
    ELIF = auto()
    ELSE = auto()
    ENDIF = auto()

class CondType(Enum):
    DEFINE = "D"     # D:A
    UNDEF = "U"      # U:A
    COMPLEX = "C"    # C:*
    NEUTRAL = "N"    # N:A or N:*

@dataclass
class CondAtom:
    kind: CondType
    macro: Optional[str]  # None for COMPLEX or NEUTRAL with "*"

@dataclass
class LineObj:
    text: str
    directive: DirectiveKind = DirectiveKind.NONE
    related_if: Optional[int] = None
    local_conds: List[CondAtom] = field(default_factory=list)
    effective_conds: List[CondAtom] = field(default_factory=list)

# 正規表現
re_misc   = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')
re_ifdef  = re.compile(r'^\s*#\s*ifdef\s+(\w+)\b')
re_ifndef = re.compile(r'^\s*#\s*ifndef\s+(\w+)\b')
re_if     = re.compile(r'^\s*#\s*if\b(.*)')
re_elif_defined = re.compile(r'^\s*#\s*elif\s+defined\s*\(\s*(\w+)\s*\)')
re_elif_not_defined = re.compile(r'^\s*#\s*elif\s+!\s*defined\s*\(\s*(\w+)\s*\)')
re_elif   = re.compile(r'^\s*#\s*elif\b(.*)')
re_else   = re.compile(r'^\s*#\s*else\b')
re_endif  = re.compile(r'^\s*#\s*endif\b')

def parse_stdin():
    objs: List[LineObj] = []
    if_stack: List[int] = []

    lines = sys.stdin.read().splitlines()

    for idx, line in enumerate(lines):
        lo = LineObj(text=line)

        # #ifdef
        if m := re_ifdef.match(line):
            lo.directive = DirectiveKind.IFDEF
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.DEFINE, macro))
            if_stack.append(idx)

        # #ifndef
        elif m := re_ifndef.match(line):
            lo.directive = DirectiveKind.IFNDEF
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.UNDEF, macro))
            if_stack.append(idx)

        # #if（複雑条件）
        elif m := re_if.match(line):
            lo.directive = DirectiveKind.IF
            lo.local_conds.append(CondAtom(CondType.COMPLEX, None))
            if_stack.append(idx)

        # #elif defined(X)
        elif m := re_elif_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]
            # 親条件の反転
            for atom in parent.local_conds:
                if atom.kind == CondType.DEFINE:
                    lo.local_conds.append(CondAtom(CondType.UNDEF, atom.macro))
                elif atom.kind == CondType.UNDEF:
                    lo.local_conds.append(CondAtom(CondType.DEFINE, atom.macro))
                else:
                    lo.local_conds.append(CondAtom(CondType.COMPLEX, None))

            # さらに D:B を追加
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.DEFINE, macro))

        # #elif !defined(X)
        elif m := re_elif_not_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]
            # 親条件の反転
            for atom in parent.local_conds:
                if atom.kind == CondType.DEFINE:
                    lo.local_conds.append(CondAtom(CondType.UNDEF, atom.macro))
                elif atom.kind == CondType.UNDEF:
                    lo.local_conds.append(CondAtom(CondType.DEFINE, atom.macro))
                else:
                    lo.local_conds.append(CondAtom(CondType.COMPLEX, None))

            # さらに U:B を追加
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.UNDEF, macro))

        # #elif（複雑条件扱い）
        elif m := re_elif.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]
            lo.local_conds.append(CondAtom(CondType.COMPLEX, None))

        # #else（反転処理）
        elif re_else.match(line):
            lo.directive = DirectiveKind.ELSE
            if not if_stack:
                raise SyntaxError(f"unmatched #else at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]
            for atom in parent.local_conds:
                if atom.kind == CondType.DEFINE:
                    lo.local_conds.append(CondAtom(CondType.UNDEF, atom.macro))
                elif atom.kind == CondType.UNDEF:
                    lo.local_conds.append(CondAtom(CondType.DEFINE, atom.macro))
                else:
                    lo.local_conds.append(CondAtom(CondType.COMPLEX, None))

        # #endif（NEUTRAL）
        elif re_endif.match(line):
            lo.directive = DirectiveKind.ENDIF
            if not if_stack:
                raise SyntaxError(f"unmatched #endif at line {idx+1}")

            related = if_stack.pop()
            lo.related_if = related

            parent = objs[related]
            for atom in parent.local_conds:
                lo.local_conds.append(
                    CondAtom(CondType.NEUTRAL, atom.macro)
                )

        # #define, #undef, #include, #pragma, #error, #line, etc are marked but not parsed.
        elif re_misc.match(line):
            lo.directive = DirectiveKind.PP_MISC

        objs.append(lo)

    if if_stack:
        raise SyntaxError("unclosed #if block(s)")

    return objs


# 🌟 ここから effective_conds の伝播
def propagate_effective_conditions(objs: List[LineObj]):
    cond_stack: List[List[CondAtom]] = [[]]

    for idx, lo in enumerate(objs):

        if lo.directive in (DirectiveKind.IFDEF, DirectiveKind.IFNDEF):
            macro = lo.local_conds[0].macro
            lo.effective_conds = [CondAtom(CondType.NEUTRAL, macro)]
            new_layer = cond_stack[-1] + lo.local_conds
            cond_stack.append(new_layer)
            continue

        elif lo.directive == DirectiveKind.IF:
            # 複雑条件は構造行ではない
            lo.effective_conds = cond_stack[-1][:]
            new_layer = cond_stack[-1] + lo.local_conds
            cond_stack.append(new_layer)
            continue

        elif lo.directive == DirectiveKind.ELIF:
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            # effective_conds = N:MACRO（親の local_conds のマクロ）＋ N:MACRO（自分の local_conds のマクロ）
            lo.effective_conds = []

            # 親の local_conds → N:MACRO
            for atom in parent.local_conds:
                lo.effective_conds.append(CondAtom(CondType.NEUTRAL, atom.macro))

            # 自分の local_conds（D:B など）→ N:B
            for atom in lo.local_conds:
                if atom.macro is not None:
                    lo.effective_conds.append(CondAtom(CondType.NEUTRAL, atom.macro))

            # 反転 + 自分の local_conds で新しい条件レイヤを作る
            reversed_parent = []
            for atom in parent.local_conds:
                if atom.kind == CondType.DEFINE:
                    reversed_parent.append(CondAtom(CondType.UNDEF, atom.macro))
                elif atom.kind == CondType.UNDEF:
                    reversed_parent.append(CondAtom(CondType.DEFINE, atom.macro))
                else:
                    reversed_parent.append(CondAtom(CondType.COMPLEX, None))

            new_layer = cond_stack[-2] + reversed_parent + lo.local_conds
            cond_stack[-1] = new_layer
            continue

        elif lo.directive == DirectiveKind.ELSE:
            # effective_conds = N:MACRO（反転元のマクロ）
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            lo.effective_conds = []
            for atom in parent.local_conds:
                lo.effective_conds.append(CondAtom(CondType.NEUTRAL, atom.macro))

            # 反転条件で新しいレイヤを作る
            reversed_parent = []
            for atom in parent.local_conds:
                if atom.kind == CondType.DEFINE:
                    reversed_parent.append(CondAtom(CondType.UNDEF, atom.macro))
                elif atom.kind == CondType.UNDEF:
                    reversed_parent.append(CondAtom(CondType.DEFINE, atom.macro))
                else:
                    reversed_parent.append(CondAtom(CondType.COMPLEX, None))

            new_layer = cond_stack[-2] + reversed_parent
            cond_stack[-1] = new_layer
            continue

        elif lo.directive == DirectiveKind.ENDIF:
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            # effective_conds = N:MACRO（親の local_conds のマクロ）
            lo.effective_conds = [
                CondAtom(CondType.NEUTRAL, atom.macro)
                for atom in parent.local_conds
            ]

            cond_stack.pop()
            continue

        else:
            lo.effective_conds = cond_stack[-1][:]

def eval_effective_conds(effective_conds, defined_set, undefined_set):

    for atom in effective_conds:
        macro = atom.macro

        if atom.kind == CondType.COMPLEX:
            continue

        if atom.kind == CondType.NEUTRAL:
            if macro in defined_set or macro in undefined_set:
                return False
            continue

        if atom.kind == CondType.DEFINE:
            if macro in defined_set:
                continue
            if macro in undefined_set:
                return False
            continue

        if atom.kind == CondType.UNDEF:
            if macro in defined_set:
                return False
            if macro in undefined_set:
                continue
            continue

    return True

def filter_output_lines(objs, defined_set, undefined_set):
    out = []
    for lo in objs:
        if eval_effective_conds(lo.effective_conds, defined_set, undefined_set):
            out.append(lo.text)
    return out

def detect_header_guard(objs, debug = False):
    print("start detect_header_guard()")
    """
    ヘッダガードを検出する。
    見つかった場合は (if_idx, define_idx, endif_idx, macro) を返す。
    見つからなければ None を返す。
    """

    # --- 1. 最後の #endif を探す ---
    endif_idx = None
    for i in reversed(range(len(objs))):
        if objs[i].directive == DirectiveKind.ENDIF:
            endif_idx = i
            break

    if endif_idx is None:
        return None

    if debug:
        print(f"detect_header_guard: last endif is at {endif_idx}")

    endif_obj = objs[endif_idx]

    # related_if が無い場合は不正
    if endif_obj.related_if is None:
        return None

    if_idx = endif_obj.related_if
    if_obj = objs[if_idx]

    if debug:
        print(f"detect_header_guard: condition starts at {if_idx}")

    # --- 2. #ifndef X または #if !defined(X) であることを確認 ---
    if if_obj.directive not in (DirectiveKind.IFNDEF, DirectiveKind.IF):
        return None

    if debug:
        print(f"detect_header_guard: {if_idx} is '#ifndef' or '#if defined'")

    # local_conds からマクロ名を取り出す
    # ifndef → U:X が1つだけ
    # if !defined(X) → U:X が1つだけ
    if len(if_obj.local_conds) != 1:
        return None

    cond_atom = if_obj.local_conds[0]
    macro = cond_atom.macro

    if debug:
        print(f"detect_header_guard: {macro} is header guard macro")

    # --- 3. #ifndef の次のプリプロセッサ指示子を探す ---
    re_cpp_directive = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')

    j = if_idx + 1
    while j < len(objs) and not re_cpp_directive.match(objs[j].text):
        j += 1

    if j >= len(objs):
        return None

    if debug:
        print(f"detect_header_guard: {macro} might be redefined at {j}")

    # --- 4. #define X であることを確認 ---
    re_define_macro = re.compile(r'^\s*#\s*define\s+([A-Za-z_]\w*)')
    m = re_define_macro.match(objs[j].text)
    if m is None:
        return None

    if m.group(1) != macro:
        return None

    define_idx = j

    if debug:
        print(f"detect_header_guard: {macro} is confirmed to be redefined at {j}")

    # --- 5. #ifndef がファイル最初の指示子であることを確認（任意） ---
    # これを入れると誤検出が減る
    for k in range(if_idx):
        if objs[k].directive is not DirectiveKind.NONE:
            # 最初の指示子ではない → ヘッダガードではない
            if debug:
                print(f"detect_header_guard: first cpp directive is found at {k}")
            return None

    if debug:
        print(f"detect_header_guard: {if_idx} is confirmed to be the first cpp directive")

    # --- 6. すべてパスしたのでヘッダガードと判定 ---
    return (if_idx, define_idx, endif_idx, macro)

def main():
    parser = argparse.ArgumentParser(
        formatter_class = argparse.RawTextHelpFormatter
    )
    parser.add_argument("-D", action = "append", default = [],
                        help = "define macro")
    parser.add_argument("-U", action = "append", default = [],
                        help = "undefine macro")
    parser.add_argument("--debug", action = "store_true",
        help = "Debug mode")
    parser.add_argument("--patch", action = "store_true",
        help = "Emit as an unified patch")
    parser.add_argument("--analyze-header-guard", action = "store_true",
        help = "Treat header guards (#ifndef X, #define X, #endif) as normal conditions.\n"
               "By default, macros used in header guards cannot be defined by -D.")

    args, rest = parser.parse_known_args()
    for tok in rest:
        if tok.startswith("-D") and len(tok) > 2:
            args.D.append(tok[2:])
        elif tok.startswith("-U") and len(tok) > 2:
            args.U.append(tok[2:])

    objs = parse_stdin()

    if not args.analyze_header_guard:
        guard = detect_header_guard(objs, args.debug)
        if guard is not None:
            guard_start, guard_define, guard_end, guard_macro = guard
            for idx in [guard_start, guard_define, guard_end]:
                objs[idx].directive   = DirectiveKind.DISABLED
                objs[idx].local_conds = []

    propagate_effective_conditions(objs)

    if args.debug:
        for lo in objs:
            local = ",".join(f"{a.kind.value}:{a.macro or '*'}" for a in lo.local_conds)
            eff   = ",".join(f"{a.kind.value}:{a.macro or '*'}" for a in lo.effective_conds)
            print(f"[DEBUG] {local:20s} {eff:20s} {lo.text}")

    source_processed = filter_output_lines(objs, args.D, args.U)

    if args.patch:
        source_original = [lo.text for lo in objs]
        for line in difflib.unified_diff( source_original,
                                          source_processed,
                                          fromfile = "a.txt",
                                          tofile = "b.txt",
                                          lineterm = "" ):
            print(line)
    else:
        for line in source_processed:
            print(line)

if __name__ == "__main__":
    main()

