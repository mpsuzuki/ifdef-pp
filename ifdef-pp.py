#!/usr/bin/env python3
import argparse
import sys
import os
import re
import difflib
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

class DirectiveKind(Enum):
    NONE     = auto()	# normal content, not preprocessor directive
    DISABLED = auto()	# preprocessor directive, known type, but disabled
    PP_MISC  = auto()	# preprocessor directive, unknown type, would not be treated
    IF       = auto()	# if
    IFDEF    = auto()	# ifdef
    IFNDEF   = auto()	# ifndef
    ELIF     = auto()	# elif
    ELSE     = auto()	# else
    ENDIF    = auto()	# endif

class CondType(Enum):
    DEFINE  = "D"	# D:A (emit this line if A is defined or neutral)
    UNDEF   = "U"	# U:A (emit this line if A is undefined or neutral)
    COMPLEX = "C"	# C:* (emit this line always - condition is complex)
    NEUTRAL = "N"	# N:A or N:* (emit this line if A is neutral)

@dataclass
class CondAtom:
    kind: CondType
    macro: Optional[str]  # None for COMPLEX or NEUTRAL with "*"

    @classmethod
    def define(cls, macro):
        return cls(CondType.DEFINE, macro)

    @classmethod
    def undef(cls, macro):
        return cls(CondType.UNDEF, macro)

    @classmethod
    def complex(cls, macro = None):
        return cls(CondType.COMPLEX, macro)

    @classmethod
    def neutral(cls, macro):
        return cls(CondType.NEUTRAL, macro)

    def is_define(self):
        return self.kind == CondType.DEFINE
    def is_undef(self):
        return self.kind == CondType.UNDEF
    def is_complex(self):
        return self.kind == CondType.COMPLEX
    def is_neutral(self):
        return self.kind == CondType.NEUTRAL

    def has_macro(self):
        return self.macro is not None

    def negated(self, keep_macro = False):
        if self.is_define():
            return CondAtom.undef(self.macro)
        if self.is_undef():
            return CondAtom.define(self.macro)
        return CondAtom.neutral(self.macro if keep_macro else None)

    def neutralized(self):
        return CondAtom.neutral(self.macro)

@dataclass
class LineObj:
    text: str
    directive: DirectiveKind = DirectiveKind.NONE
    related_if: Optional[int] = None
    local_conds: List[CondAtom] = field(default_factory=list)
    effective_conds: List[CondAtom] = field(default_factory=list)
    def is_directive_none(self):
        return self.directive == DirectiveKind.NONE
    def is_directive_disabled(self):
        return self.directive == DirectiveKind.DISABLED
    def is_directive_pp_misc(self):
        return self.directive == DirectiveKind.PP_MISC
    def is_directive_if(self):
        return self.directive == DirectiveKind.IF
    def is_directive_ifdef(self):
        return self.directive == DirectiveKind.IFDEF
    def is_directive_ifndef(self):
        return self.directive == DirectiveKind.IFNDEF
    def is_directive_elif(self):
        return self.directive == DirectiveKind.ELIF
    def is_directive_else(self):
        return self.directive == DirectiveKind.ELSE
    def is_directive_endif(self):
        return self.directive == DirectiveKind.ENDIF

    def is_single_define(self):
        return len(self.local_conds) == 1 and self.local_conds[0].is_define()

    def is_single_undef(self):
        return len(self.local_conds) == 1 and self.local_conds[0].is_undef()

    def negated_conds(self):
        return [ atom.negated()
                 for atom in self.local_conds ]

    def neutralized_conds(self):
        return [ atom.neutralized()
                 for atom in self.local_conds ]

    def neutralized_macro_conds(self):
        return [ atom.neutralized()
                 for atom in self.local_conds
                 if atom.has_macro() ]

@dataclass
class AppleLibcBlock:
    begin_libc: Optional[int]
    end_libc: Optional[int]
    is_empty: bool = True

regex_misc   = re.compile(r'^\s*#\s*([A-Za-z_]\w*)')
regex_ifdef  = re.compile(r'^\s*#\s*ifdef\s+(\w+)\b')
regex_ifndef = re.compile(r'^\s*#\s*ifndef\s+(\w+)\b')
regex_if     = re.compile(r'^\s*#\s*if\b(.*)')
regex_elif_defined     = re.compile(r'^\s*#\s*elif\s+defined\s*\(\s*(\w+)\s*\)')
regex_elif_not_defined = re.compile(r'^\s*#\s*elif\s+!\s*defined\s*\(\s*(\w+)\s*\)')
regex_elif   = re.compile(r'^\s*#\s*elif\b(.*)')
regex_else   = re.compile(r'^\s*#\s*else\b')
regex_endif  = re.compile(r'^\s*#\s*endif\b')

def parse_lines(lines):
    objs: List[LineObj] = []
    if_stack: List[int] = []

    for idx, line in enumerate(lines):
        lo = LineObj(text=line)

        # #ifdef
        if m := regex_ifdef.match(line):
            lo.directive = DirectiveKind.IFDEF
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.DEFINE, macro))
            if_stack.append(idx)

        # #ifndef
        elif m := regex_ifndef.match(line):
            lo.directive = DirectiveKind.IFNDEF
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.UNDEF, macro))
            if_stack.append(idx)

        # #if (complex/composite condition)
        elif m := regex_if.match(line):
            lo.directive = DirectiveKind.IF
            lo.local_conds.append(CondAtom(CondType.COMPLEX, None))
            if_stack.append(idx)

        # #elif defined(X)
        elif m := regex_elif_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]

            # negate the parental conditions
            lo.local_conds.extend(parent.negated_conds())

            # append D:X for "defined(X)"
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.DEFINE, macro))

        # #elif !defined(X)
        elif m := regex_elif_not_defined.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]

            # negate the parental conditions
            lo.local_conds.extend(parent.negated_conds())

            # append U:X for "!defined(X)"
            macro = m.group(1)
            lo.local_conds.append(CondAtom(CondType.UNDEF, macro))

        # #elif (complex)
        elif m := regex_elif.match(line):
            lo.directive = DirectiveKind.ELIF
            if not if_stack:
                raise SyntaxError(f"unmatched #elif at line {idx+1}")
            lo.related_if = if_stack[-1]
            lo.local_conds.append(CondAtom(CondType.COMPLEX, None))

        # #else (negate)
        elif regex_else.match(line):
            lo.directive = DirectiveKind.ELSE
            if not if_stack:
                raise SyntaxError(f"unmatched #else at line {idx+1}")
            lo.related_if = if_stack[-1]

            parent = objs[lo.related_if]
            lo.local_conds.extend(parent.negated_conds())

        # #endif ( NEUTRAL )
        elif regex_endif.match(line):
            lo.directive = DirectiveKind.ENDIF
            if not if_stack:
                raise SyntaxError(f"unmatched #endif at line {idx+1}")

            related = if_stack.pop()
            lo.related_if = related

            parent = objs[related]
            lo.local_conds.extend(parent.neutralized_conds())

        # #define, #undef, #include, #pragma, #error, #line, etc are marked but not parsed.
        elif regex_misc.match(line):
            lo.directive = DirectiveKind.PP_MISC

        objs.append(lo)

    if if_stack:
        raise SyntaxError("unclosed #if block(s)")

    return objs


def propagate_effective_conditions(objs: List[LineObj]):
    """
    Propagate conditions downward.
    effective_conds = conditions required for THIS line to appear.
    local_conds     = conditions introduced BY this line (affecting following lines).
    cond_stack      = accumulated D/U/N conditions from outer scopes.
    """
    cond_stack: List[List[CondAtom]] = [[]]

    for idx, lo in enumerate(objs):

        if lo.is_directive_ifdef() or lo.is_directive_ifndef():
            # effective_conds = outer conditions + neutralized local conds
            lo.effective_conds = cond_stack[-1] + lo.neutralized_conds()

            # push new layer
            cond_stack.append(cond_stack[-1] + lo.local_conds)
            continue

        elif lo.is_directive_if():
            lo.effective_conds = cond_stack[-1].copy()
            cond_stack.append(cond_stack[-1] + lo.local_conds)
            continue

        elif lo.is_directive_elif():
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            # outer conditions (one level above the whole if-chain)
            outer = cond_stack[-2].copy()

            # effective_conds = outer + NEUTRAL(parent + my local macros)
            lo.effective_conds = outer[:]
            seen = set()
            for atom in parent.neutralized_macro_conds() + lo.neutralized_macro_conds():
                if atom.macro not in seen:
                    lo.effective_conds.append(atom)
                    seen.add(atom.macro)

            # create new layer by negated parental local_conds and my local_conds
            cond_stack[-1] = cond_stack[-2] + parent.negated_conds() + lo.local_conds
            continue

        elif lo.is_directive_else():
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            # outer conditions (one level above the whole if-chain)
            outer = cond_stack[-2].copy()

            # effective_conds = outer + NEUTRAL(parent macros)
            lo.effective_conds = outer[:]
            lo.effective_conds.extend(parent.neutralized_macro_conds())

            # create new layer by negated parental local_conds
            cond_stack[-1] = cond_stack[-2] + parent.negated_conds()
            continue

        elif lo.is_directive_endif():
            parent_idx = lo.related_if
            parent = objs[parent_idx]

            # outer conditions (one level above the whole if-chain)
            outer = cond_stack[-2].copy() if len(cond_stack) >= 2 else []

            # effective_conds = outer + NEUTRAL(parent macros)
            lo.effective_conds = outer[:]
            lo.effective_conds.extend(parent.neutralized_macro_conds())

            cond_stack.pop()
            continue

        else:
            lo.effective_conds = cond_stack[-1].copy()

def eval_effective_conds(effective_conds, defined_set, undefined_set):

    for atom in effective_conds:
        macro = atom.macro

        if atom.is_complex():
            continue

        if atom.is_neutral():
            if macro in defined_set or macro in undefined_set:
                return False
            continue

        if atom.is_define():
            if macro in defined_set:
                continue
            if macro in undefined_set:
                return False
            continue

        if atom.is_undef():
            if macro in defined_set:
                return False
            if macro in undefined_set:
                continue
            continue

    return True

def postprocess_repair_structure(objs: List[LineObj], removed: set):
    """
    Repair broken #if/#elif/#else/#endif structures after filtering.
    If the parent #if was removed but an #elif or #else remains,
    convert them into standalone #ifdef/#ifndef blocks.
    """

    new_lines = []
    for idx, lo in enumerate(objs):

        if idx in removed:
            continue

        # If this is an ELIF whose parent was removed
        if lo.is_directive_elif():
            parent = lo.related_if
            if parent in removed:
                # Convert to standalone #ifdef / #ifndef
                if lo.is_single_define():
                    new_lines.append(f"#ifdef {lo.local_conds[0].macro}")
                    continue
                elif lo.is_single_undef():
                    new_lines.append(f"#ifndef {lo.local_conds[0].macro}")
                    continue
                else:
                    new_lines.append("#if /* complex */")
                    continue

        # If this is an ELSE whose parent was removed
        if lo.is_directive_else():
            parent = lo.related_if
            if parent in removed:
                new_lines.append("#else")
                continue

        # ENDIF always stays
        new_lines.append(lo.text)

    return new_lines

def filter_output_lines(objs, defined_set, undefined_set, apple_libc_blocks):
    idx_to_remove = set()

    # First pass: evaluate conditions
    for idx, lo in enumerate(objs):
        if not eval_effective_conds(lo.effective_conds, defined_set, undefined_set):
            idx_to_remove.add(idx)
            continue

        for blk in apple_libc_blocks:
            if blk.begin_libc < idx < blk.end_libc:
                blk.is_empty = False

    # Remove empty Apple Libc blocks
    for blk in apple_libc_blocks:
        if blk.is_empty:
            idx_to_remove.add(blk.begin_libc)
            idx_to_remove.add(blk.end_libc)

    # --- NEW: repair structure ---
    repaired = postprocess_repair_structure(objs, idx_to_remove)

    return repaired

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

    # if no related_if for the last endif, the source is broken
    if endif_obj.related_if is None:
        return None

    if_idx = endif_obj.related_if
    if_obj = objs[if_idx]

    if debug:
        print(f"detect_header_guard: condition starts at {if_idx}")

    # confirm related pp is '#ifndef X' or '#if !defined(X)'
    if if_obj.directive not in (DirectiveKind.IFNDEF, DirectiveKind.IF):
        return None

    if debug:
        print(f"detect_header_guard: {if_idx} is '#ifndef' or '#if defined'")

    # extract a macro used for header guard
    #   confirm local_conds has one macro only
    if len(if_obj.local_conds) != 1:
        return None

    cond_atom = if_obj.local_conds[0]
    macro = cond_atom.macro

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
                objs[idx].local_conds = []

    propagate_effective_conditions(objs)

    if args.debug:
        for lo in objs:
            local = ",".join(f"{a.kind.value}:{a.macro or '*'}" for a in lo.local_conds)
            eff   = ",".join(f"{a.kind.value}:{a.macro or '*'}" for a in lo.effective_conds)
            print(f"[DEBUG] {local:20s} {eff:20s} {lo.text}")

    source_processed = filter_output_lines(objs, args.D, args.U, apple_libc_blocks)

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

