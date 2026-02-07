# ifdef-pp

## ifdef preprocessor

This Python script apply partial proprocess for
the conditional directives `#ifdef`,
`#if defined()`, `#ifndef`, `#if !defined()`,
`#else`, `#elif defined()`, `#elif !defined()`,
and `#endif`.

Taking a [sample](sample1.h) like:
```
#ifdef A
#ifdef B
	case1
#else
	case2
#endif
	case3
#endif
```

If A is set to defined, partial preprocess
`./ifdef-pp.py -DA sample1.h` emits a code
looking like:
```
#ifdef B
	case1
#else
	case2
#endif
	case3

```

If B is set to undefined, partial preprocess
`./ifdef-pp.py -UB sample1.h` emits a code
looking like:
```
#ifdef A
	case2
	case3
#endif
```

## Who needs such tool?

Some of Apple OSS sources, like Libc or xnu have so
many conditionals which are removed in macOS SDK,
like `#ifdef KERNEL` or `#ifdef __LIBC__`.

To simplify the headers by removing the codes for
internal usage, this script preprocess the headers
partially.

## Similar existing tools

(partial-pp)[https://github.com/awishnick/partial-pp]
might have similar purposes with more readable
implementation, but resulted code seems to be too
restricted to process Apple OSS sources.

## Todo

### Handling of macros defined/undefined internally

Currently, ifdef-pp.py does not care about the macros
defined internally, except of the explict header guard
like:
```
#ifndef X
#define X
...
#endif
```
In such header guards, the set of directives (`#ifndef X`,
`#define X`, and last `#endif`) is excluded from
preprossing, however, if `#ifdef X` appears in the middle
of headers, it would not be handled correctly.

The macros which can be defined internally should be
refused or warned if users try to set them externally.

### Ignore quasi-directives in comments or string constants

Currently, ifdef-pp.py cannot detect whether the pp directives
appearing in the middle of the comments or string constants.

### Purpose and Evaluation Model

This tool is *not* an implementation of the C preprocessor,
nor does it attempt to reproduce its exact behavior.
Instead, it evaluates conditional directives using a
*custom three‑valued logic* designed for partial macro
information.

The tool distinguishes three states for each macro:

* defined — explicitly specified by the user via -D
* undefined — explicitly specified by the user via -U
* pending — not specified by the user;
	its truth value is unknown

This third state (pending) does not exist
in the real C preprocessor, but it is essential
for this tool.

#### Key principles
1. Pending is a first‑class state.

If a macro is neither in the defined set nor in
the undefined set, its value must remain pending.

2. The goal is simplification, not full evaluation.

The tool simplifies conditional blocks only when
the result is unambiguously true or false.
If the result depends on a pending macro,
the corresponding `#if` structure must be preserved.

3. Do not apply normal C preprocessor rules.

* Do not assume undefined macros evaluate to 0.
* Do not collapse expressions using C’s short‑circuit
	rules unless the result is fully determined.
* Do not attempt to fully evaluate expressions
	involving pending macros.

4. Three‑valued logic is used for all expressions.
Operators (`&&`, `||`, `!`) propagate pending values
unless the result is forced to true or false.

#### Summary
This tool operates under a custom evaluation model:

* It simplifies what can be simplified based
	on user‑provided macro information.
* It preserves conditional structures when
	the truth value cannot be determined.
* It must not behave like a full C preprocessor.

All reasoning about conditions must follow this model.
