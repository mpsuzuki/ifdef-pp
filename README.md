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
