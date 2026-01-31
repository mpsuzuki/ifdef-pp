# ifdef-pp

## ifdef preprocessor

This Python script apply partial proprocess for
the conditional directives `#ifdef`,
`#if defined()`, `#ifndef`, `#if !defined()`,
`#else`, `#elif defined()`, `#elif !defined()`,
and `#endif`.

Taking a sample like:
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
modifies this sample like:
```
#ifdef B
	case1
#else
	case2
#endif
	case3

```

If B is set to undefined, partial preprocess
modifies this sample like:

```
#ifdef A
	case2
	case3
#endif
```

## Why?

Some of Apple OSS sources, like Libc or xnu have so
many conditionals which are removed in macOS SDK,
like `#ifdef KERNEL` or `#ifdef __LIBC__`.

To simplify the headers by removing the codes for
internal usage, this script preprocess the headers
partially.
