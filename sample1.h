#ifdef A
/* A is defined */
#else
/* A is undefined */
#endif

#ifdef A
#ifdef B
/* A & B are defined */
#else
/* A is defined but B is undefined */
#endif
#endif
