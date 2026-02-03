#ifdef A
  /* A is defined */
#else
  /* A is undefined */
#endif

#ifdef A
  #ifdef B
    /* A & B are defined */
  #elif defined(C)
    /* A & C are defined but B is undefined */
  #else
    /* A is defined but B & C are undefined */
  #endif
#endif
