// Set memory scope based on the target device (the compiler will set these for
// us)
#if defined(TARGET_CPU)

// CPU: operations affect the whole device
#define ATOMIC_SCOPE memory_scope_device

#else

// GPU: operations affect only the work-group
#define ATOMIC_SCOPE memory_scope_work_group

#endif

#define MEMORY_ORDER memory_order_relaxed

// ========= 32-bit Integers Atomics (supported natively)=========

#define ATOMIC_INT_PTR atomic_int *

#define DECLARE_ATOMIC_INT_FUNCS(ADDR_SPACE) \
\
inline __attribute__((overloadable)) void init_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  atomic_init(ptr, val); \
} \
\
inline __attribute__((overloadable)) int load_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr) { \
  return atomic_load_explicit(ptr, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
// This one is here just to demonstrate how stupid libSPIRV is. It thinks, because of the function 'atomic_' prefix,
// that this function is a built-in OpenCL C atomic function. This makes the ENTIRE Erlang BEAM VM to crash :D
// This stupid idiot compiler took me 2hrs to debug. - Henrique
inline __attribute__((overloadable)) int atomic_load_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr) { \
  return atomic_load_explicit(ptr, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int add_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int sub_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int exchange_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int min_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int max_atomic_int(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, memory_scope_device); \
} \
\
inline __attribute__((overloadable)) int and_atomic(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_and_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int or_atomic(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_or_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) int xor_atomic(volatile ADDR_SPACE ATOMIC_INT_PTR ptr, int val) { \
  return atomic_fetch_xor_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
}

// Declare atomic functions for both global and local address spaces

DECLARE_ATOMIC_INT_FUNCS(__global)
DECLARE_ATOMIC_INT_FUNCS(__local)

// ========= 32-bit Floats Atomics (only supported by extension)=========

#if defined(cl_ext_float_atomics)

// Enable float atomics extension
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable

#define ATOMIC_FLOAT_PTR atomic_float *

#define DECLARE_ATOMIC_FLOAT_FUNCS(ADDR_SPACE) \
\
inline __attribute__((overloadable)) void init_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  atomic_init(ptr, val); \
} \
\
inline __attribute__((overloadable)) float load_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr) { \
  return atomic_load_explicit(ptr, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) float add_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) float sub_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) float exchange_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) float min_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) float max_atomic_float(volatile ADDR_SPACE ATOMIC_FLOAT_PTR ptr, float val) { \
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
}

DECLARE_ATOMIC_FLOAT_FUNCS(__global)
DECLARE_ATOMIC_FLOAT_FUNCS(__local)

#endif

/*
- From OpenCL specification:

The atomic_double type is only supported if double precision is supported and
the cl_khr_int64_base_atomics and cl_khr_int64_extended_atomics extensions are
supported and have been enabled. If this is the case then an OpenCL C 3.0
compiler must also define the __opencl_c_fp64 feature.
*/

#if defined(__opencl_c_fp64) && defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable

#define ATOMIC_DOUBLE_PTR atomic_double *

#define DECLARE_ATOMIC_DOUBLE_FUNCS(ADDR_SPACE) \
\
inline __attribute__((overloadable)) void init_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  atomic_init(ptr, val); \
} \
\
inline __attribute__((overloadable)) double load_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr) { \
  return atomic_load_explicit(ptr, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) double add_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) double sub_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) double exchange_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) double min_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
} \
\
inline __attribute__((overloadable)) double max_atomic_double(volatile ADDR_SPACE ATOMIC_DOUBLE_PTR ptr, double val) { \
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE); \
}

DECLARE_ATOMIC_DOUBLE_FUNCS(__global)
DECLARE_ATOMIC_DOUBLE_FUNCS(__local)

#endif
