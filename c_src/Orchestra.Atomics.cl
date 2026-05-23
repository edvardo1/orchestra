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

#define ATOMIC_INT_PTR volatile __global atomic_int *

inline int atomic_add_int(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_sub_int(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_exchange_int(ATOMIC_INT_PTR ptr, int val) {
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_min_int(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_max_int(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_and(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_and_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_or(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_or_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline int atomic_xor(ATOMIC_INT_PTR ptr, int val) {
  return atomic_fetch_xor_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

// ========= 32-bit Floats Atomics (only supported by extension)=========

#if defined(cl_ext_float_atomics)

// Enable float atomics extension
#pragma OPENCL EXTENSION cl_ext_float_atomics : enable

#define ATOMIC_FLOAT_PTR volatile __global atomic_float *

inline float atomic_add_float(ATOMIC_FLOAT_PTR ptr, float val) {
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline float atomic_sub_float(ATOMIC_FLOAT_PTR ptr, float val) {
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline float atomic_exchange_float(ATOMIC_FLOAT_PTR ptr, float val) {
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline float atomic_min_float(ATOMIC_FLOAT_PTR ptr, float val) {
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline float atomic_max_float(ATOMIC_FLOAT_PTR ptr, float val) {
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

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

#define ATOMIC_DOUBLE_PTR volatile __global atomic_double *

inline double atomic_add_double(ATOMIC_DOUBLE_PTR ptr, double val) {
  return atomic_fetch_add_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline double atomic_sub_double(ATOMIC_DOUBLE_PTR ptr, double val) {
  return atomic_fetch_sub_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline double atomic_exchange_double(ATOMIC_DOUBLE_PTR ptr, double val) {
  return atomic_exchange_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline double atomic_min_double(ATOMIC_DOUBLE_PTR ptr, double val) {
  return atomic_fetch_min_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

inline double atomic_max_double(ATOMIC_DOUBLE_PTR ptr, double val) {
  return atomic_fetch_max_explicit(ptr, val, MEMORY_ORDER, ATOMIC_SCOPE);
}

#endif
