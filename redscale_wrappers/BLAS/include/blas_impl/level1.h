/**
 * amax()
 */
#define amax_ARGS n, x, incx, result
BLAS_API(Is, is, amax, Amax, int n, const float *x, int incx, int *result)
BLAS_API(Id, id, amax, Amax, int n, const double *x, int incx, int *result)
BLAS_API(Ic, ic, amax, Amax, int n, const cuComplex *x, int incx, int *result)
BLAS_API(Iz, iz, amax, Amax, int n, const cuDoubleComplex *x, int incx, int *result)
#undef amax_ARGS

/**
 * amin()
 */
#define amin_ARGS n, x, incx, result
BLAS_API(Is, is, amin, Amin, int n, const float *x, int incx, int *result)
BLAS_API(Id, id, amin, Amin, int n, const double *x, int incx, int *result)
BLAS_API(Ic, ic, amin, Amin, int n, const cuComplex *x, int incx, int *result)
BLAS_API(Iz, iz, amin, Amin, int n, const cuDoubleComplex *x, int incx, int *result)
#undef amin_ARGS

/**
 * asum()
 */
#define asum_ARGS n, x, incx, result
BLAS_API(S, s, asum, Asum, int n, const float* x, int incx, float* result)
BLAS_API(D, d, asum, Asum, int n, const double* x, int incx, double* result)
BLAS_API(Sc, sc, asum, Asum, int n, const cuComplex* x, int incx, float* result)
BLAS_API(Dz, dz, asum, Asum, int n, const cuDoubleComplex* x, int incx, double* result)
#undef asum_ARGS

/**
 * axpy()
 */
#define axpy_ARGS n, alpha, x, incx, y, incy
/* BLAS_API(Haxpy, int n, const __half *alpha, const __half *x, int incx, __half *y,  int incy) */
BLAS_API(S, s, axpy, Axpy, int n, const float *alpha, const float *x, int incx, float *y,  int incy)
BLAS_API(D, d, axpy, Axpy, int n, const double *alpha, const double *x, int incx, double *y,  int incy)
BLAS_API(C, c, axpy, Axpy, int n, const cuComplex *alpha, const cuComplex *x, int incx, cuComplex *y,  int incy)
BLAS_API(Z, z, axpy, Axpy, int n, const cuDoubleComplex *alpha, const cuDoubleComplex *x, int incx, cuDoubleComplex *y, int incy)
#undef axpy_ARGS

/**
 * copy()
 */
#define copy_ARGS n, x, incx, y, incy
BLAS_API(S, s, copy, Copy, int n, const float *x, int incx, float* y, int incy)
BLAS_API(D, d, copy, Copy, int n, const double *x, int incx, double* y, int incy)
BLAS_API(C, c, copy, Copy, int n, const cuComplex *x, int incx, cuComplex* y, int incy)
BLAS_API(Z, z, copy, Copy, int n, const cuDoubleComplex *x, int incx, cuDoubleComplex* y, int incy)
#undef copy_ARGS

/**
 * dot()
 */
#define dot_ARGS n, x, incx, y, incy, result
#define dotu_ARGS dot_ARGS
#define dotc_ARGS dot_ARGS
BLAS_API(S, s, dot, Dot, int n, const float *x, int incx, const float *y, int incy, float *result)
BLAS_API(D, d, dot, Dot, int n, const double *x, int incx, const double *y, int incy, double *result)
BLAS_API(C, c, dotu, Dotu, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
BLAS_API(Z, z, dotu, Dotu, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
BLAS_API(C, c, dotc, Dotc, int n, const cuComplex *x, int incx, const cuComplex *y, int incy, cuComplex *result)
BLAS_API(Z, z, dotc, Dotc, int n, const cuDoubleComplex *x, int incx, const cuDoubleComplex *y, int incy, cuDoubleComplex *result)
#undef dot_ARGS
#undef dotu_ARGS
#undef dotc_ARGS

/**
 * nrm2()
 */
#define nrm2_ARGS n, x, incx, result
BLAS_API(S, s, nrm2, Nrm2, int n, const float *x, int incx, float *result)
BLAS_API(D, d, nrm2, Nrm2, int n,  const double *x, int incx,  double *result)
BLAS_API(Sc, sc, nrm2, Nrm2, int n, const cuComplex *x, int incx,  float *result)
BLAS_API(Dz, dz, nrm2, Nrm2, int n, const cuDoubleComplex *x, int incx, double *result)
#undef nrm2_ARGS

/**
 * rot()
 */
#define rot_ARGS n, x, incx, y, incy, c, s
BLAS_API(S, s, rot, Rot, int n, float* x, int incx, float* y, int incy, const float* c, const float* s)
BLAS_API(D, d, rot, Rot, int n, double* x, int incx, double* y, int incy, const double* c, const double* s)
BLAS_API(C, c, rot, Rot, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const cuComplex* s)
BLAS_API(Cs, cs, rot, Rot, int n, cuComplex* x, int incx, cuComplex* y, int incy, const float* c, const float* s)
BLAS_API(Z, z, rot, Rot, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const cuDoubleComplex* s)
BLAS_API(Zd, zd, rot, Rot, int n, cuDoubleComplex* x, int incx, cuDoubleComplex* y, int incy, const double* c, const double* s)
#undef rot_ARGS

/**
 * rotg()
 */
#define rotg_ARGS a, b, c, s
BLAS_API(S, s, rotg, Rotg, float* a, float* b, float* c, float* s)
BLAS_API(D, d, rotg, Rotg, double* a, double* b, double* c, double* s)
BLAS_API(C, c, rotg, Rotg, cuComplex* a, cuComplex* b, float* c, cuComplex* s)
BLAS_API(Z, z, rotg, Rotg, cuDoubleComplex* a, cuDoubleComplex* b, double* c, cuDoubleComplex* s)
#undef rotg_ARGS

/**
 * rotm()
 */
#define rotm_ARGS n, x, incx, y, incy, param
BLAS_API(S, s, rotm, Rotm, int n, float* x, int incx, float* y, int incy, const float* param)
BLAS_API(D, d, rotm, Rotm, int n, double* x, int incx, double* y, int incy, const double* param)
#undef rotm_ARGS

/**
 * rotmg()
 */
#define rotmg_ARGS d1, d2, x1, y1, param
BLAS_API(S, s, rotmg, Rotmg, float* d1, float* d2, float* x1, const float* y1, float* param)
BLAS_API(D, d, rotmg, Rotmg, double* d1, double* d2, double* x1, const double* y1, double* param)
#undef rotmg_ARGS

/**
 * scal()
 */
#define scal_ARGS n, alpha, x, incx
BLAS_API(S, s, scal, Scal, int n, const float *alpha, float *x, int incx)
BLAS_API(D, d, scal, Scal, int n, const double *alpha, double *x, int incx)
BLAS_API(C, c, scal, Scal, int n, const cuComplex *alpha, cuComplex *x, int incx)
BLAS_API(Z, z, scal, Scal, int n, const cuDoubleComplex *alpha, cuDoubleComplex *x, int incx)
BLAS_API(Cs, cs, scal, Scal, int n, const float *alpha, cuComplex *x, int incx)
BLAS_API(Zd, zd, scal, Scal, int n, const double *alpha, cuDoubleComplex *x, int incx)
#undef scal_ARGS

/**
 * swap()
 */
#define swap_ARGS n, x, incx, y, incy
BLAS_API(S, s, swap, Swap, int n, float *x, int incx, float* y, int incy)
BLAS_API(D, d, swap, Swap, int n, double *x, int incx, double* y, int incy)
BLAS_API(C, c, swap, Swap, int n, cuComplex *x, int incx, cuComplex* y, int incy)
BLAS_API(Z, z, swap, Swap, int n, cuDoubleComplex *x, int incx, cuDoubleComplex* y, int incy)
#undef swap_ARGS
