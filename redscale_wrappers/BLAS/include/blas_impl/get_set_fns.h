/** Copy a vector from host to device **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasSetVector(int n, int elem_size, const void *x, int incx, void *y, int incy);


/** Copy a vector from device to host. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetVectorAsync(int n, int elem_size, const void *x, int incx, void *y, int incy, cudaStream_t stream);
GPUBLAS_EXPORT cublasStatus_t
cublasGetVector(int n, int elem_size, const void *x, int incx, void *y, int incy);


/** Copy a matrix from host to device. **/
GPUBLAS_EXPORT cublasStatus_t
cublasSetMatrixAsync(int rows, int cols,
                     int elem_size,
                     const void *a, int lda,
                     void *b, int ldb, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasSetMatrix(int rows, int cols,
                int elem_size,
                const void *a, int lda,
                void *b, int ldb);


/** Copy a matrix from device to host. **/
GPUBLAS_EXPORT cublasStatus_t
cublasGetMatrixAsync(int rows, int cols,
                     int elem_size,
                     const void *a, int lda,
                     void *b, int ldb, cudaStream_t stream);

GPUBLAS_EXPORT cublasStatus_t
cublasGetMatrix(int rows, int cols,
                int elem_size,
                const void *a, int lda,
                void *b, int ldb);
