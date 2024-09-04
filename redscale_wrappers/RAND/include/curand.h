#ifndef GPURAND_AUXILLARY_H
#define GPURAND_AUXILLARY_H

// Alien's helper_cuda.h looks for this.
#define CURAND_H_

#include "cuda.h"
#ifdef __cplusplus
#include <cstddef>
#else /* __cplusplus */
#include <stddef.h>
#endif /* __cplusplus */
#include "rand_types.h"
#include <library_types.h>
#include "curand/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct curandGenerator curandGenerator;
typedef curandGenerator* curandGenerator_t;

/* Creation/management */

GPURAND_EXPORT curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t type);
GPURAND_EXPORT curandStatus_t curandCreateGeneratorHost(curandGenerator_t* gen, curandRngType_t type);
GPURAND_EXPORT curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* distrib);

GPURAND_EXPORT curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t distrib);
GPURAND_EXPORT curandStatus_t curandDestroyGenerator(curandGenerator_t gen);

GPURAND_EXPORT curandStatus_t curandSetStream(curandGenerator_t gen, cudaStream_t str);

/* Versioney fluff */
GPURAND_EXPORT curandStatus_t curandGetVersion(int* v);
GPURAND_EXPORT curandStatus_t curandGetProperty(libraryPropertyType type, int* v);


/* Launch kernels */

GPURAND_EXPORT curandStatus_t curandGenerate(curandGenerator_t gen, unsigned int* out, size_t num);
GPURAND_EXPORT curandStatus_t curandGenerateLongLong(curandGenerator_t gen, unsigned long long* out, size_t num);

GPURAND_EXPORT curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n, float mean, float std);
GPURAND_EXPORT curandStatus_t curandGenerateNormalDouble(curandGenerator_t gen, double* out, size_t n, double mean, double std);

GPURAND_EXPORT curandStatus_t curandGenerateLogNormal(curandGenerator_t gen, float* out, size_t n, float mean, float std);
GPURAND_EXPORT curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t gen, double* out, size_t n, double mean, double std);

GPURAND_EXPORT curandStatus_t curandGeneratePoisson(curandGenerator_t gen, unsigned int* out, size_t n, double lambda);
GPURAND_EXPORT curandStatus_t curandGenerateSeeds(curandGenerator_t gen);

GPURAND_EXPORT curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t num);
GPURAND_EXPORT curandStatus_t curandGenerateUniformDouble(curandGenerator_t gen, double* out, size_t num);


GPURAND_EXPORT curandStatus_t curandGetScrambleConstants32(unsigned int** constants);
GPURAND_EXPORT curandStatus_t curandGetScrambleConstants64(unsigned long long** constants);

GPURAND_EXPORT curandStatus_t curandSetGeneratorOffset(curandGenerator_t gen, unsigned long long offs);
GPURAND_EXPORT curandStatus_t curandSetGeneratorOrdering(curandGenerator_t gen, curandOrdering_t order);

GPURAND_EXPORT curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t gen, unsigned long long seed);
GPURAND_EXPORT curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t gen, unsigned int dims);


/*
GPURAND_EXPORT curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vecs, curandDirectionVectorSet_t set);
GPURAND_EXPORT curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vecs, curandDirectionVectorSet_t set);
*/

#ifdef __cplusplus
}
#endif

#endif /* GPURAND_AUXILLARY_H */
