#include "curand.h"
#include "cosplay_impl/HIPSynchronisedStream.hpp"
#include "cosplay_impl/cu_to_roc.hpp"
#include <cuda.h>

ENUM_CU_TO_ROC_PARTIAL(rocrand_rng_type, curandRngType_t,
     (CURAND_RNG_TEST, ROCRAND_RNG_PSEUDO_DEFAULT),
     (CURAND_RNG_PSEUDO_DEFAULT, ROCRAND_RNG_PSEUDO_DEFAULT),
     (CURAND_RNG_PSEUDO_XORWOW, ROCRAND_RNG_PSEUDO_XORWOW),
     (CURAND_RNG_PSEUDO_MRG32K3A, ROCRAND_RNG_PSEUDO_MRG32K3A),
     (CURAND_RNG_PSEUDO_MTGP32, ROCRAND_RNG_PSEUDO_MTGP32),

     // TODO: Panik.
//     (CURAND_RNG_PSEUDO_MT19937, ROCRAND_RNG_PSEUDO_DEFAULT),

     (CURAND_RNG_PSEUDO_PHILOX4_32_10, ROCRAND_RNG_PSEUDO_PHILOX4_32_10),
     (CURAND_RNG_QUASI_DEFAULT, ROCRAND_RNG_QUASI_DEFAULT),
     (CURAND_RNG_QUASI_SOBOL32, ROCRAND_RNG_QUASI_SOBOL32),
     (CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32),
     (CURAND_RNG_QUASI_SOBOL64, ROCRAND_RNG_QUASI_SOBOL64),
     (CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64)
)

struct curandGenerator {
    std::shared_ptr<CudaRocmWrapper::HIPSynchronisedStream> str =
        CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(nullptr);
    rocrand_generator generator;

    operator rocrand_generator() {
        return generator;
    }
};

curandStatus_t curandCreateGenerator(curandGenerator_t* gen, curandRngType_t type) {
    *gen = new curandGenerator;
    return mapReturnCode(rocrand_create_generator((&((*gen)->generator)), cuToRoc(type)));
}
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* gen, curandRngType_t type) {
    // Conceivably...? :D
    return curandCreateGenerator(gen, type);
}
curandStatus_t curandCreatePoissonDistribution(double lambda, curandDiscreteDistribution_t* distrib) {
    return mapReturnCode(rocrand_create_poisson_distribution(lambda, distrib));
}

curandStatus_t curandDestroyDistribution(curandDiscreteDistribution_t distrib) {
    return mapReturnCode(rocrand_destroy_discrete_distribution(distrib));
}
curandStatus_t curandDestroyGenerator(curandGenerator_t gen) {
    return mapReturnCode(rocrand_destroy_generator(*gen));
}

curandStatus_t curandSetStream(curandGenerator_t gen, cudaStream_t str) {
    gen->str = CudaRocmWrapper::HIPSynchronisedStream::getForCudaStream(str);
    return mapReturnCode(rocrand_set_stream(*gen, *gen->str));
}



curandStatus_t curandGenerate(curandGenerator_t gen, unsigned int* out, size_t num) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate(*gen, out, num));
}
curandStatus_t curandGenerateLongLong(curandGenerator_t gen, unsigned long long* out, size_t num) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_long_long(*gen, out, num));
}

curandStatus_t curandGenerateNormal(curandGenerator_t gen, float* out, size_t n, float mean, float std) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_normal(*gen, out, n, mean, std));
}
curandStatus_t curandGenerateNormalDouble(curandGenerator_t gen, double* out, size_t n, double mean, double std) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_normal_double(*gen, out, n, mean, std));
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t gen, float* out, size_t n, float mean, float std) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_log_normal(*gen, out, n, mean, std));
}
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t gen, double* out, size_t n, double mean, double std) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_log_normal_double(*gen, out, n, mean, std));
}

curandStatus_t curandGeneratePoisson(curandGenerator_t gen, unsigned int* out, size_t n, double lambda) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_poisson(*gen, out, n, lambda));
}
curandStatus_t curandGenerateSeeds(curandGenerator_t gen) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_initialize_generator(*gen));
}

curandStatus_t curandGenerateUniform(curandGenerator_t gen, float* out, size_t num) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_uniform(*gen, out, num));
}
curandStatus_t curandGenerateUniformDouble(curandGenerator_t gen, double* out, size_t num) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_generate_uniform_double(*gen, out, num));
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t gen, unsigned long long offs) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_set_offset(*gen, offs));
}
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t, curandOrdering_t) {
    // It's fairly reasonable for this to be a no-op. I do note that AMD document that this works, but it
    // seems not to exist. Fun.
    //
    // The default ordering is defined by curand to be "best" (in terms of quality of random numbers).
    // Other orderings are trading randomness quality for performance (or matching the exact byte
    // sequence produced by previous versions of curand). Neither of these seem like terribly useful
    // things for us, because:
    // - Via the magic of competence, we can both have our cake and eat it (w.r.t performance vs
    //   randomness quality)
    // - Since it is undocumented, it is unrealistic that we can ever implement the "compatibility with
    //   ancient curand" mode without reverse engineering it. I'd rather not be yelled at by Legal.
    //
    // Anyhow: the practical upshot is that a program which uses this function will likely get better-quality
    // random numbers than it is expecting, and not exactly the same ones as it got with nvidia's library.
    // That should not hurt a sanely-written program...
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t gen, unsigned long long seed) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_set_seed(*gen, seed));
}
curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t gen, unsigned int dims) {
    CudaRocmWrapper::HIPSynchronisedStream::EnqueueHipItems q{*gen->str};
    return mapReturnCode(rocrand_set_quasi_random_generator_dimensions(*gen, dims));
}


// Documented unsupported by the AMD libraries (and unlike a few others, we couldn't instantly figure
// out how to cheat):
// https://docs.amd.com/projects/HIPIFY/en/docs-5.2.0/tables/CURAND_API_supported_by_HIP.html#device-api-functions
//
//curandStatus_t curandGetScrambleConstants32(unsigned int** constants);
//curandStatus_t curandGetScrambleConstants64(unsigned long long** constants);
//curandStatus_t curandGetDirectionVectors32(curandDirectionVectors32_t* vecs, curandDirectionVectorSet_t set);
//curandStatus_t curandGetDirectionVectors64(curandDirectionVectors64_t* vecs, curandDirectionVectorSet_t set);
