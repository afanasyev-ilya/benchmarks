#ifdef __USE_INTEL__
#include <mkl.h>
#endif

template <typename AT>
void init(AT* A, AT *B, AT*C, size_t size)
{
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();

        #pragma omp for schedule(static)
        for(size_t i = 0; i < size*size; i++)
        {
            A[i] = rand_r(&seed);
            B[i] = rand_r(&seed);
            C[i] = 0;
        }
    }
}

template <typename AT>
void re_init(AT* A, AT *B, AT*C, size_t size)
{
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();

        #pragma omp for schedule(static)
        for(size_t i = 0; i < size*size; i++)
        {
            C[i] = 0;
        }
    }
}

template <typename AT>
void kernel(AT* A, AT *B, AT*C, size_t size)
{
    AT alpha = 1.0, beta = 0.0;
    #ifdef __USE_INTEL__
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                size, size, size, alpha, A, size, B, size, beta, C, size);
    #endif
}
