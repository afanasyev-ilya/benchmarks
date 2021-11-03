#include <string>

#ifdef __USE_INTEL__
#include <immintrin.h>
#endif

#ifdef __USE_ARM_SVE__
#include <arm_sve.h>
#endif

#include <random>
#include <algorithm>

template<typename IT, typename DT>
void uniform_init(DT *result, IT *indexes, DT **small_data, size_t large_size, size_t small_size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            indexes[i] = (int)rand_r(&myseed) % small_size;
        }
    }
}

template<typename IT, typename DT>
void init(DT *result, IT *indexes, DT **small_data, size_t large_size, size_t small_size)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            result[i] = rand_r(&myseed) / RAND_MAX;
            indexes[i] = (int)rand_r(&myseed) % small_size;
        }
        DT * small_current = small_data[tid];

        for (size_t i = 0; i < small_size; i++)
        {
            small_current[i] = rand_r(&myseed);
        }
    }

    uniform_init(result, indexes, small_data, large_size, small_size);
}

template<typename DT>
void re_init(DT **small_data, size_t small_size)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        unsigned int myseed = omp_get_thread_num();

        DT * small_current = small_data[tid];
        for (size_t i = 0; i < small_size; i++)
        {
            small_current[i] = rand_r(&myseed);
        }
    }
}

template<typename IT, typename DT>
void gather(DT *large, IT *indexes, DT **small, size_t size)
{
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        const DT * __restrict__ small_current = small[tid];
        #pragma omp for schedule(static)
        for(size_t i = 0; i < size; i++)
        {
            large[i] = small_current[indexes[i]];
        }
    }
}


template<typename IT, typename DT>
void kernel(DT *large, IT *indexes, DT ** small, size_t size)
{
    gather(large, indexes, small, size);
}
