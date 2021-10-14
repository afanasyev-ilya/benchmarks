#include <string>

#ifdef __USE_INTEL__
#include <immintrin.h>
#endif

#ifdef __USE_ARM_SVE__
#include <arm_sve.h>
#endif

template<typename IT, typename DT>
void init(DT *a, IT *b, DT *c, size_t large_size, size_t small_size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            a[i] = rand_r(&myseed);
            b[i] = (int)rand_r(&myseed) % small_size;
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < small_size; i++)
        {
            c[i] = rand_r(&myseed);
        }
    }
}

template<typename DT>
void re_init(DT *a, size_t small_size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < small_size; i++)
        {
            a[i] = rand_r(&myseed);
        }
    }
}

template<typename IT, typename DT>
void gather(DT *large, IT *indexes, const DT * __restrict__ small, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
    {
        large[i] = small[indexes[i]];
    }
}

template<typename IT, typename DT>
void gather_optimized(DT *large, IT *indexes, const DT * __restrict__ small, size_t size)
{
    #ifdef __USE_INTEL__
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i += 8)
    {
        __m512d val;
        __m256i idx;
        idx = _mm256_load_si256((__m256i *)(&indexes[i]));
        val = _mm512_i32gather_pd(idx, small, 8);
        _mm512_store_pd(&large[i], val);
    }
    #endif

    #ifdef __USE_ARM_SVE__
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i += 8)
    {
        svbool_t pg = svwhilelt_b64(i, size);
        svuint64_t vec_idx = svld1sw_u64(pg, indexes + i);
        svfloat64_t vec_res = svld1_gather_index(pg, small, vec_idx);
        svst1(svptrue_b64(), &(large[i]), vec_res);
    }
    #endif
}

template<typename IT, typename DT>
void kernel(int mode, DT *large, IT *indexes, const DT * __restrict__ small, size_t size)
{
    if(mode == 0)
        gather(large, indexes, small, size);
    else if(mode == 1)
    {
        gather_optimized(large, indexes, small, size);
    }
}
