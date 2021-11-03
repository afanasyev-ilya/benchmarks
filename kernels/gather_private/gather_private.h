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
void rmat_init(DT *result, IT *indexes, DT *small_data, size_t large_size, size_t small_size)
{
    unsigned int seed = 0;
    int _a_prob = 57;
    int _b_prob = 19;
    int _c_prob = 19;
    int _d_prob = 5;
    int n = (int)log2(small_size);
    #pragma omp parallel private(seed)
    {
        seed = int(time(NULL)) * omp_get_thread_num();

        #pragma omp for schedule(guided, 1024)
        for (long long cur_edge = 0; cur_edge < large_size; cur_edge++)
        {
            int x_middle = small_size / 2, y_middle = small_size / 2;
            for (long long i = 1; i < n; i++)
            {
                int a_beg = 0, a_end = _a_prob;
                int b_beg = _a_prob, b_end = b_beg + _b_prob;
                int c_beg = _a_prob + _b_prob, c_end = c_beg + _c_prob;
                int d_beg = _a_prob + _b_prob + _c_prob, d_end = d_beg + _d_prob;

                int step = (int)pow(2, n - (i + 1));

                int probability = rand_r(&seed) % 100;
                if (a_beg <= probability && probability < a_end)
                {
                    x_middle -= step, y_middle -= step;
                }
                else if (b_beg <= probability && probability < b_end)
                {
                    x_middle -= step, y_middle += step;
                }
                else if (c_beg <= probability && probability < c_end)
                {
                    x_middle += step, y_middle -= step;
                }
                else if (d_beg <= probability && probability < d_end)
                {
                    x_middle += step, y_middle += step;
                }
            }
            if (rand_r(&seed) % 2 == 0)
                x_middle--;
            if (rand_r(&seed) % 2 == 0)
                y_middle--;

            int from = x_middle;
            int to = y_middle;

            indexes[cur_edge] = to;
        }
    }
}

template<typename IT, typename DT>
void uniform_init(DT *result, IT *indexes, DT *small_data, size_t large_size, size_t small_size)
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
void init(RAND_DATA_TYPE rand_data_type, DT *result, IT *indexes, DT *small_data, size_t large_size, size_t small_size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < large_size; i++)
        {
            result[i] = rand_r(&myseed);
            indexes[i] = (int)rand_r(&myseed) % small_size;
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < small_size; i++)
        {
            small_data[i] = rand_r(&myseed);
        }
    }

    if(rand_data_type == UNIFORM)
        uniform_init(result, indexes, small_data, large_size, small_size);
    else if(rand_data_type == RMAT)
        rmat_init(result, indexes, small_data, large_size, small_size);
    else if(rand_data_type == RMAT_SHUFFLED)
    {
        rmat_init(result, indexes, small_data, large_size, small_size);
        random_shuffle(indexes, indexes + large_size);
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
    #elif defined(__USE_ARM_SVE__)
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i += 8)
    {
        svbool_t pg = svwhilelt_b64(i, size);
        svuint64_t vec_idx = svld1sw_u64(pg, indexes + i);
        svfloat64_t vec_res = svld1_gather_index(pg, small, vec_idx);
        svst1(svptrue_b64(), &(large[i]), vec_res);
    }
    #else
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
    {
        large[i] = small[indexes[i]];
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
