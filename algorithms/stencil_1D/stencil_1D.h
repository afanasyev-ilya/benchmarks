#ifdef __USE_INTEL__
#include <immintrin.h>
#elif __USE_KUNPENG_920__
#include <arm_neon.h>
#endif

template<typename AT>
void init(AT *a, AT *b, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            a[i] = rand_r(&myseed);
            b[i] = rand_r(&myseed);
        }
    }
}

template<typename AT>
void kernel_basic(AT * __restrict__ a, const AT * __restrict__ b, const size_t size, const int radius)
{
    #pragma ivdep
    #pragma simd
    #pragma vector
    #pragma omp parallel for schedule(static)
    for(size_t i = radius; i < size - radius; i++)
    {
        AT local_sum = 0;

        for (int j = -radius; j <= radius; j++)
        {
            local_sum += b[i + j];
        }

        a[i] = local_sum;
    }
}

#ifdef __USE_AVX_512__
#define ADD(OFFSET) \
data = _mm512_load_ps(&b[i - 5]);\
vec_sum = _mm512_add_ps(vec_sum, data);
#endif


template<typename AT>
void kernel_optimized(AT * __restrict__ a, const AT * __restrict__ b, const size_t size, const int radius)
{
    #pragma omp parallel
    {
        __m512 data;
        __m512 vec_sum;
        __m512 null = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        #pragma omp for schedule(static)
        for(size_t i = radius; i < size - radius; i += 16)
        {
            vec_sum = null;
            ADD(-7)
            ADD(-6)
            ADD(-5)
            ADD(-4)
            ADD(-3)
            ADD(-2)
            ADD(-1)
            ADD(-0)
            ADD(1)
            ADD(2)
            ADD(3)
            ADD(4)
            ADD(5)
            ADD(6)
            ADD(7)
            _mm512_storeu_ps(&a[i], vec_sum);
        }
    }
}

template<typename AT>
void kernel(int mode, AT * __restrict__ a, const AT * __restrict__ b, const size_t size, const int radius)
{
    if(mode == 0)
        kernel_basic(a, b, size, radius);
    else if(mode == 1)
        kernel_optimized(a, b, size, radius);
}
