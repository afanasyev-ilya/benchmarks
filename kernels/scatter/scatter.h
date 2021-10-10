#include <string>

#ifdef __USE_INTEL__
#include <immintrin.h>
#endif

using std::string;

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
void scatter(DT *large, IT *indexes, DT *small, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
    {
        small[indexes[i]] = large[i];
    }
}

template<typename IT, typename DT>
void kernel(int mode, DT *large, IT *indexes, DT *small, size_t size)
{
    scatter(large, indexes, small, size);
}
