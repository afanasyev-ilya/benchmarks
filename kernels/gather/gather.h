#include <string>

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
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
    {
        large[i] = small[indexes[i]];
    }
}

template<typename IT, typename DT>
void kernel(int mode, DT *large, IT *indexes, const DT * __restrict__ small, size_t size)
{
    if(mode == 0)
        gather(large, indexes, small, size);
    else if(mode == 1)
        gather_optimized(large, indexes, small, size);
}
