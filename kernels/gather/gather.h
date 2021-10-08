#include <string>

using std::string;

template<typename IT, typename DT>
void Init(DT *a, IT *b, DT *c, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            a[i] = rand_r(&myseed);
            b[i] = (int)rand_r(&myseed) % RADIUS_IN_ELEMENTS;
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < RADIUS_IN_ELEMENTS; i++)
        {
            c[i] = rand_r(&myseed);
        }
    }
}

template<typename IT, typename DT>
void gather(DT *large, IT *indexes, DT *small, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t i = 0; i < size; i++)
    {
        large[i] = small[indexes[i]];
    }
}

template<typename IT, typename DT>
void kernel(int core_type, DT *large, IT *indexes, DT *small, size_t size)
{
    gather(large, indexes, small, size);
}
