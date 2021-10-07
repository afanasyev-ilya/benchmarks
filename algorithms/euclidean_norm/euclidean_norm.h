#include <math.h>
#include <omp.h>


#define MAX_INNER_ITERS 10
const double eps = 0.001;

template <typename AT>
void Init(AT* x, AT *y, size_t size)
{
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();

        #pragma omp for schedule(static)
        for(size_t i = 0; i < size; i++)
        {
            x[i] = rand_r(&seed);
            y[i] = rand_r(&seed);
        }
    }
}

template <typename AT>
AT euclidean_distance(AT* x, AT *y, size_t size)
{
    AT norm = 0;
    #pragma omp parallel for reduction(+: norm)
    for(size_t h = 0; h < size; h++)
    {
        norm += (x[h] - y[h]) * (x[h] - y[h]);
    }
    norm = sqrt(norm);
    return norm;
}

template <typename AT>
AT Kernel(AT* x, AT *y, size_t size)
{
    return euclidean_distance(x, y, size);
}
