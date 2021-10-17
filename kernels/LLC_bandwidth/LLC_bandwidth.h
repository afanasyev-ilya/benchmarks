#pragma once

template<typename AT>
void init(AT *a, AT *b, AT *c, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < size; i++)
        {
            a[i] = rand_r(&myseed);
            b[i] = rand_r(&myseed);
            c[i] = rand_r(&myseed);
        }
    }
}

template<typename AT>
inline void tuned_add(AT *a, AT *b, AT *c, size_t size)
{
    size_t j;
    #pragma omp parallel
    {
        for(int i = 0; i < OUTER_ITERS; i++)
        {
            #pragma omp for
            for (j=0; j<size; j++)
                a[j] = b[j] + c[j];
        }
    };

}

template<typename AT>
void kernel(AT *a, AT *b, AT *c, size_t size)
{
    tuned_add(a, b, c, size);
}
