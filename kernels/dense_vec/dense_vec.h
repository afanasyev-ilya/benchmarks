#pragma once

template<typename AT>
void init(AT *a, AT *b, AT *c, AT *d, AT*e, size_t size)
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
            d[i] = rand_r(&myseed);
            e[i] = rand_r(&myseed);
        }
    }
}

template<typename AT>
void re_init(AT *a, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < size; i++)
        {
            a[i] = 0;
        }
    }
}

template<typename AT>
inline void tuned_copy(AT *a, AT *c, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = c[j];
}

template<typename AT>
inline void tuned_scale(AT scalar, AT *a, AT *b, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = scalar*b[j];
}

template<typename AT>
inline void tuned_add(AT *a, AT *b, AT *c, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = b[j]+c[j];
}

template<typename AT>
inline void tuned_triad(AT scalar, AT *a, AT *b, AT *c, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = b[j]+scalar*c[j];
}

template<typename AT>
inline void tuned_four_vect(AT *a, AT *b, AT *c, AT *d, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = b[j] + c[j] + d[j];
}

template<typename AT>
inline void tuned_five_vect(AT *a, AT *b, AT *c, AT *d, AT *e, size_t size)
{
    size_t j;
    #pragma omp parallel for
    for (j=0; j<size; j++)
        a[j] = b[j] + c[j] + d[j] + e[j];
}

template<typename AT>
void kernel(int mode, AT *a, AT *b, AT *c, AT *d, AT *e, size_t size)
{
    if(mode == 0)
    {
        tuned_copy(a, b, size);
    }
    else if(mode == 1)
    {
        tuned_scale(3.0, a, b, size);
    }
    else if(mode == 2)
    {
        tuned_add(a, b, c, size);
    }
    else if(mode == 3)
    {
        tuned_triad(3.0, a, b, c, size);
    }
    else if(mode == 4)
    {
        tuned_four_vect(a, b, c, d, size);
    }
    else if(mode == 5)
    {
        tuned_four_vect(a, b, c, d, size);
    }
}
