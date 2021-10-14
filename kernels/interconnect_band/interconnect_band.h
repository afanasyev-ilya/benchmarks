#pragma once

#define SAME_SOCKET 0
#define DIFF_SOCKET 1


template<typename AT>
void init(int mode, AT *a, AT *b, AT *c, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int myseed = tid;

        if(mode == 0 || mode == 1)
        {
            if(tid == 0)
            {
                for (size_t i = 0; i < size/2; i++) {
                    a[i] = rand_r(&myseed);
                    b[i] = rand_r(&myseed);
                    c[i] = rand_r(&myseed);
                }
            }
            if(tid == max_threads/2)
            {
                for (size_t i = size/2; i < size; i++)
                {
                    a[i] = rand_r(&myseed);
                    b[i] = rand_r(&myseed);
                    c[i] = rand_r(&myseed);
                }
            }
        }
        else if(mode == 2 || mode == 3)
        {
            #pragma omp for
            for (size_t i = 0; i < size; i++)
            {
                a[i] = rand_r(&myseed);
                b[i] = rand_r(&myseed);
                c[i] = rand_r(&myseed);
            }
        }
    }
}

template<typename AT>
inline void triada_local_seq_accesses(AT scalar, AT *a, AT *b, AT *c, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads/2)
    {
        #pragma omp for
        for (size_t j=0; j<size/2; j++)
            c[j] = a[j]+scalar*b[j];
    }
}

template<typename AT>
inline void triada_remote_seq_accesses(AT scalar, AT *a, AT *b, AT *c, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads/2)
    {
        #pragma omp for
        for (size_t j=size/2; j<size; j++)
            c[j] = a[j]+scalar*b[j];
    }
}

template<typename AT>
inline void triada_remote_seq_accesses_both_sockets_work(AT scalar, AT *a, AT *b, AT *c, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads)
    {
        #pragma omp for
        for (long long j = size; j >= 0; j--)
            c[j] = a[j]+scalar*b[j];
    }
}

template<typename AT>
inline void triada_local_seq_accesses_both_sockets_work(AT scalar, AT *a, AT *b, AT *c, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads)
    {
        #pragma omp for
        for (long long j = 0; j < size; j++)
            c[j] = a[j]+scalar*b[j];
    }
}

template<typename AT>
void kernel(int mode, AT *a, AT *b, AT *c, size_t size)
{
    AT scalar = 123.456;
    if (mode == 0)
        triada_local_seq_accesses(scalar, a, b, c, size);
    else if (mode == 1)
        triada_remote_seq_accesses(scalar, a, b, c, size);
    else if (mode == 2)
        triada_local_seq_accesses_both_sockets_work(scalar, a, b, c, size);
    else if (mode == 3)
        triada_remote_seq_accesses_both_sockets_work(scalar, a, b, c, size);
}