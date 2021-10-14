#pragma once

template<typename AT>
void init(int mode, AT *result, AT *accessed, int *rand_indexes, size_t size, size_t radius)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        unsigned int myseed = tid;

        if(mode == 0) // local
        {
            if(tid == 0)
            {
                for (size_t i = 0; i < size; i++) // everything is on socket 0
                {
                    result[i] = 0;
                    rand_indexes[i] = (int)rand_r(&myseed) % radius;
                }

                for (size_t i = 0; i < size; i++) // random accessed array is on socket 0
                {
                    accessed[i] = rand_r(&myseed);
                }
            }
        }
        else if(mode == 1) // remote
        {
            if(tid == 0)
            {
                for (size_t i = 0; i < size; i++) // everything is on socket 0
                {
                    result[i] = 0;
                    rand_indexes[i] = (int) rand_r(&myseed) % radius;
                }
            }

            if(tid == (omp_get_thread_num() - 1))
            {
                for (size_t i = 0; i < size; i++) // random accessed array is on socket 1 (different)
                {
                    accessed[i] = rand_r(&myseed);
                }
            }
        }
    }
}

template<typename AT>
inline void triada_local_rand_accesses_one_socket(AT *result, AT *accessed, int *rand_indexes, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads/2)
    {
        #pragma omp for
        for (long long j = 0; j < size; j++)
            result[j] = accessed[ rand_indexes[j] ];
    }
}

template<typename AT>
inline void triada_remote_rand_accesses_one_socket(AT *result, AT *accessed, int *rand_indexes, size_t size)
{
    int max_threads = omp_get_max_threads();
    #pragma omp parallel num_threads(max_threads/2)
    {
        #pragma omp for
        for (long long j = 0; j < size; j++)
            result[j] = accessed[ rand_indexes[j] ];
    }
}

template<typename AT>
void kernel(int mode, AT *result, AT *accessed, int *rand_indexes, size_t size)
{
    AT scalar = 123.456;
    if (mode == 0)
        triada_local_rand_accesses_one_socket(result, accessed, rand_indexes, size);
    else if (mode == 1)
        triada_remote_rand_accesses_one_socket(result, accessed, rand_indexes, size);
}