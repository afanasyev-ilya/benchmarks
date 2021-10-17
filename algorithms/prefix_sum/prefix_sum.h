template<typename AT>
void init(AT *x, int size)
{
    srand(time(0));

    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < size; i++)
        {
            x[i] = rand_r(&myseed) / RAND_MAX;
        }
    }
}

template<typename AT>
void prefixsum_inplace(AT *x, int N)
{
    AT *suma;
    #pragma omp parallel
    {
        const int ithread = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        #pragma omp single
        {
            suma = new AT[nthreads+1];
            suma[0] = 0;
        }
        AT sum = 0;
        #pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            sum += x[i];
            x[i] = sum;
        }
        suma[ithread+1] = sum;
        #pragma omp barrier
        AT offset = 0;
        for(int i=0; i<(ithread+1); i++) {
            offset += suma[i];
        }
        #pragma omp for schedule(static)
        for (int i=0; i<N; i++) {
            x[i] += offset;
        }
    }
    delete[] suma;
}

template<typename AT>
void kernel(AT *x, size_t size)
{
    prefixsum_inplace(x, size);
}
