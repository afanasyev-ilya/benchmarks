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

template<typename AT>
void kernel(int mode, AT * __restrict__ a, const AT * __restrict__ b, const size_t size, const int radius)
{
    kernel_basic(a, b, size, radius);
}
