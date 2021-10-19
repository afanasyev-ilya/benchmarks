
void init()
{

}

template<typename AT>
void kernel_storage(AT *a, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            a[i] = rand_r(&myseed);
        }
    }
}

template<typename AT>
void kernel(AT *a, size_t size)
{
    kernel_storage(a, size);
}
