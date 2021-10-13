void init(int *array, size_t size)
{
    for(size_t i = 0; i < size; i++)
        array[i] = 0;
}

size_t kernel_cnt(int *array, size_t size)
{
    size_t cnt = 0;
    for(size_t i = 2; i < size; i++)
    {
        array[i] = 1;
        for(size_t j = 2; j < i; j++)
        {
            cnt++;
            if(i % j == 0)
            {
                array[i] = 0;
                break;
            }
        }
    }
    return cnt;
}

void kernel(int *array, size_t size)
{
    #pragma omp parallel for
    for(size_t i = 2; i < size; i++)
    {
        array[i] = 1;
        for(size_t j = 2; j < i; j++)
        {
            if(i % j == 0) {
                array[i] = 0;
                break;
            }
        }
    }
}
