void init(char *array, size_t size)
{
    for(size_t i = 0; i < size; i++) {
        array[i*3] = rand()%255;;
        array[i*3 + 1] = rand()%255;;
        array[i*3 + 2] = rand()%255;;
    }
}

void kernel(char *array, size_t size)
{
    char K;
    #pragma omp parallel for reduction(min:K)
    for(size_t i = 0; i < size; i++)
    {
       array[i*3] = 255 - array[i*3];
       K = min(K, array[i*3]);
       array[i*3 + 1] = 255 - array[i*3 + 1];
       K = min(K, array[i*3 + 1]);
       array[i*3 + 2] = 255 - array[i*3 + 2];
       K = min(K, array[i*3 + 2]);
    }

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        array[i*3] -= K;
        array[i*3 + 1] -= K;
        array[i*3 + 2] -= K;
    }
}