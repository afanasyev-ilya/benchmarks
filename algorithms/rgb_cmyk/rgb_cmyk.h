void init(base_type *array, size_t size)
{
    for(size_t i = 0; i < size; i++) {
        array[i*3] = rand()%255;;
        array[i*3 + 1] = rand()%255;;
        array[i*3 + 2] = rand()%255;;
    }
}

#define RGB_STRUCTS(array_pos, rgb_pos) ((array_pos)*3+rgb_pos)
#define RGB_UNROLLED(array_pos, rgb_pos) ((rgb_pos)*size+array_pos)

void kernel_structs(base_type *array, size_t size)
{
    base_type K = 255;
    #pragma omp parallel for reduction(min:K)
    for(size_t i = 0; i < size; i++)
    {
       array[RGB_STRUCTS(i, 0)] = 255 - array[RGB_STRUCTS(i, 0)];
       K = min(K, array[RGB_STRUCTS(i, 0)]);
       array[RGB_STRUCTS(i, 1)] = 255 - array[RGB_STRUCTS(i, 1)];
       K = min(K, array[RGB_STRUCTS(i, 1)]);
       array[RGB_STRUCTS(i, 2)] = 255 - array[RGB_STRUCTS(i, 2)];
       K = min(K, array[RGB_STRUCTS(i, 2)]);
    }

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        array[RGB_STRUCTS(i, 0)] -= K;
        array[RGB_STRUCTS(i, 1)] -= K;
        array[RGB_STRUCTS(i, 2)] -= K;
    }
}

void kernel_optimized(base_type *array, size_t size)
{
    base_type K = 255;
    #pragma omp parallel for reduction(min:K)
    for(size_t i = 0; i < size; i++)
    {
        array[ RGB_UNROLLED(i, 0)] = 255 - array[ RGB_UNROLLED(i, 0)];
        K = min(K, array[ RGB_UNROLLED(i, 0)]);
        array[ RGB_UNROLLED(i, 1)] = 255 - array[ RGB_UNROLLED(i, 1)];
        K = min(K, array[ RGB_UNROLLED(i, 1)]);
        array[ RGB_UNROLLED(i, 2)] = 255 - array[ RGB_UNROLLED(i, 2)];
        K = min(K, array[ RGB_UNROLLED(i, 2)]);
    }

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        array[ RGB_UNROLLED(i, 0)] -= K;
        array[ RGB_UNROLLED(i, 1)] -= K;
        array[ RGB_UNROLLED(i, 2)] -= K;
    }
}

void kernel(OPT_MODE opt_mode, base_type *array, size_t size)
{
    if(opt_mode == OPTIMIZED)
        kernel_optimized(array, size);
    if(opt_mode == GENERIC)
        kernel_structs(array, size);
}