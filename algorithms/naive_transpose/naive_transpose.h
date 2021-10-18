#pragma once

template <typename AT>
void init(AT *a, AT *b, size_t size)
{
    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(size_t i = 0; i < size; i++)
        {
            for (size_t j = 0; j < size; j++)
            {
                b[i * size + j] = 0.0;
                a[i * size + j] = rand_r(&seed);
            }
        }
    }
}

template <typename AT>
void KernelTranspIJ(AT *a, AT *b, size_t size)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            for(size_t j = 0; j < size; j++)
            {
                b[i * size + j] = a[j * size + i]; // sequential writes (stores)
            }
        }
    }
}

template <typename AT>
void KernelTranspJI(AT *a, AT *b, size_t size)
{
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            for(size_t j = 0; j < size; j++)
            {
                b[j * size + i] = a[i * size + j]; // sequential reads (loads)
            }
        }
    }
}

template <typename AT>
void KernelBlockTranspIJ(AT *a, AT *b, size_t block_size, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t ii = 0; ii < size; ii += block_size)
    {
        for(size_t jj = 0; jj < size; jj += block_size)
        {
            for (size_t j = jj; j < jj+block_size; j++)
            {
                for (size_t i = ii; i < ii+block_size; i++)
                {
                    if(i < size && j < size)
                        b[j*size + i] = a[i*size + j]; // sequential writes (stores)
                }
            }
        }
    }
}

template <typename AT>
void KernelBlockTranspJI(AT *a, AT *b, size_t block_size, size_t size)
{
    #pragma omp parallel for schedule(static)
    for(size_t ii = 0; ii < size; ii += block_size)
    {
        for(size_t jj = 0; jj < size; jj += block_size)
        {
            for (size_t j = jj; j < jj+block_size; j++)
            {
                for (size_t i = ii; i < ii+block_size; i++)
                {
                    if(i < size && j < size)
                        b[j*size + i] = a[i*size + j]; // sequential reads (loads)
                }
            }
        }
    }
}


template <typename AT>
void kernel(int mode, AT *a, AT *b, size_t block_size, size_t size)
{
    switch(mode)
    {
        //ij
        case 0: KernelTranspIJ(a, b, size); break;

            //ji
        case 1: KernelTranspJI(a, b, size); break;

            //ij block
        case 2: KernelBlockTranspIJ(a, b, block_size, size); break;

            //ji block
        case 3: KernelBlockTranspJI(a, b, block_size, size); break;

        default: fprintf(stderr, "unexpected core type in matrix transpose");
    }
}

