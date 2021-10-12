#include <string>


using std::string;

template<typename DT>
void init(DT *in_data, DT *out_data, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            in_data[i] = rand_r(&myseed);
            out_data[i] = rand_r(&myseed);
        }
    }
}

template<typename DT>
void re_init(DT *in_data, DT *out_data, size_t size)
{
    #pragma omp parallel
    {
        unsigned int myseed = omp_get_thread_num();
        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i++)
        {
            in_data[i] = rand_r(&myseed);
            out_data[i] = rand_r(&myseed);
        }
    }
}

void lehmer_rng(int *seeds, int *result, size_t size, int a, int m)
{
    #pragma novector
    #pragma omp parallel for
    for (size_t i = 0; i < size; i ++)
    {
        int x_cur = seeds[i];
        int x_next = 0;
        #pragma novector
        for(int step = 0; step < INNER_ITERATIONS; step++)
        {
            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

            x_next = a*x_cur % m;
            x_cur = x_next;

        }
        result[i] = x_next;
    }
}

template<typename DT>
void kernel(DT *in_data, DT *out_data, size_t size)
{
    lehmer_rng(in_data, out_data, size, 3, 8);
}
