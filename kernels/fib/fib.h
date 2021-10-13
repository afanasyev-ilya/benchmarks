#pragma once

void init(size_t *res, size_t num_steps)
{
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        res[tid] = 0;
    }
}

void re_init(size_t *res, size_t num_steps)
{
    #pragma omp parallel
    {
        unsigned int tid = omp_get_thread_num();
        res[tid] = 0;
    }
}

void fib(size_t *res, size_t num_steps)
{
    #pragma omp parallel
    {
        const int unroll_steps = 10;
        int tid = omp_get_thread_num();

        int n, t1 = 0, t2 = 1, nextTerm = 0;

        for (size_t i = 3; i <= num_steps; i += unroll_steps)
        {
            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;

            nextTerm = t1 + t2;
            t1 = t2;
            t2 = nextTerm;
        }
        res[tid] = nextTerm;
    }
}

void kernel(size_t *res, size_t num_steps)
{
    fib(res, num_steps);
}
