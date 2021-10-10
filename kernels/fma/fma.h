#include <string>

#ifdef __USE_INTEL__
#define SIMD_SIZE 8
#include <immintrin.h>
#endif

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
            out_data[i] = 0;
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
            out_data[i] = 0;
        }
    }
}

template<typename DT>
inline void fma(DT reg1[], DT reg2[])
{
    reg1[0] = reg1[0] * reg2[0] + reg1[0];
    reg1[1] = reg1[1] * reg2[1] + reg1[1];
    reg1[2] = reg1[2] * reg2[2] + reg1[2];
    reg1[3] = reg1[3] * reg2[3] + reg1[3];
    reg1[4] = reg1[4] * reg2[4] + reg1[4];
    reg1[5] = reg1[5] * reg2[5] + reg1[5];
    reg1[6] = reg1[6] * reg2[6] + reg1[6];
    reg1[7] = reg1[7] * reg2[7] + reg1[7];
    /*for(int i = 0; i < SIMD_SIZE; i++)
    {
        reg1[i] = reg1[i] * reg2[i] + reg1[i];
    }*/
}

template<typename DT>
inline void load(DT reg[], DT *memory)
{
    for(int i = 0; i < SIMD_SIZE; i++)
    {
        reg[i] = memory[i];
    }
}

template<typename DT>
inline void store(DT reg[], DT *memory)
{
    for(int i = 0; i < SIMD_SIZE; i++)
    {
         memory[i] = reg[i];
    }
}

template<typename DT>
void kernel_basic(DT *in_data, DT *out_data, size_t size)
{
    const int simd_size = 512 / (sizeof(DT)*8);
    const int num_vectors = 4;

    #pragma omp parallel
    {
        DT reg1[simd_size];
        DT reg2[simd_size];
        DT reg3[simd_size];
        DT reg4[simd_size];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += simd_size*num_vectors)
        {
            load(reg1, &in_data[i]);
            load(reg2, &in_data[i + SIMD_SIZE*1]);
            load(reg3, &in_data[i + SIMD_SIZE*2]);
            load(reg4, &in_data[i + SIMD_SIZE*3]);

            fma(reg1, reg1);
            fma(reg1, reg2);
            fma(reg1, reg3);
            fma(reg1, reg4);

            fma(reg2, reg1);
            fma(reg2, reg2);
            fma(reg2, reg3);
            fma(reg2, reg4);

            fma(reg3, reg1);
            fma(reg3, reg2);
            fma(reg3, reg3);
            fma(reg3, reg4);

            fma(reg4, reg1);
            fma(reg4, reg2);
            fma(reg4, reg3);
            fma(reg4, reg4);

            store(reg1, &out_data[i]);
            store(reg2, &out_data[i + SIMD_SIZE*1]);
            store(reg3, &out_data[i + SIMD_SIZE*2]);
            store(reg4, &out_data[i + SIMD_SIZE*3]);
        }
    };
}

#ifdef __USE_INTEL__
template<typename DT>
void kernel_optimized(DT *in_data, DT *out_data, size_t size)
{
    const int simd_size = 512 / (sizeof(DT)*8);
    const int num_vectors = 4;

    #pragma omp parallel
    {
        __m512 reg1 = _mm512_setzero_ps();
        __m512 reg2 = _mm512_setzero_ps();
        __m512 reg3 = _mm512_setzero_ps();
        __m512 reg4 = _mm512_setzero_ps();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += simd_size*num_vectors)
        {
            reg1 = _mm512_loadu_ps(&(in_data[i]));
            reg2 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*1]));
            reg3 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*2]));
            reg4 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*3]));

            reg1 = _mm512_fmadd_ps(reg1, reg1, reg1);
            reg1 = _mm512_fmadd_ps(reg1, reg2, reg1);
            reg1 = _mm512_fmadd_ps(reg1, reg3, reg1);
            reg1 = _mm512_fmadd_ps(reg1, reg4, reg1);

            reg2 = _mm512_fmadd_ps(reg2, reg1, reg2);
            reg2 = _mm512_fmadd_ps(reg2, reg2, reg2);
            reg2 = _mm512_fmadd_ps(reg2, reg3, reg2);
            reg2 = _mm512_fmadd_ps(reg2, reg4, reg2);

            reg3 = _mm512_fmadd_ps(reg3, reg1, reg3);
            reg3 = _mm512_fmadd_ps(reg3, reg2, reg3);
            reg3 = _mm512_fmadd_ps(reg3, reg3, reg3);
            reg3 = _mm512_fmadd_ps(reg3, reg4, reg3);

            reg4 = _mm512_fmadd_ps(reg4, reg1, reg4);
            reg4 = _mm512_fmadd_ps(reg4, reg2, reg4);
            reg4 = _mm512_fmadd_ps(reg4, reg3, reg4);
            reg4 = _mm512_fmadd_ps(reg4, reg4, reg4);

            _mm512_storeu_ps (&out_data[i], reg1);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*1], reg2);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*2], reg3);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*3], reg4);
        }
    };
}
#endif

template<typename DT>
void kernel(int mode, DT *in_data, DT *out_data, size_t size)
{
    if(mode == 0)
    {
        kernel_basic(in_data, out_data, size);
    }
    else if(mode == 1)
    {
        kernel_optimized(in_data, out_data, size);
    }
}
