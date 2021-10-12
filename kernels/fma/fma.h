#include <string>

#ifdef __USE_INTEL__
#define SIMD_SIZE 16
#include <immintrin.h>
#endif

using std::string;

#define INNER_FMA_ITERATIONS 10000
#define NUM_VECTORS 8

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
inline void fma(DT reg_res[], DT a[], DT b[], DT c[])
{
    reg_res[0] = a[0] * b[0] + c[0];
    reg_res[1] = a[1] * b[1] + c[1];
    reg_res[2] = a[2] * b[2] + c[2];
    reg_res[3] = a[3] * b[3] + c[3];
    reg_res[4] = a[4] * b[4] + c[4];
    reg_res[5] = a[5] * b[5] + c[5];
    reg_res[6] = a[6] * b[6] + c[6];
    reg_res[7] = a[7] * b[7] + c[7];
}

#define FMA_GROUP(reg) \
fma(reg1, reg1, reg, reg);        \
fma(reg2, reg2, reg, reg);        \
fma(reg3, reg3, reg, reg);        \
fma(reg4, reg4, reg, reg);        \
fma(reg5, reg5, reg, reg);        \
fma(reg6, reg6, reg, reg);        \
fma(reg7, reg7, reg, reg);        \
fma(reg8, reg8, reg, reg);        \

#define AVX_FMA_GROUP(reg) \
reg1 = _mm512_fmadd_ps(reg1, reg, reg);\
reg2 = _mm512_fmadd_ps(reg2, reg, reg);\
reg3 = _mm512_fmadd_ps(reg3, reg, reg);\
reg4 = _mm512_fmadd_ps(reg4, reg, reg);\
reg5 = _mm512_fmadd_ps(reg5, reg, reg);\
reg6 = _mm512_fmadd_ps(reg6, reg, reg);\
reg7 = _mm512_fmadd_ps(reg7, reg, reg);\
reg8 = _mm512_fmadd_ps(reg8, reg, reg);\

template<typename DT>
inline void load(DT reg[], DT *memory)
{
    #pragma unroll(SIMD_SIZE)
    for(int i = 0; i < SIMD_SIZE; i++)
    {
        reg[i] = memory[i];
    }
}

template<typename DT>
inline void store(DT reg[], DT *memory)
{
    #pragma unroll(SIMD_SIZE)
    for(int i = 0; i < SIMD_SIZE; i++)
    {
         memory[i] = reg[i];
    }
}

template<typename DT>
inline void copy_reg(DT dst[], DT src[])
{
    #pragma unroll(SIMD_SIZE)
    for(int i = 0; i < SIMD_SIZE; i++)
    {
        dst[i] = src[i];
    }
}

template<typename DT>
void kernel_basic(DT *in_data, DT *out_data, size_t size)
{
    #pragma omp parallel
    {
        DT reg1[SIMD_SIZE];
        DT reg2[SIMD_SIZE];
        DT reg3[SIMD_SIZE];
        DT reg4[SIMD_SIZE];
        DT reg5[SIMD_SIZE];
        DT reg6[SIMD_SIZE];
        DT reg7[SIMD_SIZE];
        DT reg8[SIMD_SIZE];

        DT reg1_old[SIMD_SIZE];
        DT reg2_old[SIMD_SIZE];
        DT reg3_old[SIMD_SIZE];
        DT reg4_old[SIMD_SIZE];
        DT reg5_old[SIMD_SIZE];
        DT reg6_old[SIMD_SIZE];
        DT reg7_old[SIMD_SIZE];
        DT reg8_old[SIMD_SIZE];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += SIMD_SIZE*NUM_VECTORS)
        {
            load(reg1, &in_data[i + SIMD_SIZE*0]);
            load(reg2, &in_data[i + SIMD_SIZE*1]);
            load(reg3, &in_data[i + SIMD_SIZE*2]);
            load(reg4, &in_data[i + SIMD_SIZE*3]);
            load(reg1, &in_data[i + SIMD_SIZE*4]);
            load(reg2, &in_data[i + SIMD_SIZE*5]);
            load(reg3, &in_data[i + SIMD_SIZE*6]);
            load(reg4, &in_data[i + SIMD_SIZE*7]);

            #pragma unroll(10)
            for(int step = 0; step < INNER_FMA_ITERATIONS; step++)
            {
                copy_reg(reg1_old, reg1);
                copy_reg(reg2_old, reg2);
                copy_reg(reg3_old, reg3);
                copy_reg(reg4_old, reg4);
                copy_reg(reg5_old, reg5);
                copy_reg(reg6_old, reg6);
                copy_reg(reg7_old, reg7);
                copy_reg(reg8_old, reg8);

                FMA_GROUP(reg1_old)
                FMA_GROUP(reg2_old)
                FMA_GROUP(reg3_old)
                FMA_GROUP(reg4_old)
                FMA_GROUP(reg5_old)
                FMA_GROUP(reg6_old)
                FMA_GROUP(reg7_old)
                FMA_GROUP(reg8_old)
            }

            store(reg1, &out_data[i + SIMD_SIZE*0]);
            store(reg2, &out_data[i + SIMD_SIZE*1]);
            store(reg3, &out_data[i + SIMD_SIZE*2]);
            store(reg4, &out_data[i + SIMD_SIZE*3]);
            store(reg1, &out_data[i + SIMD_SIZE*4]);
            store(reg2, &out_data[i + SIMD_SIZE*5]);
            store(reg3, &out_data[i + SIMD_SIZE*6]);
            store(reg4, &out_data[i + SIMD_SIZE*7]);
        }
    };
}

#ifdef __USE_INTEL__
template<typename DT>
void kernel_asm(DT *in_data, DT *out_data, size_t size)
{
    const int simd_size = 512 / (sizeof(DT)*8);
    const int num_vectors = 4;

    #pragma omp parallel
    {
        __m512 reg1 = _mm512_setzero_ps();
        __m512 reg2 = _mm512_setzero_ps();
        __m512 reg3 = _mm512_setzero_ps();
        __m512 reg4 = _mm512_setzero_ps();
        __m512 reg5 = _mm512_setzero_ps();
        __m512 reg6 = _mm512_setzero_ps();
        __m512 reg7 = _mm512_setzero_ps();
        __m512 reg8 = _mm512_setzero_ps();

        __m512 reg1_old = _mm512_setzero_ps();
        __m512 reg2_old = _mm512_setzero_ps();
        __m512 reg3_old = _mm512_setzero_ps();
        __m512 reg4_old = _mm512_setzero_ps();
        __m512 reg5_old = _mm512_setzero_ps();
        __m512 reg6_old = _mm512_setzero_ps();
        __m512 reg7_old = _mm512_setzero_ps();
        __m512 reg8_old = _mm512_setzero_ps();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE)
        {
            reg1 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*0]));
            reg2 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*1]));
            reg3 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*2]));
            reg4 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*3]));
            reg5 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*4]));
            reg6 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*5]));
            reg7 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*6]));
            reg8 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE*7]));

            for(int step = 0; step < INNER_FMA_ITERATIONS; step++)
            {
                reg1_old = reg1;
                reg2_old = reg2;
                reg3_old = reg3;
                reg4_old = reg4;
                reg5_old = reg5;
                reg6_old = reg6;
                reg7_old = reg7;
                reg8_old = reg8;

                AVX_FMA_GROUP(reg1)
                AVX_FMA_GROUP(reg2)
                AVX_FMA_GROUP(reg3)
                AVX_FMA_GROUP(reg4)
                AVX_FMA_GROUP(reg5)
                AVX_FMA_GROUP(reg6)
                AVX_FMA_GROUP(reg7)
                AVX_FMA_GROUP(reg8)
            }

            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*0], reg1);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*1], reg2);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*2], reg3);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*3], reg4);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*4], reg5);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*5], reg6);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*6], reg7);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE*7], reg8);
        }
    }
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
        kernel_asm(in_data, out_data, size);
    }
}
