#include <string>

#ifdef __USE_INTEL__
#include <immintrin.h>
#elif __USE_KUNPENG_920__
#include <arm_neon.h>
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

#ifdef __USE_AVX_512__
#define AVX_512_FMA_GROUP_S(reg) \
reg1 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg1), reg, reg);\
reg2 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg2), reg, reg);\
reg3 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg3), reg, reg);\
reg4 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg4), reg, reg);\
reg5 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg5), reg, reg);\
reg6 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg6), reg, reg);\
reg7 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg7), reg, reg);\
reg8 = _mm512_fmadd_ps(_mm512_invsqrt_ps(reg8), reg, reg);
#endif

#ifdef __USE_INTEL__
void kernel_asm(float *in_data, float *out_data, size_t size)
{
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
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE_S)
        {
            reg1 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*0]));
            reg2 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*1]));
            reg3 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*2]));
            reg4 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*3]));
            reg5 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*4]));
            reg6 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*5]));
            reg7 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*6]));
            reg8 = _mm512_loadu_ps(&(in_data[i + SIMD_SIZE_S*7]));

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

                AVX_512_FMA_GROUP_S(reg1_old)
                AVX_512_FMA_GROUP_S(reg2_old)
                AVX_512_FMA_GROUP_S(reg3_old)
                AVX_512_FMA_GROUP_S(reg4_old)
                AVX_512_FMA_GROUP_S(reg5_old)
                AVX_512_FMA_GROUP_S(reg6_old)
                AVX_512_FMA_GROUP_S(reg7_old)
                AVX_512_FMA_GROUP_S(reg8_old)
            }

            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*0], reg1);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*1], reg2);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*2], reg3);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*3], reg4);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*4], reg5);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*5], reg6);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*6], reg7);
            _mm512_storeu_ps (&out_data[i + SIMD_SIZE_S*7], reg8);
        }
    }
}
#endif


#define GENERIC_LOAD(reg, offset, data) \
reg[0] = data[offset + 0];  \
reg[1] = data[offset + 1];  \
reg[2] = data[offset + 2];  \
reg[3] = data[offset + 3];  \
reg[4] = data[offset + 4];  \
reg[5] = data[offset + 5];  \
reg[6] = data[offset + 6];  \
reg[7] = data[offset + 7];  \
reg[8] = data[offset + 8];  \
reg[9] = data[offset + 9];  \
reg[10] = data[offset + 10];  \
reg[11] = data[offset + 11];  \
reg[12] = data[offset + 12];  \
reg[13] = data[offset + 13];  \
reg[14] = data[offset + 14];  \
reg[15] = data[offset + 15];

#define GENERIC_COPY(dst, src) \
dst[0] = src[0];  \
dst[1] = src[1];  \
dst[2] = src[2];  \
dst[3] = src[3];  \
dst[4] = src[4];  \
dst[5] = src[5];  \
dst[6] = src[6];  \
dst[7] = src[7];  \
dst[8] = src[8];  \
dst[9] = src[9];  \
dst[10] = src[10];  \
dst[11] = src[11];  \
dst[12] = src[12];  \
dst[13] = src[13];  \
dst[14] = src[14];  \
dst[15] = src[15];

#define GENERIC_OP(result, fr, sr) \
result[0] = sqrt(fr[0])+sr[0];\
result[1] = sqrt(fr[1])+sr[1];\
result[2] = sqrt(fr[2])+sr[2];\
result[3] = sqrt(fr[3])+sr[3];\
result[4] = sqrt(fr[4])+sr[4];\
result[5] = sqrt(fr[5])+sr[5];\
result[6] = sqrt(fr[6])+sr[6];\
result[7] = sqrt(fr[7])+sr[7];\
result[8] = sqrt(fr[8])+sr[8];\
result[9] = sqrt(fr[9])+sr[9];\
result[10] = sqrt(fr[10])+sr[10];\
result[11] = sqrt(fr[11])+sr[11];\
result[12] = sqrt(fr[12])+sr[12];\
result[13] = sqrt(fr[13])+sr[13];\
result[14] = sqrt(fr[14])+sr[14];\
result[15] = sqrt(fr[15])+sr[15];\

#define GENERIC_GROUP(reg) \
GENERIC_OP(reg1, reg1_old, reg);\
GENERIC_OP(reg2, reg2_old, reg);\
GENERIC_OP(reg3, reg3_old, reg);\
GENERIC_OP(reg4, reg4_old, reg);\
GENERIC_OP(reg5, reg5_old, reg);\
GENERIC_OP(reg6, reg6_old, reg);\
GENERIC_OP(reg7, reg7_old, reg);\
GENERIC_OP(reg8, reg8_old, reg);\


#define GENERIC_STORE(reg, offset, data) \
data[offset + 0] = reg[0];  \
data[offset + 1] = reg[1];  \
data[offset + 2] = reg[2];  \
data[offset + 3] = reg[3];  \
data[offset + 4] = reg[4];  \
data[offset + 5] = reg[5];  \
data[offset + 6] = reg[6];  \
data[offset + 7] = reg[7];  \
data[offset + 8] = reg[8];  \
data[offset + 9] = reg[9];  \
data[offset + 10] = reg[10];  \
data[offset + 11] = reg[11];  \
data[offset + 12] = reg[12];  \
data[offset + 13] = reg[13];  \
data[offset + 14] = reg[14];  \
data[offset + 15] = reg[15];


void kernel_generic(float *in_data, float *out_data, size_t size)
{
    #pragma omp parallel
    {
        float reg1[SIMD_SIZE_S];
        float reg2[SIMD_SIZE_S];
        float reg3[SIMD_SIZE_S];
        float reg4[SIMD_SIZE_S];
        float reg5[SIMD_SIZE_S];
        float reg6[SIMD_SIZE_S];
        float reg7[SIMD_SIZE_S];
        float reg8[SIMD_SIZE_S];

        float reg1_old[SIMD_SIZE_S];
        float reg2_old[SIMD_SIZE_S];
        float reg3_old[SIMD_SIZE_S];
        float reg4_old[SIMD_SIZE_S];
        float reg5_old[SIMD_SIZE_S];
        float reg6_old[SIMD_SIZE_S];
        float reg7_old[SIMD_SIZE_S];
        float reg8_old[SIMD_SIZE_S];

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS * 16)
        {
            GENERIC_LOAD(reg1, i + SIMD_SIZE_S*0, in_data);
            GENERIC_LOAD(reg2, i + SIMD_SIZE_S*1, in_data);
            GENERIC_LOAD(reg3, i + SIMD_SIZE_S*2, in_data);
            GENERIC_LOAD(reg4, i + SIMD_SIZE_S*3, in_data);
            GENERIC_LOAD(reg5, i + SIMD_SIZE_S*4, in_data);
            GENERIC_LOAD(reg6, i + SIMD_SIZE_S*5, in_data);
            GENERIC_LOAD(reg7, i + SIMD_SIZE_S*6, in_data);
            GENERIC_LOAD(reg8, i + SIMD_SIZE_S*7, in_data);

            for(int step = 0; step < INNER_FMA_ITERATIONS; step++)
            {
                GENERIC_COPY(reg1_old, reg1);
                GENERIC_COPY(reg2_old, reg2);
                GENERIC_COPY(reg3_old, reg3);
                GENERIC_COPY(reg4_old, reg4);
                GENERIC_COPY(reg5_old, reg5);
                GENERIC_COPY(reg6_old, reg6);
                GENERIC_COPY(reg7_old, reg7);
                GENERIC_COPY(reg8_old, reg8);

                GENERIC_GROUP(reg1_old);
                GENERIC_GROUP(reg2_old);
                GENERIC_GROUP(reg3_old);
                GENERIC_GROUP(reg4_old);
                GENERIC_GROUP(reg5_old);
                GENERIC_GROUP(reg6_old);
                GENERIC_GROUP(reg7_old);
                GENERIC_GROUP(reg8_old);
            }

            GENERIC_STORE(reg1, i + SIMD_SIZE_S*0, in_data);
            GENERIC_STORE(reg2, i + SIMD_SIZE_S*1, in_data);
            GENERIC_STORE(reg3, i + SIMD_SIZE_S*2, in_data);
            GENERIC_STORE(reg4, i + SIMD_SIZE_S*3, in_data);
            GENERIC_STORE(reg5, i + SIMD_SIZE_S*4, in_data);
            GENERIC_STORE(reg6, i + SIMD_SIZE_S*5, in_data);
            GENERIC_STORE(reg7, i + SIMD_SIZE_S*6, in_data);
            GENERIC_STORE(reg8, i + SIMD_SIZE_S*7, in_data);
        }
    }
}

template<typename DT>
void kernel(DT *in_data, DT *out_data, size_t size)
{
    kernel_generic(in_data, out_data, size);
}
