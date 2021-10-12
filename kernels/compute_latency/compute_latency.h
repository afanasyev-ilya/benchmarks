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

#define AVX_512_FMA_GROUP_D(reg) \
reg1 = _mm512_fmadd_pd(reg1, reg, reg);\
reg2 = _mm512_fmadd_pd(reg2, reg, reg);\
reg3 = _mm512_fmadd_pd(reg3, reg, reg);\
reg4 = _mm512_fmadd_pd(reg4, reg, reg);\
reg5 = _mm512_fmadd_pd(reg5, reg, reg);\
reg6 = _mm512_fmadd_pd(reg6, reg, reg);\
reg7 = _mm512_fmadd_pd(reg7, reg, reg);\
reg8 = _mm512_fmadd_pd(reg8, reg, reg);
#endif

#ifdef __USE_ARM_NEON__
#define ARM_NEON_FMA_GROUP_S(reg) \
reg1 = vfmaq_laneq_f32(reg1, reg, reg, 0);\
reg2 = vfmaq_laneq_f32(reg2, reg, reg, 0);\
reg3 = vfmaq_laneq_f32(reg3, reg, reg, 0);\
reg4 = vfmaq_laneq_f32(reg4, reg, reg, 0);\
reg5 = vfmaq_laneq_f32(reg5, reg, reg, 0);\
reg6 = vfmaq_laneq_f32(reg6, reg, reg, 0);\
reg7 = vfmaq_laneq_f32(reg7, reg, reg, 0);\
reg8 = vfmaq_laneq_f32(reg8, reg, reg, 0);

#define ARM_NEON_FMA_GROUP_D(reg) \
reg1 = vfmaq_laneq_f64(reg1, reg, reg, 0);\
reg2 = vfmaq_laneq_f64(reg2, reg, reg, 0);\
reg3 = vfmaq_laneq_f64(reg3, reg, reg, 0);\
reg4 = vfmaq_laneq_f64(reg4, reg, reg, 0);\
reg5 = vfmaq_laneq_f64(reg5, reg, reg, 0);\
reg6 = vfmaq_laneq_f64(reg6, reg, reg, 0);\
reg7 = vfmaq_laneq_f64(reg7, reg, reg, 0);\
reg8 = vfmaq_laneq_f64(reg8, reg, reg, 0);
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

void kernel_asm(double *in_data, double *out_data, size_t size)
{
    #pragma omp parallel
    {
        __m512d reg1 = _mm512_setzero_pd();
        __m512d reg2 = _mm512_setzero_pd();
        __m512d reg3 = _mm512_setzero_pd();
        __m512d reg4 = _mm512_setzero_pd();
        __m512d reg5 = _mm512_setzero_pd();
        __m512d reg6 = _mm512_setzero_pd();
        __m512d reg7 = _mm512_setzero_pd();
        __m512d reg8 = _mm512_setzero_pd();

        __m512d reg1_old = _mm512_setzero_pd();
        __m512d reg2_old = _mm512_setzero_pd();
        __m512d reg3_old = _mm512_setzero_pd();
        __m512d reg4_old = _mm512_setzero_pd();
        __m512d reg5_old = _mm512_setzero_pd();
        __m512d reg6_old = _mm512_setzero_pd();
        __m512d reg7_old = _mm512_setzero_pd();
        __m512d reg8_old = _mm512_setzero_pd();

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE_D)
        {
            reg1 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*0]));
            reg2 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*1]));
            reg3 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*2]));
            reg4 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*3]));
            reg5 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*4]));
            reg6 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*5]));
            reg7 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*6]));
            reg8 = _mm512_loadu_pd(&(in_data[i + SIMD_SIZE_D*7]));

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

                AVX_512_FMA_GROUP_D(reg1_old)
                AVX_512_FMA_GROUP_D(reg2_old)
                AVX_512_FMA_GROUP_D(reg3_old)
                AVX_512_FMA_GROUP_D(reg4_old)
                AVX_512_FMA_GROUP_D(reg5_old)
                AVX_512_FMA_GROUP_D(reg6_old)
                AVX_512_FMA_GROUP_D(reg7_old)
                AVX_512_FMA_GROUP_D(reg8_old)
            }

            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*0], reg1);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*1], reg2);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*2], reg3);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*3], reg4);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*4], reg5);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*5], reg6);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*6], reg7);
            _mm512_storeu_pd (&out_data[i + SIMD_SIZE_D*7], reg8);
        }
    }
}
#endif


#ifdef __USE_KUNPENG_920__
void kernel_asm(float *in_data, float *out_data, size_t size)
{
    #pragma omp parallel
    {
        float32x4_t reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
        float32x4_t reg1_old, reg2_old, reg3_old, reg4_old, reg5_old, reg6_old, reg7_old, reg8_old;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE_S)
        {
            reg1 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*0]));
            reg2 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*1]));
            reg3 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*2]));
            reg4 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*3]));
            reg5 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*4]));
            reg6 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*5]));
            reg7 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*6]));
            reg8 = vld1q_f32(&(in_data[i + SIMD_SIZE_S*7]));

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

                ARM_NEON_FMA_GROUP_S(reg1_old)
                ARM_NEON_FMA_GROUP_S(reg2_old)
                ARM_NEON_FMA_GROUP_S(reg3_old)
                ARM_NEON_FMA_GROUP_S(reg4_old)
                ARM_NEON_FMA_GROUP_S(reg5_old)
                ARM_NEON_FMA_GROUP_S(reg6_old)
                ARM_NEON_FMA_GROUP_S(reg7_old)
                ARM_NEON_FMA_GROUP_S(reg8_old)
            }

            vst1q_f32 (&out_data[i + SIMD_SIZE_D*0], reg1);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*1], reg2);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*2], reg3);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*3], reg4);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*4], reg5);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*5], reg6);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*6], reg7);
            vst1q_f32 (&out_data[i + SIMD_SIZE_D*7], reg8);
        }
    }
}

void kernel_asm(double *in_data, double *out_data, size_t size)
{
    #pragma omp parallel
    {
        float64x2_t reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
        float64x2_t reg1_old, reg2_old, reg3_old, reg4_old, reg5_old, reg6_old, reg7_old, reg8_old;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE_D)
        {
            reg1 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*0]));
            reg2 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*1]));
            reg3 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*2]));
            reg4 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*3]));
            reg5 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*4]));
            reg6 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*5]));
            reg7 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*6]));
            reg8 = vld1q_f64(&(in_data[i + SIMD_SIZE_D*7]));

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

                ARM_NEON_FMA_GROUP_D(reg1_old)
                ARM_NEON_FMA_GROUP_D(reg2_old)
                ARM_NEON_FMA_GROUP_D(reg3_old)
                ARM_NEON_FMA_GROUP_D(reg4_old)
                ARM_NEON_FMA_GROUP_D(reg5_old)
                ARM_NEON_FMA_GROUP_D(reg6_old)
                ARM_NEON_FMA_GROUP_D(reg7_old)
                ARM_NEON_FMA_GROUP_D(reg8_old)
            }

            vst1q_f64 (&out_data[i + SIMD_SIZE_D*0], reg1);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*1], reg2);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*2], reg3);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*3], reg4);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*4], reg5);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*5], reg6);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*6], reg7);
            vst1q_f64 (&out_data[i + SIMD_SIZE_D*7], reg8);
        }
    }
}
#endif


template<typename DT, int SIMD_SIZE>
void kernel(DT *in_data, DT *out_data, size_t size)
{
    kernel_asm(in_data, out_data, size);
}
