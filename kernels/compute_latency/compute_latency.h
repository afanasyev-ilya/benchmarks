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

#ifdef __USE_ARM_NEON__
#define ARM_NEON_FMA_GROUP_S(reg) \
reciprocal1 = vrsqrteq_f32(reg1);\
reciprocal1 = vmulq_f32(vrsqrtsq_f32(reg1, reciprocal1), reciprocal1);\
reciprocal1 = vmulq_f32(vrsqrtsq_f32(reg1, reciprocal1), reciprocal1);\
reciprocal1 = vmulq_f32(vrsqrtsq_f32(reg1, reciprocal1), reciprocal1);\
reg1 = vfmaq_laneq_f32(reciprocal1, reg, reg, 0);\
reciprocal2 = vrsqrteq_f32(reg2);\
reciprocal2 = vmulq_f32(vrsqrtsq_f32(reg2, reciprocal2), reciprocal2);\
reciprocal2 = vmulq_f32(vrsqrtsq_f32(reg2, reciprocal2), reciprocal2);\
reciprocal2 = vmulq_f32(vrsqrtsq_f32(reg2, reciprocal2), reciprocal2);\
reg2 = vfmaq_laneq_f32(reciprocal2, reg, reg, 0);\
reciprocal3 = vrsqrteq_f32(reg3);\
reciprocal3 = vmulq_f32(vrsqrtsq_f32(reg3, reciprocal3), reciprocal3);\
reciprocal3 = vmulq_f32(vrsqrtsq_f32(reg3, reciprocal3), reciprocal3);\
reciprocal3 = vmulq_f32(vrsqrtsq_f32(reg3, reciprocal3), reciprocal3);\
reg3 = vfmaq_laneq_f32(reciprocal3, reg, reg, 0);\
reciprocal4 = vrsqrteq_f32(reg4);\
reciprocal4 = vmulq_f32(vrsqrtsq_f32(reg4, reciprocal4), reciprocal4);\
reciprocal4 = vmulq_f32(vrsqrtsq_f32(reg4, reciprocal4), reciprocal4);\
reciprocal4 = vmulq_f32(vrsqrtsq_f32(reg4, reciprocal4), reciprocal4);\
reg4 = vfmaq_laneq_f32(reciprocal4, reg, reg, 0);\
reciprocal5 = vrsqrteq_f32(reg5);\
reciprocal5 = vmulq_f32(vrsqrtsq_f32(reg5, reciprocal5), reciprocal5);\
reciprocal5 = vmulq_f32(vrsqrtsq_f32(reg5, reciprocal5), reciprocal5);\
reciprocal5 = vmulq_f32(vrsqrtsq_f32(reg5, reciprocal5), reciprocal5);\
reg5 = vfmaq_laneq_f32(reciprocal5, reg, reg, 0);\
reciprocal6 = vrsqrteq_f32(reg1);\
reciprocal6 = vmulq_f32(vrsqrtsq_f32(reg6, reciprocal6), reciprocal6);\
reciprocal6 = vmulq_f32(vrsqrtsq_f32(reg6, reciprocal6), reciprocal6);\
reciprocal6 = vmulq_f32(vrsqrtsq_f32(reg6, reciprocal6), reciprocal6);\
reg6 = vfmaq_laneq_f32(reciprocal6, reg, reg, 0);\
reciprocal7 = vrsqrteq_f32(reg7);\
reciprocal7 = vmulq_f32(vrsqrtsq_f32(reg7, reciprocal7), reciprocal7);\
reciprocal7 = vmulq_f32(vrsqrtsq_f32(reg7, reciprocal7), reciprocal7);\
reciprocal7 = vmulq_f32(vrsqrtsq_f32(reg7, reciprocal7), reciprocal7);\
reg7 = vfmaq_laneq_f32(reciprocal7, reg, reg, 0);\
reciprocal8 = vrsqrteq_f32(reg1);\
reciprocal8 = vmulq_f32(vrsqrtsq_f32(reg8, reciprocal8), reciprocal8);\
reciprocal8 = vmulq_f32(vrsqrtsq_f32(reg8, reciprocal8), reciprocal8);\
reciprocal8 = vmulq_f32(vrsqrtsq_f32(reg8, reciprocal8), reciprocal8);\
reg8 = vfmaq_laneq_f32(reciprocal8, reg, reg, 0);
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


#ifdef __USE_KUNPENG_920__
void kernel_asm(float *in_data, float *out_data, size_t size)
{
    #pragma omp parallel
    {
        float32x4_t reg1, reg2, reg3, reg4, reg5, reg6, reg7, reg8;
        float32x4_t reg1_old, reg2_old, reg3_old, reg4_old, reg5_old, reg6_old, reg7_old, reg8_old;
        float32x4_t reciprocal1, reciprocal2, reciprocal3, reciprocal4, reciprocal5, reciprocal6, reciprocal7, reciprocal8;

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
#endif

#ifdef __USE_A64FX__
void kernel_asm(float *in_data, float *out_data, size_t size)
{

}
#endif

template<typename DT, int SIMD_SIZE>
void kernel(DT *in_data, DT *out_data, size_t size)
{
    kernel_asm(in_data, out_data, size);
}
