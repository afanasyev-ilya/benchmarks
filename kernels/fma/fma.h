#ifdef __USE_INTEL__
#include <immintrin.h>
#elif __USE_KUNPENG_920__
#include <arm_neon.h>
#elif __USE_A64FX__
#include <arm_sve.h>
#endif

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

#if defined (__USE_INTEL__) || defined (__USE_A64FX__)
inline void fma(float reg_res[], float a[], float b[], float c[])
{
    reg_res[0] = a[0] * b[0] + c[0];
    reg_res[1] = a[1] * b[1] + c[1];
    reg_res[2] = a[2] * b[2] + c[2];
    reg_res[3] = a[3] * b[3] + c[3];
    reg_res[4] = a[4] * b[4] + c[4];
    reg_res[5] = a[5] * b[5] + c[5];
    reg_res[6] = a[6] * b[6] + c[6];
    reg_res[7] = a[7] * b[7] + c[7];
    reg_res[8] = a[8] * b[8] + c[8];
    reg_res[9] = a[9] * b[9] + c[9];
    reg_res[10] = a[10] * b[10] + c[10];
    reg_res[11] = a[11] * b[11] + c[11];
    reg_res[12] = a[12] * b[12] + c[12];
    reg_res[13] = a[13] * b[13] + c[13];
    reg_res[14] = a[14] * b[14] + c[14];
    reg_res[15] = a[15] * b[15] + c[15];
}

inline void fma(double reg_res[], double a[], double b[], double c[])
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
#endif

#ifdef __USE_KUNPENG_920__
inline void fma(float reg_res[], float a[], float b[], float c[])
{
    reg_res[0] = a[0] * b[0] + c[0];
    reg_res[1] = a[1] * b[1] + c[1];
    reg_res[2] = a[2] * b[2] + c[2];
    reg_res[3] = a[3] * b[3] + c[3];
}

inline void fma(double reg_res[], double a[], double b[], double c[])
{
    reg_res[0] = a[0] * b[0] + c[0];
    reg_res[1] = a[1] * b[1] + c[1];
}
#endif

#if defined (__USE_INTEL__) || defined (__USE_A64FX__)
inline void copy_reg(float dst[], float src[])
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];
    dst[8] = src[8];
    dst[9] = src[9];
    dst[10] = src[10];
    dst[11] = src[11];
    dst[12] = src[12];
    dst[13] = src[13];
    dst[14] = src[14];
    dst[15] = src[15];
}

inline void copy_reg(double dst[], double src[])
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
    dst[4] = src[4];
    dst[5] = src[5];
    dst[6] = src[6];
    dst[7] = src[7];
}
#endif

#ifdef __USE_KUNPENG_920__
inline void copy_reg(float dst[], float src[])
{
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = src[3];
}

inline void copy_reg(double dst[], double src[])
{
    dst[0] = src[0];
    dst[1] = src[1];
}
#endif

template<typename DT, int SIMD_SIZE>
inline void load(DT reg[], DT *memory)
{
    #pragma unroll(SIMD_SIZE)
    for(int i = 0; i < SIMD_SIZE; i++)
    {
        reg[i] = memory[i];
    }
}

template<typename DT, int SIMD_SIZE>
inline void store(DT reg[], DT *memory)
{
    #pragma unroll(SIMD_SIZE)
    for(int i = 0; i < SIMD_SIZE; i++)
    {
        memory[i] = reg[i];
    }
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

#ifdef __USE_AVX_512__
#define AVX_512_FMA_GROUP_S(reg) \
reg1 = _mm512_fmadd_ps(reg1, reg, reg);\
reg2 = _mm512_fmadd_ps(reg2, reg, reg);\
reg3 = _mm512_fmadd_ps(reg3, reg, reg);\
reg4 = _mm512_fmadd_ps(reg4, reg, reg);\
reg5 = _mm512_fmadd_ps(reg5, reg, reg);\
reg6 = _mm512_fmadd_ps(reg6, reg, reg);\
reg7 = _mm512_fmadd_ps(reg7, reg, reg);\
reg8 = _mm512_fmadd_ps(reg8, reg, reg);

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

#ifdef __USE_SVE__
#define SVE_FMA_GROUP_S(reg) \
reg1 = svmla_lane_f32(reg1, reg, reg, 0);\
reg2 = svmla_lane_f32(reg2, reg, reg, 0);\
reg3 = svmla_lane_f32(reg3, reg, reg, 0);\
reg4 = svmla_lane_f32(reg4, reg, reg, 0);\
reg5 = svmla_lane_f32(reg5, reg, reg, 0);\
reg6 = svmla_lane_f32(reg6, reg, reg, 0);\
reg7 = svmla_lane_f32(reg7, reg, reg, 0);\
reg8 = svmla_lane_f32(reg8, reg, reg, 0);

#define AVX_512_FMA_GROUP_D(reg) \
reg1 = svmla_lane_f64(reg1, reg, reg, 0);\
reg2 = svmla_lane_f64(reg2, reg, reg, 0);\
reg3 = svmla_lane_f64(reg3, reg, reg, 0);\
reg4 = svmla_lane_f64(reg4, reg, reg, 0);\
reg5 = svmla_lane_f64(reg5, reg, reg, 0);\
reg6 = svmla_lane_f64(reg6, reg, reg, 0);\
reg7 = svmla_lane_f64(reg7, reg, reg, 0);\
reg8 = svmla_lane_f64(reg8, reg, reg, 0);
#endif

template<typename DT, int SIMD_SIZE>
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
            load<DT, SIMD_SIZE>(reg1, &in_data[i + SIMD_SIZE*0]);
            load<DT, SIMD_SIZE>(reg2, &in_data[i + SIMD_SIZE*1]);
            load<DT, SIMD_SIZE>(reg3, &in_data[i + SIMD_SIZE*2]);
            load<DT, SIMD_SIZE>(reg4, &in_data[i + SIMD_SIZE*3]);
            load<DT, SIMD_SIZE>(reg1, &in_data[i + SIMD_SIZE*4]);
            load<DT, SIMD_SIZE>(reg2, &in_data[i + SIMD_SIZE*5]);
            load<DT, SIMD_SIZE>(reg3, &in_data[i + SIMD_SIZE*6]);
            load<DT, SIMD_SIZE>(reg4, &in_data[i + SIMD_SIZE*7]);

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

            store<DT, SIMD_SIZE>(reg1, &out_data[i + SIMD_SIZE*0]);
            store<DT, SIMD_SIZE>(reg2, &out_data[i + SIMD_SIZE*1]);
            store<DT, SIMD_SIZE>(reg3, &out_data[i + SIMD_SIZE*2]);
            store<DT, SIMD_SIZE>(reg4, &out_data[i + SIMD_SIZE*3]);
            store<DT, SIMD_SIZE>(reg1, &out_data[i + SIMD_SIZE*4]);
            store<DT, SIMD_SIZE>(reg2, &out_data[i + SIMD_SIZE*5]);
            store<DT, SIMD_SIZE>(reg3, &out_data[i + SIMD_SIZE*6]);
            store<DT, SIMD_SIZE>(reg4, &out_data[i + SIMD_SIZE*7]);
        }
    };
}

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

#ifdef __USE_A64FX__
void kernel_asm(float *in_data, float *out_data, size_t size)
{
    #pragma omp parallel
    {
        svfloat32_t reg1= svdup_n_f32(0);
        svfloat32_t reg2= svdup_n_f32(0);
        svfloat32_t reg3= svdup_n_f32(0);
        svfloat32_t reg4= svdup_n_f32(0);
        svfloat32_t reg5= svdup_n_f32(0);
        svfloat32_t reg6= svdup_n_f32(0);
        svfloat32_t reg7= svdup_n_f32(0);
        svfloat32_t reg8= svdup_n_f32(0);

        svfloat32_t reg1_old= svdup_n_f32(0);
        svfloat32_t reg2_old= svdup_n_f32(0);
        svfloat32_t reg3_old= svdup_n_f32(0);
        svfloat32_t reg4_old= svdup_n_f32(0);
        svfloat32_t reg5_old= svdup_n_f32(0);
        svfloat32_t reg6_old= svdup_n_f32(0);
        svfloat32_t reg7_old= svdup_n_f32(0);
        svfloat32_t reg8_old= svdup_n_f32(0);

        svbool_t pred = svwhilelt_b32_u32(0, size);

        #pragma omp for schedule(static)
        for (size_t i = 0; i < size; i += NUM_VECTORS*SIMD_SIZE_S)
        {
            reg1 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*1);
            reg2 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*2);
            reg3 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*3);
            reg4 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*4);
            reg5 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*5);
            reg6 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*6);
            reg7 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*7);
            reg8 = svld1_f32(pred, in_data + i + SIMD_SIZE_S*8);

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

                /*SVE_FMA_GROUP_S(reg1_old)
                SVE_FMA_GROUP_S(reg2_old)
                SVE_FMA_GROUP_S(reg3_old)
                SVE_FMA_GROUP_S(reg4_old)
                SVE_FMA_GROUP_S(reg5_old)
                SVE_FMA_GROUP_S(reg6_old)
                SVE_FMA_GROUP_S(reg7_old)
                SVE_FMA_GROUP_S(reg8_old)*/
            }
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*1, reg1);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*2, reg2);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*3, reg3);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*4, reg4);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*5, reg5);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*6, reg6);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*7, reg7);
            svst1_f32(pred, out_data + i + SIMD_SIZE_S*8, reg8);
        }
    }
}

void kernel_asm(double *in_data, double *out_data, size_t size)
{

}
#endif

template<typename DT, int SIMD_SIZE>
void kernel(OPT_MODE mode, DT *in_data, DT *out_data, size_t size)
{
    if(mode == GENERIC)
    {
        kernel_basic<DT, SIMD_SIZE>(in_data, out_data, size);
    }
    else if(mode == OPTIMIZED)
    {
        kernel_asm(in_data, out_data, size);
    }
}
