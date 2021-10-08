#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#elif defined(__AVX512F__)
#include <immintrin.h>
#elif defined(__AVX2__)
#include <immintrin.h>
#elif defined(__AVX__)
#include <immintrin.h>
#else
#warning Intrinsic version not found, fall-back will be called
#endif /* __ARM_FEATURE_SVE */


void avx_spmv_sell32()
{
    /*const uint64_t C_value = 32;
    if(mat->C != C_value)
    {
        ERROR_PRINT("Wrong kernel this is SELL-%d\n", (int)C_value);
    }
    else
    {
        INFO_PRINT("AVX512: SELL-%d kernel\n", C_value);
    }

    #pragma omp parallel
    {
        double *rval = x->val;
        double *mval = mat->valSellC;
        int *col = mat->colSellC;
        __m512d val;
        __m512d rhs = _mm512_setzero_pd();

        #pragma omp for schedule(static)
        for(int chunk=0; chunk<mat->nchunks; ++chunk)
        {
            __m512d tmp0 = _mm512_setzero_pd();
            __m512d tmp1 = _mm512_setzero_pd();
            __m512d tmp2 = _mm512_setzero_pd();
            __m512d tmp3 = _mm512_setzero_pd();
            int offs = mat->chunkPtr[chunk];
            __m256i idx;
            #pragma GCC unroll 1 //no unrolling
            for(int j=0; j<mat->chunkLen[chunk]; ++j)
            {

                val = _mm512_load_pd(&mval[offs]);idx = _mm256_load_si256((__m256i *)(&col[offs]));rhs = _mm512_i32gather_pd(idx,rval,8); tmp0 = _mm512_fmadd_pd(val,rhs,tmp0); offs+=8;
                val = _mm512_load_pd(&mval[offs]);idx = _mm256_load_si256((__m256i *)(&col[offs]));rhs = _mm512_i32gather_pd(idx,rval,8); tmp1 = _mm512_fmadd_pd(val,rhs,tmp1); offs+=8;
                val = _mm512_load_pd(&mval[offs]);idx = _mm256_load_si256((__m256i *)(&col[offs]));rhs = _mm512_i32gather_pd(idx,rval,8); tmp2 = _mm512_fmadd_pd(val,rhs,tmp2); offs+=8;
                val = _mm512_load_pd(&mval[offs]);idx = _mm256_load_si256((__m256i *)(&col[offs]));rhs = _mm512_i32gather_pd(idx,rval,8); tmp3 = _mm512_fmadd_pd(val,rhs,tmp3); offs+=8;
            }
            _mm512_storeu_pd(&(b->val[chunk*C_value+(8*0)]),tmp0);
            _mm512_storeu_pd(&(b->val[chunk*C_value+(8*1)]),tmp1);
            _mm512_storeu_pd(&(b->val[chunk*C_value+(8*2)]),tmp2);
            _mm512_storeu_pd(&(b->val[chunk*C_value+(8*3)]),tmp3);
        }
    }*/
}