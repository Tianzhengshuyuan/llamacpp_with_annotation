// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-quants.h"
#include <stdio.h>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__) || defined(__loongarch_asx)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

#ifdef __clang__
#define VREGS_PREFIX "$vr"
#define XREGS_PREFIX "$xr"
#else // GCC
#define VREGS_PREFIX "$f"
#define XREGS_PREFIX "$f"
#endif
#define __ALL_REGS "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31"

namespace {

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
// 在一个 256 位寄存器中并行地对 8 个 32 位浮点数（单精度浮点数）进行加法运算。
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif // __AVX__

#if defined(__loongarch_asx)
inline __m256 add(__m256 x, __m256 y) { return __lasx_xvfadd_s(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return __lasx_xvfsub_s(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return  __lasx_xvfmul_s(x, y); }
#endif // __loongarch_asx

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
inline __m512 sub(__m512 x, __m512 y) { return _mm512_sub_ps(x, y); }
inline __m512 mul(__m512 x, __m512 y) { return _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) { return vaddq_f16(x, y); }
inline float16x8_t sub(float16x8_t x, float16x8_t y) { return vsubq_f16(x, y); }
inline float16x8_t mul(float16x8_t x, float16x8_t y) { return vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

#if defined(__FMA__)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    //对256位向量中的32位浮点，a[i]*b[i]+c[i], 0<=i<8
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#endif

#if defined(__loongarch_asx)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    //对256位向量中的32位浮点，a[i]*b[i]+c[i], 0<=i<8
    return __lasx_xvfmadd_s(a, b, c);
}
#endif //__loongarch_asx

#if defined(__ARM_FEATURE_FMA)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vfmaq_f32(c, b, a);
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
template <>
inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vfmaq_f16(c, b, a);
}
#endif
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
inline float hsum(float32x4_t x) {
    //把128位向量中所有32位浮点数加起来
    return vaddvq_f32(x);
}
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
inline float hsum(float16x8_t x) {
    return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)),
                                vcvt_f32_f16(vget_high_f16(x))));
}
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    //_mm_movehl_ps(x, x)把x的高64位放到新向量的高、低64位
    //_mm_add_ps对4对浮点数执行加法：[x[0],x[1],x[2],x[3]] + [x[2],x[3],x[2],x[3]] = [x[0]+x[2],x[1]+x[3],x[2]+x[2],x[3]+x[3]]
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    //_mm_movehdup_ps的作用是[a0, a1, a2, a3]->[a1, a1, a3, a3]
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    //返回32位浮点数x[0]+x[1]+x[2]+x[3]
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    //_mm256_castps256_ps128(x)取256位向量的低128位，_mm256_extractf128_ps(x, 1)取256位向量的高128位
    //_mm_add_ps执行4对32位浮点数的加法
    //调用输入为128位的hsum,把128位向量中4个浮点数加起来
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

#if defined(__loongarch_asx)
// Convert __m256i low part to __m128i
// 取256位向量in的低128位到out里
static inline __m128i lasx_extracti128_lo(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".ifnc %[out], %[in]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    vori.b $vr\\i, $vr\\j, 0        \n\t" //低128位和0或
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}
// Convert __m256i high part to __m128i
// 取256位向量in的高128位到out
static inline __m128i lasx_extracti128_hi(__m256i in) {
    __m128i out;
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " VREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[in], " XREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x11  \n\t"
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        : [out] "=f" (out) : [in] "f" (in)
    );
    return out;
}

static __m128 lasx_extractf128( __m256 a, int pos) {
    __m128 ret;
    if( pos == 0)
    {
    ret = (__m128)lasx_extracti128_lo((__m256i)a);
    } else {
    ret = (__m128)lasx_extracti128_hi((__m256i)a);
    }
    return ret;
}
//8个float相加
static inline float hsum_float_8(const __m256 x) {
    //lasx_extractf128(x, 1)取x的高128位
    //lasx_extractf128(x, 0)取x的低128位
    __m128 res = lasx_extractf128(x, 1);
    ft_union tmp;
    //2个128位向量中的4个32位单精度浮点数相加
    res = __lsx_vfadd_s(res, lasx_extractf128(x, 0));
    //__lsx_vpickod_d拼接a的高64位和b的高64位（b在低位）
    //[res[0],res[1],res[2],res[3]] + [res[2],res[3],res[2],res[3]] = [res[0]+res[2],res[1]+res[3],res[2]*2,res[3]*2]
    res = __lsx_vfadd_s(res, (__m128)__lsx_vpickod_d((__m128i)res, (__m128i)res));
    //__lsx_vpickve2gr_w(res, 1)选择res中index为1的元素，即res[1]+res[3]
    //__lsx_vldi(0)返回一个全0的128位向量
    //__lsx_vinsgr2vr_w把返回值的第0个元素设置为__lsx_vpickve2gr_w(res, 1)的返回值：res[1]+res[3]，其他设为全0
    //__lsx_vfadd_s的结果为[res[0]+res[2]+res[1]+res[3],...]
    res = __lsx_vfadd_s(res, (__m128)__lsx_vinsgr2vr_w(__lsx_vldi(0), __lsx_vpickve2gr_w(res, 1), 0));
    //__lsx_vpickve2gr_w(res, 0)选出res中index为0的元素，即res[0]+res[2]+res[1]+res[3]
    tmp.i = __lsx_vpickve2gr_w(res, 0);
    return tmp.f;
}

inline float hsum(__m256 x) {
    return hsum_float_8(x);
}
#endif // __loongarch_asx

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(const U *);

#if defined(__ARM_NEON)
template <> inline float32x4_t load(const float *p) {
    return vld1q_f32(p);
}
#if !defined(_MSC_VER)
template <> inline float16x8_t load(const ggml_fp16_t *p) {
    return vld1q_f16((const float16_t *)p);
}
template <> inline float32x4_t load(const ggml_fp16_t *p) {
    return vcvt_f32_f16(vld1_f16((const float16_t *)p));
}
#endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m128 load(const float *p) {
    //从指定的内存地址加载 128 位（即 4 个 32 位浮点数）的数据，并将其存储在一个 128 位的寄存器中
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const float *p) {
    //从指定的内存地址加载 256 位（即 8 个 32 位浮点数）的数据，并将其存储在一个 256 位的寄存器中
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__loongarch_asx)
template <> inline __m256 load(const float *p) {
    //从指定的内存地址加载 256(32*8) 位的数据，并将其存储在一个 256 位的寄存器中
    return (__m256)__lasx_xvld(p, 0);
}
    // Convert two __m128i to __m256i
static inline __m256i lasx_set_q(__m128i inhi, __m128i inlo) {
    __m256i out;
    //irp用于迭代寄存器，这里的i是0~31
    //ifc是如果后面的两个操作数相等则执行，ifnc是不相等则执行
    //0x20 = 0b 0010 0000
    __asm__ volatile (
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[hi], " VREGS_PREFIX "\\i    \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[lo], " VREGS_PREFIX "\\j  \n\t"
        "    xvpermi.q $xr\\i, $xr\\j, 0x20  \n\t"//拼接hi和lo
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".ifnc %[out], %[hi]                 \n\t"
        ".irp i," __ALL_REGS                "\n\t"
        " .ifc %[out], " XREGS_PREFIX "\\i   \n\t"
        "  .irp j," __ALL_REGS              "\n\t"
        "   .ifc %[hi], " VREGS_PREFIX "\\j  \n\t"
        "    xvori.b $xr\\i, $xr\\j, 0       \n\t"//复制hi到out
        "   .endif                           \n\t"
        "  .endr                             \n\t"
        " .endif                             \n\t"
        ".endr                               \n\t"
        ".endif                              \n\t"
        : [out] "=f" (out), [hi] "+f" (inhi)
        : [lo] "f" (inlo)
    );
    return out;
}
//和ggml.c里的__lasx_f32cx8_load功能一样？但是不用再访存了
template <> inline __m256 load(const ggml_fp16_t *p) {
    //load出来8个16位浮点
    __m128i vector16 = __lsx_vld((const __m128i *)p, 0);
    //高位的4个16位浮点转为32位浮点
    __m128i hi = (__m128i)__lsx_vfcvth_s_h(vector16);
    //低位的4个16位浮点转为32位浮点
    __m128i lo = (__m128i)__lsx_vfcvtl_s_h(vector16);
    //合并成包含8个32位浮点的256位向量
    return (__m256)lasx_set_q(hi,lo);

}
#endif // __loongarch_asx

#if defined(__F16C__)
template <> inline __m256 load(const ggml_fp16_t *p) {
    //_mm_loadu_si128从内存加载128位数据
    //_mm256_cvtph_ps将包含 8 个半精度浮点数（16 位）的 128 位寄存器转换为包含 8 个单精度浮点数（32 位）的 256 位寄存器
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <> inline __m512 load(const ggml_fp16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION
//KN是k的步长，D是Cv的类型
template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc,
             int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 5) << 4) | MIN(n - n0, 5)) {
#if VECTOR_REGISTERS == 32
        case 0x55:
            mc = 5;
            nc = 5;
            gemm<5, 5>(m0, m, n0, n);
            break;
        case 0x45:
            mc = 4;
            nc = 5;
            gemm<4, 5>(m0, m, n0, n);
            break;
        case 0x54:
            mc = 5;
            nc = 4;
            gemm<5, 4>(m0, m, n0, n);
            break;
        case 0x44:
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
            break;
        case 0x53:
            mc = 5;
            nc = 3;
            gemm<5, 3>(m0, m, n0, n);
            break;
        case 0x35:
            mc = 3;
            nc = 5;
            gemm<3, 5>(m0, m, n0, n);
            break;
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
#else
        case 0x55:
        case 0x54:
        case 0x53:
        case 0x45:
        case 0x44:
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
        case 0x35:
#endif
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n);
            break;
        case 0x52:
            mc = 5;
            nc = 2;
            gemm<5, 2>(m0, m, n0, n);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x25:
            mc = 2;
            nc = 5;
            gemm<2, 5>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x51:
            mc = 5;
            nc = 1;
            gemm<5, 1>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x15:
            mc = 1;
            nc = 5;
            gemm<1, 5>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            D Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; l += KN)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i)
                        //在loongarch中：load的元素为ggml_fp16_t --> vld vfcvth.s.h vfcvtl.s.h xvpermi.q xvori.b
                        //              load的元素为flaot --> xvld
                        //madd对应 xvfmadd.s，结果为8个32位浮点数
                        //prompt中：gemm<5,5>中i循环j不变时，B本可以复用，但是并没有。。。decode阶段有优化
                        //prompt中：gemm<5,5>中Cv[j][i]要先xvld然后xvst存回内存，而不是保留在寄存器里。。。decode阶段有优化
                        //gemm<5,5>本可以展成25个计算，A也可以复用，但是只展成5了。。。
                        Cv[j][i] = madd(load<V>(A + lda * (ii + i) + l),
                                        load<V>(B + ldb * (jj + j) + l),
                                        Cv[j][i]);
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
template <typename TA>
class tinyBLAS_Q0_ARM {
  public:
    tinyBLAS_Q0_ARM(int64_t k,
                    const TA *A, int64_t lda,
                    const block_q8_0 *B, int64_t ldb,
                    float *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    NOINLINE void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 3) << 4) | MIN(n - n0, 3ll)) {
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            float32x4_t Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i)
                        //vdupq_n_s32将一个 32 位有符号整数值广播到128位向量寄存器中的每个元素
                        //vdotq_s32(a,b,c)计算b和c中对应位置8位有符号整数的乘积，每4对加到一起，得到4个32位结果，和a中的4个32位有符号数累加
                        //vcvtq_f32_s32将 128 位向量中的 32 位有符号整数元素转换为 32 位单精度浮点数元素
                        //vmlaq_n_f32(a,b,c)将一个 128 位向量b中的每个 32 位浮点数元素与标量 32 位浮点数c相乘，将乘积加到另一个 128 位向量a的相应元素上
                        Cv[j][i] = vmlaq_n_f32(Cv[j][i],
                                               vcvtq_f32_s32(vdotq_s32(
                                                   vdotq_s32(vdupq_n_s32(0),
                                                             load_lo(A + lda * (ii + i) + l),
                                                             load_lo(B + ldb * (jj + j) + l)),
                                                   load_hi(A + lda * (ii + i) + l),
                                                   load_hi(B + ldb * (jj + j) + l))),
                                               unhalf(A[lda * (ii + i) + l].d) *
                                               unhalf(B[ldb * (jj + j) + l].d));
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    //hsum把128位向量中所有32位浮点数加起来 
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline int8x16_t load_lo(const block_q8_0 *b) {
        //从内存地址b加载16个8位有符号整数（int8_t）的数据
        return vld1q_s8(b->qs);
    }

    inline int8x16_t load_hi(const block_q8_0 *b) {
        //从内存地址b+16*sizeof(block_q8_0)加载16个8位有符号整数（int8_t）的数据
        return vld1q_s8(b->qs + 16);
    }

    inline int8x16_t load_lo(const block_q4_0 *b) {
        //vld1q_u8从b->qs加载16个无符号整数
        //vreinterpretq_s8_u8把一个128位的无符号8位整数向量重新解释为有符号8位整数向量
        //vandq_u8对两个128位向量中的每个8位无符号整数元素执行按位与操作。
        //vdupq_n_s8将8位有符号整数8广播到128位向量寄存器中的每个元素
        //vsubq_s8对两个128位向量中的每个8位有符号整数元素执行逐元素减法操作
        return vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(b->qs),
                                                     vdupq_n_u8(0x0f))),
                        vdupq_n_s8(0x8));
    }

    inline int8x16_t load_hi(const block_q4_0 *b) {
        //vld1q_u8从b->qs加载16个无符号整数
        //vshrq_n_u8对128位向量中的每个 8 位无符号整数元素逻辑右移4位
        //vreinterpretq_s8_u8把一个128位的无符号8位整数向量重新解释为有符号8位整数向量
        //vdupq_n_s8将8位有符号整数8广播到128位向量寄存器中的每个元素
        //vsubq_s8对两个128位向量中的每个8位有符号整数元素执行逐元素减法操作
        return vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(b->qs), 4)),
                        vdupq_n_s8(0x8));
    }

    const TA *const A;
    const block_q8_0 *const B;
    float *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX {
  public:
    tinyBLAS_Q0_AVX(int64_t k,
                    const TA *A, int64_t lda,
                    const TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    //m0=0,n0=0,m和n分别是两个矩阵没有重合的维度
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        //最大处理4*4的块？
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 1)) {
#if VECTOR_REGISTERS == 32
        case 0x44:
            mc = 4;
            nc = 4;
            gemm<4, 4>(m0, m, n0, n);
            break;
        case 0x43:
            mc = 4;
            nc = 3;
            gemm<4, 3>(m0, m, n0, n);
            break;
        case 0x34:
            mc = 3;
            nc = 4;
            gemm<3, 4>(m0, m, n0, n);
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
#else
        case 0x44:
        case 0x43:
        case 0x42:
            mc = 4;
            nc = 2;
            gemm<4, 2>(m0, m, n0, n);
            break;
        case 0x34:
        case 0x24:
            mc = 2;
            nc = 4;
            gemm<2, 4>(m0, m, n0, n);
            break;
        case 0x33:
#endif
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }


    template <int RM, int RN>
    //一个RM*RN的tile中，最后的结果是交替计算的（对于第number个block，分别计算一遍tile中RM*RN个元素的结果，然后再到下一个tile）
    //一个线程计算的tile是连续的，而原始方法中一个线程计算的chunk是分开的
    //其他和原始方法一模一样
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        //tile是分块的大小，边长为xtiles和ytiles
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;//1 * 1024
        //duty是一个线程要处理的元素的数量，从start开始，到end
        int64_t duty = (tiles + nth - 1) / nth;//nth=1,duty=tiles=1024
        int64_t start = duty * ith; //start=0
        int64_t end = start + duty; //end=1024
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            //ii和jj是这个tile在m和n的维度上实际是从哪个元素开始的
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            //初始化一个包含2*4=8个256位向量的数组
            __m256 Cv[RN][RM] = {};
            //k=4096/32=128
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i) {
#if defined(__AVX2__)
                    //_mm256_sign_epi8(psignb)：对于每一对输入向量的元素，如果第二个向量的元素是正数，保持第一个向量的对应元素不变；
                    //如果第二个向量的元素是负数，将第一个向量的对应元素取相反数；如果第二个向量的元素是零，将第一个向量的对应元素变为零
                    //load函数从内存地址中加载数据，如果数据类型是block_q8_0，则直接加载一个block中量化后的256(8*32)位数据到向量中
                    //如果是block_q4_0，则将128位量化后数据放到256位向量中，且按照[0,1,2......31] -> [0,2,4,...30,1,3,5,...31]调整顺序
                    //undot将两个输入的对应位相乘，相邻4个相加，得到8个32位数
                    //在gemm<4,2>中，当i在0和RM之间循环的时候，load(B)的操作只用做一次；此外，当不同的j循环到同一个i时,sign(A,A)的结果可以重复利用
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#else
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        const __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#endif
                        //unhalf调用GGML_FP16_TO_FP32，把A和B的scale转为32位
                        //_mm256_set1_ps将指定的单精度浮点数广播到 256 位的 AVX 寄存器的每个 32 位浮点数位置
                        //madd实现a[i]*b[i]+c[i](vfmadd)
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline __m256i load(const block_q8_0 *b) {
        //_mm256_loadu_si256从内存加载256位数据到向量寄存器(vmovdqu)
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m128i load0(const block_q8_0 *b) {
        return _mm_loadu_si128((const __m128i *)b->qs);
    }

    inline __m128i load1(const block_q8_0 *b) {
        return _mm_loadu_si128(((const __m128i *)b->qs) + 1);
    }

    inline __m256i load(const block_q4_0 *b) {
        //_mm256_set1_epi8(8)创建包含32个8位整数8的向量
        //denibble把128位数据放到256位向量里：[0,1,2......31] -> [0,2,4,...30,1,3,5,...31]
        //_mm256_sub_epi8给每个8位整数减8
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m128i load0(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    inline __m128i load1(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        //一条指令实现u和s对应位置8位整数相乘，得到的相邻4个结果相加
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        //_mm256_set1_epi16(1)把16位整数1广播到256位寄存器的每个16位
        //_mm256_maddubs_epi16(u, s)得到16个16位数：[u[0]*s[0]+u[1]*s[1], u[2]*s[2]+u[3]*s[3],...u[30]*s[30]+u[31]*s[31]]
        //_mm256_madd_epi16得到8个32位数：[a[0]*b[0]+a[1]*b[1],...,a[14]*b[14]+a[15]*b[15]]
        //相当于是_mm256_maddubs_epi16(u, s)得到的16个16位数相邻2个相加
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        //将寄存器中的 8 个 32 位整数转换为 8 个单精度浮点数
        return _mm256_cvtepi32_ps(res);
    }

    static inline __m256i denibble(const uint8_t *p) {
        //_mm_loadu_si128从指定的内存位置加载128位数据到x中(movdqu)
        __m128i x = _mm_loadu_si128((const __m128i *)p);
        //_mm256_set1_epi8(15)将256位向量中32个8位整数都设为15, 15=0b00001111
        //_mm256_castsi128_si256(x)将x设为一个256位向量的低128位
        //_mm_srli_epi16(x, 4)将128位向量中每个16位元素右移4位
        //_mm256_insertf128_si256将_mm_srli_epi16(x, 4)插入_mm256_castsi128_si256(x)生成的256位向量高位
        //_mm256_and_si256对两个256位向量执行按位与(vpand)
        //最后得到的是[0,1,2......31] -> [0,2,4,...30,1,3,5,...31]
        return _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif // __AVX__

#if defined(__loongarch_asx)
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_LOONGARCH {
  public:
    tinyBLAS_Q0_LOONGARCH(int64_t k,
                    const TA *A, int64_t lda,
                    const TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
    }

    void matmul(int64_t m, int64_t n, int task) {
        if (task == GGML_TASK_TYPE_COMPUTE)
            mnpack(0, m, 0, n);
    }

  private:
    //m0=0,n0=0,m和n分别是两个矩阵没有重合的维度
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
        case 0x44:
        case 0x43:
        case 0x42:
        case 0x41:
            mc = 4;
            nc = 1;
            gemm<4, 1>(m0, m, n0, n);
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
            gemm<1, 4>(m0, m, n0, n);
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    inline __m256i load(const block_q8_0 *b) {
        //__lasx_xvld从内存地址b+0处加载256位数据到向量寄存器
        return __lasx_xvld(((const __m256i *)b->qs), 0);
    }

    static __m256i lasx_insertf128( __m128i x, __m128i y) {
        //xvpermi.q 
        //xvori.b
        return lasx_set_q(x, y);
    }

    // Unpack 32 4-bit fields into 32 bytes
    // The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
    static inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
        //从内存地址(rsi+0) load一个128位的向量，128=32 * 4 //vld
        const __m128i lo = __lsx_vld((const __m128i *)rsi, 0);
        //128位向量中每16位向量右移4位后得到hi //vsrli
        __m128i hi = __lsx_vsrli_h(lo, 4);
        //hi和lo拼成256位，得到的数的32个字节均与0xf按位与
        //最后128位向量中每个4位元素扩展为8位[0,1,2...31] -> [0,2,...30,1,3,...31]
        //xvpermi.q  xvori.b  xvandi.b
        return __lasx_xvandi_b(lasx_insertf128(hi, lo), 0xf);
    }

    inline __m256i load(const block_q4_0 *b) {
        // __lasx_xvreplgr2vr_b(8)创建包含32个8位整数8的向量 //xvreplgr2vr.b
        //bytes_from_nibbles_32把128位数据放到256位向量里：[0,1,2......31] -> [0,2,4,...30,1,3,5,...31]
        //__lasx_xvsub_b给每个8位整数减8
        return __lasx_xvsub_b(bytes_from_nibbles_32(b->qs), __lasx_xvreplgr2vr_b(8));
    }

    // add int16_t pairwise and return as float vector
    static inline __m256 sum_i16_pairs_float(const __m256i x) {
        //[a[1],b[1],a[3],b[3],...,a[15],b[15]]
        //[x[1],x[1],x[3],x[3],...,x[15],x[15]]
        __m256i v = __lasx_xvpackod_h(x, x);
        //偶数位的元素相加，放到summed_pairs里
        //[x[0]+x[1],x[2]+x[3],...,x[14]+x[15]]
        __m256i summed_pairs = __lasx_xvaddwev_w_h(x, v);
        //32位int转换为浮点数
        return __lasx_xvffint_s_w(summed_pairs);
    }

    static __m256i lasx_maddubs_h(__m256i a, __m256i b) {
        __m256i tmp1, tmp2;
        //a和b中偶数位的8位元素相乘，得到16个16位结果，放到tmp1里
        tmp1 = __lasx_xvmulwev_h_b(a, b);
        //a和b中奇数位的8位元素相乘，得到16个16位结果，放到tmp2里
        tmp2 = __lasx_xvmulwod_h_b(a, b);
        //16位加法
        return __lasx_xvsadd_h(tmp1, tmp2);
    }
    static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
        // Perform multiplication and create 16-bit values
        // lasx中没有指令能够直接实现和avx中_mm256_maddubs_epi16相同的功能，所以写函数lasx_maddubs_h模拟
        // ax[0]*sy[0]+ax[1]*sy[1]=dot[0];dot中有16个16位结果
        // xvmulwev.h.b  xvmulwod.h.b  xvsadd.h
        const __m256i dot = lasx_maddubs_h(ax, sy);
        // 相邻两数相加，得到8个32位结果
        // xvpackod.h  xvaddwev.w.h  xvffint.s.w
        return sum_i16_pairs_float(dot);
    }

    // multiply int8_t, add results pairwise twice and return as float vector
    static inline __m256 mul_sum_i8_pairs_float(const __m256i x, const __m256i y) {

        // Get absolute values of x vectors
        //__lasx_xvsigncov_b(a, b)如果a为0，则结果为0；如果a为正数，则结果为b；如果a为负数，则结果为-b；
        //如果a=b，则相当于取a的绝对值
        const __m256i ax = __lasx_xvsigncov_b(x, x);
        // Sign the values of the y vectors
        const __m256i sy = __lasx_xvsigncov_b(x, y);

        return mul_sum_us8_pairs_float(ax, sy);
    }


    template <int RM, int RN>
    //一个RM*RN的tile中，最后的结果是交替计算的（对于第number个block，分别计算一遍tile中RM*RN个元素的结果，然后再到下一个tile）
    //一个线程计算的tile是连续的，而原始方法中一个线程计算的chunk是分开的
    //其他和原始方法一模一样
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        //tile是分块的大小，边长为xtiles和ytiles
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;//1 * 1024
        //duty是一个线程要处理的元素的数量，从start开始，到end
        int64_t duty = (tiles + nth - 1) / nth;//nth=1,duty=tiles=1024
        int64_t start = duty * ith; //start=0
        int64_t end = start + duty; //end=1024
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            //ii和jj是这个tile在m和n的维度上实际是从哪个元素开始的
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            //初始化一个包含2*4=8个256位向量的数组
            __m256 Cv[RN][RM] = {};
            //k=4096/32=128
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i) {

                    //load函数从内存地址中加载数据，如果数据类型是block_q8_0，则直接加载一个block中量化后的256(8*32)位数据到向量中
                    //如果是block_q4_0，则将128位量化后数据放到256位向量中，且按照[0,1,2......31] -> [0,2,4,...30,1,3,5,...31]调整顺序
                    //mul_sum_i8_pairs_float将两个输入的对应位相乘，相邻4个相加，得到8个32位数

                    //B中是q8_0,使用vld从内存加载；A是q4_0，使用xvld从内存加载
                    //对于gemm<4,1>(即decode阶段），B中load的数据可以在和A的其他列相乘时重复利用
                    //此外，mul_sum_i8_pairs_float中的signcov(B,B)，也可以在i循环时，重复利用计算好的结果
                    //但是gemm<4,4>，gemm<4,2>，gemm<4,3>就没这两个优化
                        __m256 udTmp = mul_sum_i8_pairs_float(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l));

                        //unhalf调用GGML_FP16_TO_FP32，把A和B的scale转为32位
                        //unhalf(A)*unhalf(B)对应fld.s fld.s fmul.s，在j不变i循环时，可以节省一次fld.s的操作
                        //__lasx_xvreplfr2vr_s将指定的单精度浮点数广播到256位向量寄存器的每个32位浮点数 // movfr2gr.s  xvreplgr2vr.w
                        //__lasx_xvfmadd_s实现a[i]*b[i]+c[i]  // xvfmadd.s
                        Cv[j][i] = __lasx_xvfmadd_s(__lasx_xvreplfr2vr_s(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum_float_8(Cv[j][i]);
        }
    }
    
    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
};
#endif // __loongarch_asx

} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1, GGML_TASK_TYPE_COMPUTE,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(int64_t m, int64_t n, int64_t k, const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int ith, int nth, int task, int Atype, int Btype, int Ctype) {
    //A[m,k] * B[k,n] = C[m,n]
    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(nth > 0);
    assert(ith < nth);

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
#if defined(__AVX512F__)
        if (k % 16)
            return false;
        tinyBLAS<16, __m512, __m512, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__AVX__) || defined(__AVX2__)
        if (k % 8)
            return false;
        tinyBLAS<8, __m256, __m256, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__loongarch_asx)
        if (k % 8)
            return false;
        tinyBLAS<8, __m256, __m256, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_NEON)
        if (n < 4)
            return false;
        if (k % 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_F16: {
#if defined(__AVX512F__)
        if (k % 16)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<16, __m512, __m512, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<8, __m256, __m256, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__loongarch_asx)
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<8, __m256, __m256, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
        if (n < 8)
            return false;
        if (k % 8)
            return false;
        if (Btype != GGML_TYPE_F16)
            return false;
        tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const ggml_fp16_t *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_NEON) && !defined(_MSC_VER)
        if (k % 4)
            return false;
        if (Btype != GGML_TYPE_F32)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float> tb{
            k, (const ggml_fp16_t *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__loongarch_asx)
        tinyBLAS_Q0_LOONGARCH<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q8_0> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__ARM_FEATURE_DOTPROD)
        tinyBLAS_Q0_ARM<block_q4_0> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#elif defined(__loongarch_asx)
        tinyBLAS_Q0_LOONGARCH<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            ith, nth};
        tb.matmul(m, n, task);
        return true;
#else
        return false;
#endif
    }

    default:
        return false;
    }

    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)ith;
    (void)nth;
    (void)task;
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}
