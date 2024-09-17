// SPDX-License-Identifier: Apache-2.0

#ifndef SHUFFLE_ARITHMETIC_64_H
#define SHUFFLE_ARITHMETIC_64_H

#include <stdint.h>
#include <mayo.h>
#include <immintrin.h>
#include "./arithmetic_common.h"
#include "./arithmetic_64.h"

// P1*0 -> P1: v x v, O: v x o
static 
inline void mayo_12_P1_times_O_avx2(const uint64_t *_P1, __m256i *O_multabs, uint64_t *_acc){

    const __m256i *P1 = (__m256i *) _P1;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    size_t cols_used = 0;
    for (size_t r = 0; r < V_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[O_MAX] = {0};
        for (size_t c = r; c < V_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1 + cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            cols_used ++;

            for (size_t k = 0; k < O_MAX; k+=2)
            {
                temp[k]     ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_odd);
                temp[k + 1] ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (size_t k = 0; k < O_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(r*O_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[(r*O_MAX) + k + 1] ^= temp[k+1] ^ t;
        }
    }
}


static 
inline void mayo_12_Ot_times_P1O_P2_avx2(const uint64_t *_P1O_P2, __m256i *O_multabs, uint64_t *_acc){

    const __m256i *P1O_P2 = (__m256i *) _P1O_P2;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    for (size_t c = 0; c < O_MAX; c++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[O_MAX] = {0};
        for (size_t r = 0; r < V_MAX; r++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1O_P2 + r*O_MAX + c);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (size_t k = 0; k < O_MAX; k+=2)
            {
                temp[k]     ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*r + k/2], in_odd);
                temp[k + 1] ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*r + k/2], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (size_t k = 0; k < O_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(k*O_MAX) + c]     ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[((k+1)*O_MAX) + c] ^= temp[k+1] ^ t;
        }
    }
}


static
inline void mayo_12_P1P1t_times_O(const uint64_t *_P1, const unsigned char *O, uint64_t *_acc){

    const __m256i *P1 = (__m256i *) _P1;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    __m256i O_multabs[O_MAX/2*V_MAX];
    mayo_O_multabs_avx2(O, O_multabs);

    size_t cols_used = 0;
    for (size_t r = 0; r < V_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[O_MAX] = {0};
        cols_used += 1;
        size_t pos = r;
        for (size_t c = 0; c < r; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1 + pos);
            pos += (V_MAX -c - 1);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (size_t k = 0; k < O_MAX; k+=2)
            {
                temp[k]     ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_odd);
                temp[k + 1] ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_even);
            }            
        }

        for (size_t c = r+1; c < V_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1 + cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            cols_used ++;

            for (size_t k = 0; k < O_MAX; k+=2)
            {
                temp[k]     ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_odd);
                temp[k + 1] ^= _mm256_shuffle_epi8(O_multabs[O_MAX/2*c + k/2], in_even);
            }            
        }

        for (size_t k = 0; k < O_MAX; k+=2)
        {
            __m256i acc0 = _mm256_loadu_si256(acc + (r*O_MAX + k    ));
            __m256i acc1 = _mm256_loadu_si256(acc + (r*O_MAX + k + 1));

            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;

            _mm256_storeu_si256(acc + (r*O_MAX + k    ), acc0 ^ temp[k  ] ^ _mm256_slli_epi16(t,4));
            _mm256_storeu_si256(acc + (r*O_MAX + k + 1), acc1 ^ temp[k+1] ^ t);
        }
    }
}


static inline __m256i gf16_hadd_avx2_64(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_si256(in, 8));
    ret = _mm256_xor_si256(ret, _mm256_permute2x128_si256(ret, ret, 129)); // 0b10000001
    return ret;
}
/// horizontal xor
static inline __m256i gf16_hadd_avx2_32(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_epi64(in, 32));
    return gf16_hadd_avx2_64(ret);
}
/// horizontal xor
static inline __m256i gf16_hadd_avx2_16(const __m256i in) {
    __m256i ret = _mm256_xor_si256(in, _mm256_srli_epi32(in, 16));
    return gf16_hadd_avx2_32(ret);
}


static void mul_matrix(uint8_t *C, const uint8_t *A, const uint8_t *B,
         const uint32_t r, const uint32_t c1, const uint32_t c2) {

}

// only for Mayo2
// L is a row major matrix
static
inline void mayo_12_Vt_times_L_avx2_v2(const uint64_t *_L,
                                       const uint8_t *V,
                                       uint64_t *_acc){

    const __m256i mul_mask = _mm256_set1_epi32(0x11111111);
    // const uint32_t oA = K_OVER_2;
    // const __m256i scatter_mask = _mm256_setr_epi32(0, oA, 2*oA, 3*oA, 4*oA, 5*oA, 6*oA, 7*oA);

    // TODO: buffer overflow
    //__m256i v[8];
    //const __m256i lmask = _mm256_setr_epi32(0,0,0,0,-1,-1,-1,-1);
    //for (uint32_t i = 0; i < 7; ++i) {
    //    v[i] = _mm256_loadu_si256((__m256i *)(V + i*32));
    //}
    //v[7] = _mm256_maskload_epi32((int *)(V + 7*32), lmask);

    // const __m256i v1 = _mm256_loadu_si256((__m256i *)(V +  0));
    // const __m256i v2 = _mm256_loadu_si256((__m256i *)(V + 32));
    // const __m256i v3 = _mm256_loadu_si256((__m256i *)(V + 64));
    // const __m256i v4 = _mm256_loadu_si256((__m256i *)(V + 96));

    const __m128i b_mask = _mm_set1_epi8(0x0F);
    //const uint64_t b64_mask = 0x0F0F0F0F0F0F0F0F;
    //const uint64_t *l64 = (const uint64_t *)_L;
    //const __m128i *l128 = (const __m128i *)_L;
    const uint8_t *l8 = (const uint8_t *)_L;
    uint16_t *acc16 = (uint16_t *)_acc;
    //uint32_t *acc32 = (uint32_t *)_acc;
    const __m128i b_mask2 = _mm_set_epi64x(0x0000FFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF);
    __m256i s1;


    // TODO 2. for loop is missing and the transposing of M;
    for (uint32_t c = 0; c < V_MAX; c++) {
        // this code is correct if V is packed
        const uint16_t v = *(uint16_t *) (V + K_MAX * c);
        const __m256i v1 = _mm256_set1_epi16((short) v);
        {
            // code loads 32 fq elements (16 bytes)
            const __m128i b0 = *(const __m128i_u *) (l8 + c * O_MAX);
            const __m128i b1 = b0 & b_mask;
            const __m128i b2 = (b0 >> 4u) & b_mask;

            const __m256i c1 = _mm256_cvtepu8_epi16(b1);
            const __m256i c2 = _mm256_cvtepu8_epi16(b2);

            const __m256i d1 = _mm256_mullo_epi16(c1, mul_mask);
            const __m256i d2 = _mm256_mullo_epi16(c2, mul_mask);

            const __m256i e1 = _mm256_permute4x64_epi64(d1, 0b11011000);
            const __m256i e2 = _mm256_permute4x64_epi64(d2, 0b11011000);

            // f1 and f2, each contain 8 fq elements each placed in a 4 byte limb
            const __m256i f1 = _mm256_unpacklo_epi16(e1, e2);
            const __m256i f2 = _mm256_unpackhi_epi16(e1, e2);

            const __m256i g1 = mul_simd_u256(v1, f1);
            const __m256i g2 = mul_simd_u256(v1, f2);

            s1 = g1 ^ g2;
        }

        const __m256i s2 = gf16_hadd_avx2_16(s1);
        uint16_t r = _mm256_extract_epi16(s2, 0);

        // tail mngt
        for (uint32_t i = 16; i < O_MAX; ++i) {
            const uint8_t t1 = l8[c * O_MAX + i] & 0xF;
            r ^= gf16v_mul_u64(v, t1);
            const uint8_t t2 = l8[c * O_MAX + i] >> 4u;
            r ^= gf16v_mul_u64(v, t2);
        }
        acc16[c] ^= r;
    }


        // code loads 32 fq elements (16 bytes)
        //const __m128i b0 = l128[c];
        //const __m128i b1 = b0 & b_mask;
        //const __m128i b2 = (b0 >> 4u) & b_mask;

        //const __m256i c1 = _mm256_cvtepu8_epi16(b1);
        //const __m256i c2 = _mm256_cvtepu8_epi16(b2);

        //const __m256i d1 = _mm256_mullo_epi16(c1, mul_mask);
        //const __m256i d2 = _mm256_mullo_epi16(c2, mul_mask);

        //const __m256i e1 =_mm256_permute4x64_epi64(d1, 0b11011000);
        //const __m256i e2 =_mm256_permute4x64_epi64(d2, 0b11011000);

        //// f1 and f2, each contain 8 fq elements each placed in a 4 byte limb
        //const __m256i f1 = _mm256_unpacklo_epi16(e1, e2);
        //const __m256i f2 = _mm256_unpackhi_epi16(e1, e2);

        //const __m256i g1 = mul_simd_u256(f1, v1);
        //const __m256i g2 = mul_simd_u256(f2, v2);

        //const __m256i s1 = g1 ^ g2;
        //const __m256i s2 = gf16_hadd_avx2_16(s1);
        //const uint16_t r = _mm256_extract_epi16(s2, 0);
        //acc16[c] ^= r;


        // code to load 16 fq elements (8 byte) into 2 avx registers
        // padded by 4 bytes each.
        //const uint64_t b0 = v64[c + 0 ];
        //const __m128i b1 = _mm_set1_epi64x(b0 & 0x0F0F0F0F0F0F0F0F);
        //const __m128i b2 = _mm_set1_epi64x(b0 << 4u);
        // // expand 8 fq elements to avx register (32 limbs)
        // const __m256i c1 = _mm256_cvtepu8_epi32(b1);
        // const __m256i c2 = _mm256_cvtepu8_epi32(b2);

        // // expand each 4 bit fq element into 8 fq elements within
        // // each 32 bit limb
        // const __m256i d1 = _mm256_mullo_epi32(c1, mul_mask);
        // const __m256i d2 = _mm256_mullo_epi32(c2, mul_mask);

        // // swap the second and third 64 bit limb
        // const __m256i e1 =_mm256_permute4x64_epi64(d1, 0b11011000);
        // const __m256i e2 =_mm256_permute4x64_epi64(d2, 0b11011000);

        // // f1 and f2, each contain 8 fq elements each placed in a 4 byte limb
        // const __m256i f1 = _mm256_unpacklo_epi32(e1, e2);
        // const __m256i f2 = _mm256_unpackhi_epi32(e1, e2);
        // (void)f1;
        // (void)f2;



        //__m256i s1 = _mm256_setzero_si256();
        //for (uint32_t i = 0; i < 4; ++i) {
        //    // code to load 16 fq elements (8 byte) into 2 avx registers
        //    // padded by 4 bytes each.
        //    const uint64_t b0 = l64[c*4 + i];
        //    const __m128i b1 = _mm_set1_epi64x(b0 & b64_mask);
        //    const __m128i b2 = _mm_set1_epi64x((b0 >> 4u) & b64_mask);
        //    // expand 8 fq elements to avx register (32 limbs)
        //    const __m256i c1 = _mm256_cvtepu8_epi32(b1);
        //    const __m256i c2 = _mm256_cvtepu8_epi32(b2);

        //    // expand each 4 bit fq element into 8 fq elements within
        //    // each 32 bit limb
        //    const __m256i d1 = _mm256_mullo_epi32(c1, mul_mask);
        //    const __m256i d2 = _mm256_mullo_epi32(c2, mul_mask);

        //    // swap the second and third 64 bit limb
        //    const __m256i e1 =_mm256_permute4x64_epi64(d1, 0b11011000);
        //    const __m256i e2 =_mm256_permute4x64_epi64(d2, 0b11011000);

        //    // f1 and f2, each contain 8 fq elements each placed in a 4 byte limb
        //    const __m256i f1 = _mm256_unpacklo_epi32(e1, e2);
        //    const __m256i f2 = _mm256_unpackhi_epi32(e1, e2);

        //    const __m256i g1 = mul_simd_u256(f1, v[i*2 + 0]);
        //    const __m256i g2 = mul_simd_u256(f2, v[i*2 + 1]);

        //    s1 ^= g1 ^ g2;
        //}

        //const __m256i s2 = gf16_hadd_avx2_32(s1);
        //const uint32_t r1 = _mm256_extract_epi32(s2, 0);
        //const uint32_t r2 = r1 & b64_mask;
        //acc32[c] ^= r2;
}
static
inline void mayo_12_Vt_times_L_avx2(const uint64_t *_L, const __m256i *V_multabs, uint64_t *_acc){

    const __m256i *L = (__m256i *) _L;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);
    size_t k;

    for (size_t c = 0; c < O_MAX; c++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};
        for (size_t r = 0; r < V_MAX; r++)
        {
            __m256i in_odd = _mm256_loadu_si256(L + r*O_MAX + c);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*r + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*r + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k+1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(k*O_MAX) + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[((k+1)*O_MAX) + c] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[k*O_MAX + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}


static 
inline void mayo_12_Vt_times_Pv_avx2(const uint64_t *_Pv, const __m256i *V_multabs, uint64_t *_acc){

    const __m256i *Pv = (__m256i *) _Pv;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);
    size_t k;

    for (size_t c = 0; c < K_MAX; c++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};
        for (size_t r = 0; r < V_MAX; r++)
        {
            __m256i in_odd = _mm256_loadu_si256(Pv + r*K_MAX + c);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*r + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*r + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k+1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(k*K_MAX) + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[((k+1)*K_MAX) + c] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[k*K_MAX + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}

static 
inline void mayo_12_P1_times_Vt_avx2(const uint64_t *_P1, __m256i *V_multabs, uint64_t *_acc){
    size_t k,c;
    const __m256i *P1 = (__m256i *) _P1;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    size_t cols_used = 0;
    for (size_t r = 0; r < V_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};

        for (c=r; c < V_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1 + cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            cols_used ++;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*c + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(V_multabs[K_OVER_2*c + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k + 1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[(r*K_MAX) + k + 1] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}

// P1*S1 -> P1: v x v, S1: v x k // P1 upper triangular
// same as mayo_12_P1_times_Vt_avx2
static
inline void mayo_12_P1_times_S1_avx2(const uint64_t *_P1, __m256i *S1_multabs, uint64_t *_acc){
    mayo_12_P1_times_Vt_avx2(_P1, S1_multabs, _acc);
}

static
inline void mayo_12_S1t_times_PS1_avx2(const uint64_t *_PS1, __m256i *S1_multabs, uint64_t *_acc){
    mayo_12_Vt_times_Pv_avx2(_PS1, S1_multabs, _acc);
}

static
inline void mayo_12_S2t_times_PS2_avx2(const uint64_t *_PS2, __m256i *S2_multabs, uint64_t *_acc){
    const __m256i *PS2 = (__m256i *) _PS2;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);
    size_t k;

    for (size_t c = 0; c < K_MAX; c++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};
        for (size_t r = 0; r < O_MAX; r++)
        {
            __m256i in_odd = _mm256_loadu_si256(PS2 + r*K_MAX + c);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*r + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*r + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k+1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(k*K_MAX) + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[((k+1)*K_MAX) + c] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[k*K_MAX + c] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}


// P2*S2 -> P2: v x o, S2: o x k
static 
inline void mayo_12_P2_times_S2_avx2(const uint64_t *_P2, __m256i *S2_multabs, uint64_t *_acc){
    size_t k,c;
    const __m256i *P2 = (__m256i *) _P2;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    size_t cols_used = 0;
    for (size_t r = 0; r < V_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};

        for (c=0; c < O_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P2 + cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            cols_used ++;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k + 1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[(r*K_MAX) + k + 1] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}


// P2*S2 -> P2: v x o, S2: o x k
static 
inline void mayo_12_P1_times_S1_plus_P2_times_S2_avx2(const uint64_t *_P1, const uint64_t *_P2, __m256i *S1_multabs, __m256i *S2_multabs, uint64_t *_acc){
    size_t k,c;
    const __m256i *P1 = (__m256i *) _P1;
    const __m256i *P2 = (__m256i *) _P2;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    size_t P1_cols_used = 0;
    for (size_t r = 0; r < V_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};


        // P1 * S1
        for (c=r; c < V_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P1 + P1_cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            P1_cols_used ++;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(S1_multabs[K_OVER_2*c + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(S1_multabs[K_OVER_2*c + k], in_even);
            }            
        }

        // P2 * S2
        for (c=0; c < O_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P2 + r*O_MAX + c);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k + 1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[(r*K_MAX) + k + 1] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}

// P3*S2 -> P3: o x o, S2: o x k // P3 upper triangular
static 
inline void mayo_12_P3_times_S2_avx2(const uint64_t *_P3, __m256i *S2_multabs, uint64_t *_acc){
    size_t k,c;
    const __m256i *P3 = (__m256i *) _P3;
    __m256i *acc = (__m256i *) _acc;
    const __m256i low_nibble_mask  = _mm256_set_epi64x(0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f, 0x0f0f0f0f0f0f0f0f);

    size_t cols_used = 0;
    for (size_t r = 0; r < O_MAX; r++)
    {
        // do multiplications for one row and accumulate results in temporary format
        __m256i temp[K_OVER_2*2] = {0};

        for (c=r; c < O_MAX; c++)
        {
            __m256i in_odd = _mm256_loadu_si256(P3 + cols_used);
            __m256i in_even = _mm256_srli_epi16(in_odd, 4) & low_nibble_mask;
            in_odd &= low_nibble_mask;
            cols_used ++;

            for (k = 0; k < K_OVER_2; k++)
            {
                temp[2*k]     ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_odd);
                temp[2*k + 1] ^= _mm256_shuffle_epi8(S2_multabs[K_OVER_2*c + k], in_even);
            }            
        }

        // convert to normal format and add to accumulator 
        for (k = 0; k + 1 < K_MAX; k+=2)
        {
            __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
            acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
            acc[(r*K_MAX) + k + 1] ^= temp[k+1] ^ t;
        }
#if K_MAX % 2 == 1
        __m256i t = (temp[k + 1] ^ _mm256_srli_epi16(temp[k],4)) & low_nibble_mask;
        acc[(r*K_MAX) + k] ^= temp[k] ^ _mm256_slli_epi16(t,4);
#endif
    }
}


static inline
void mayo12_m_upper(int m_legs, const uint64_t *in, uint64_t *out, int size) {
    (void) size;
    int m_vecs_stored = 0;

    for (int r = 0; r < O_MAX; ++r) {
        const __m256i* _in = (const __m256i*) (in + m_legs * 2 * (r * size + r));
        __m256i* _out = (__m256i*) (out + m_legs * 2 * m_vecs_stored);
        _out[0] = _in[0];
        m_vecs_stored++;
        for (int c = r + 1; c < O_MAX; ++c) {
            const __m256i* _in2 = (const __m256i*) (in + m_legs * 2 * (r * size + c));
            const __m256i* _in3 = (const __m256i*) (in + m_legs * 2 * (c * size + r));
            _out = (__m256i*) (out + m_legs * 2 * m_vecs_stored);
            _out[0] = _in2[0] ^ _in3[0];
            m_vecs_stored++;
        }
    }
}


#undef K_OVER_2
#endif

