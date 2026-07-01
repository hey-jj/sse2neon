//! SSSE3 intrinsics: absolute value, sign, byte shuffle, align, horizontal ops.

use crate::sse2::*;
use crate::types::*;
use core::arch::aarch64::*;

/// Absolute value of sixteen `i8` lanes. Matches `_mm_abs_epi8`.
#[inline]
pub fn _mm_abs_epi8(a: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vreinterpretq_s8_u8(vreinterpretq_u8_s8(vabsq_s8(a.s8()))) })
}

/// Absolute value of eight `i16` lanes. Matches `_mm_abs_epi16`.
#[inline]
pub fn _mm_abs_epi16(a: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vabsq_s16(a.s16()) })
}

/// Absolute value of four `i32` lanes. Matches `_mm_abs_epi32`.
#[inline]
pub fn _mm_abs_epi32(a: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vabsq_s32(a.s32()) })
}

/// Shuffle bytes of `a` by the indices in `b`. Matches `_mm_shuffle_epi8`.
///
/// Each index byte selects a source lane by its low 4 bits. If the top bit of an
/// index byte is set, that output lane becomes zero.
#[inline]
pub fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        // Mask indices to 0..15 plus the "zero this lane" top bit. vqtbl1q
        // returns zero for any index >= 16, so setting bit 7 (via 0x80) forces a
        // zero without extra masking. Keep bits 0-3 and bit 7.
        let idx = vandq_u8(b.u8(), vdupq_n_u8(0x8f));
        __m128i::from_u8(vqtbl1q_u8(a.u8(), idx))
    }
}

/// Sign sixteen `i8` lanes of `a` by the sign of `b`. Matches `_mm_sign_epi8`.
///
/// Lane is negated where `b < 0`, zeroed where `b == 0`, kept where `b > 0`.
#[inline]
pub fn _mm_sign_epi8(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let bv = b.s8();
        let av = a.s8();
        let zero = vdupq_n_s8(0);
        let neg = vcltq_s8(bv, zero);
        let is_zero = vceqq_s8(bv, zero);
        let negated = vnegq_s8(av);
        // Choose negated where b<0, else a; then zero where b==0.
        let chosen = vbslq_s8(neg, negated, av);
        __m128i::from_s8(vbicq_s8(chosen, vreinterpretq_s8_u8(is_zero)))
    }
}

/// Sign eight `i16` lanes of `a` by the sign of `b`. Matches `_mm_sign_epi16`.
#[inline]
pub fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let bv = b.s16();
        let av = a.s16();
        let zero = vdupq_n_s16(0);
        let neg = vcltq_s16(bv, zero);
        let is_zero = vceqq_s16(bv, zero);
        let chosen = vbslq_s16(neg, vnegq_s16(av), av);
        __m128i::from_s16(vbicq_s16(chosen, vreinterpretq_s16_u16(is_zero)))
    }
}

/// Sign four `i32` lanes of `a` by the sign of `b`. Matches `_mm_sign_epi32`.
#[inline]
pub fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let bv = b.s32();
        let av = a.s32();
        let zero = vdupq_n_s32(0);
        let neg = vcltq_s32(bv, zero);
        let is_zero = vceqq_s32(bv, zero);
        let chosen = vbslq_s32(neg, vnegq_s32(av), av);
        __m128i::from_s32(vbicq_s32(chosen, vreinterpretq_s32_u32(is_zero)))
    }
}

/// Horizontally add adjacent `i16` pairs from `a` then `b`. Matches `_mm_hadd_epi16`.
#[inline]
pub fn _mm_hadd_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vpaddq_s16(a.s16(), b.s16()) })
}

/// Horizontally add adjacent `i32` pairs from `a` then `b`. Matches `_mm_hadd_epi32`.
#[inline]
pub fn _mm_hadd_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vpaddq_s32(a.s32(), b.s32()) })
}

/// Horizontally subtract adjacent `i16` pairs. Matches `_mm_hsub_epi16`.
#[inline]
pub fn _mm_hsub_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let even = vuzp1q_s16(a.s16(), b.s16());
        let odd = vuzp2q_s16(a.s16(), b.s16());
        __m128i::from_s16(vsubq_s16(even, odd))
    }
}

/// Horizontally subtract adjacent `i32` pairs. Matches `_mm_hsub_epi32`.
#[inline]
pub fn _mm_hsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let even = vuzp1q_s32(a.s32(), b.s32());
        let odd = vuzp2q_s32(a.s32(), b.s32());
        __m128i::from_s32(vsubq_s32(even, odd))
    }
}

/// Multiply unsigned `a` by signed `b` bytes, add adjacent pairs with signed
/// saturation into `i16`. Matches `_mm_maddubs_epi16`.
#[inline]
pub fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        // Widen a as unsigned and b as signed, multiply per byte lane.
        let a_u = a.u8();
        let b_s = b.s8();
        let al = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a_u)));
        let ah = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a_u)));
        let bl = vmovl_s8(vget_low_s8(b_s));
        let bh = vmovl_s8(vget_high_s8(b_s));
        let pl = vmulq_s16(al, bl);
        let ph = vmulq_s16(ah, bh);
        // Sum adjacent pairs with signed saturation, keeping a-order.
        let mut prod = [0i16; 16];
        vst1q_s16(prod.as_mut_ptr(), pl);
        vst1q_s16(prod.as_mut_ptr().add(8), ph);
        let mut out = [0i16; 8];
        for (i, o) in out.iter_mut().enumerate() {
            let sum = prod[2 * i] as i32 + prod[2 * i + 1] as i32;
            *o = sum.clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        }
        __m128i::from_s16(vld1q_s16(out.as_ptr()))
    }
}

/// Multiply eight `i16` pairs, keep bits 16-30, round. Matches `_mm_mulhrs_epi16`.
#[inline]
pub fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let al = vget_low_s16(a.s16());
        let ah = vget_high_s16(a.s16());
        let bl = vget_low_s16(b.s16());
        let bh = vget_high_s16(b.s16());
        // (a*b + 0x4000) >> 15
        let round = vdupq_n_s32(1 << 14);
        let pl = vshrq_n_s32::<15>(vaddq_s32(vmull_s16(al, bl), round));
        let ph = vshrq_n_s32::<15>(vaddq_s32(vmull_s16(ah, bh), round));
        __m128i::from_s16(vcombine_s16(vmovn_s32(pl), vmovn_s32(ph)))
    }
}

/// Concatenate `b:a`, shift right by `IMM` bytes, keep the low 16.
/// Matches `_mm_alignr_epi8`.
#[inline]
pub fn _mm_alignr_epi8<const IMM: i32>(a: __m128i, b: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if IMM >= 32 {
        return _mm_setzero_si128();
    }
    let mut cat = [0u8; 32];
    let bv = to_u8_array(b);
    let av = to_u8_array(a);
    cat[..16].copy_from_slice(&bv);
    cat[16..].copy_from_slice(&av);
    let mut out = [0u8; 16];
    for (i, o) in out.iter_mut().enumerate() {
        let src = i + IMM as usize;
        *o = if src < 32 { cat[src] } else { 0 };
    }
    __m128i::from_u8(unsafe { vld1q_u8(out.as_ptr()) })
}
