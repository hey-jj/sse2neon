//! SSE3 intrinsics: horizontal add and subtract, add-subtract, duplicate.

use crate::sse::*;
use crate::types::*;
use core::arch::aarch64::*;

/// Horizontally add adjacent `f32` pairs from `a` then `b`. Matches `_mm_hadd_ps`.
#[inline]
pub fn _mm_hadd_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vpaddq_f32(a.f32(), b.f32()) })
}

/// Horizontally subtract adjacent `f32` pairs. Matches `_mm_hsub_ps`.
#[inline]
pub fn _mm_hsub_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // Deinterleave even and odd lanes across a and b, then subtract.
        let even = vuzp1q_f32(a.f32(), b.f32());
        let odd = vuzp2q_f32(a.f32(), b.f32());
        __m128::from_f32(vsubq_f32(even, odd))
    }
}

/// Add odd lanes and subtract even lanes: `a0-b0, a1+b1, a2-b2, a3+b3`.
/// Matches `_mm_addsub_ps`.
#[inline]
pub fn _mm_addsub_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // Negate the even lanes of b, then add.
        let mask = _mm_setr_ps(-0.0, 0.0, -0.0, 0.0);
        let flipped = veorq_u32(b.u32(), mask.u32());
        __m128::from_f32(vaddq_f32(a.f32(), vreinterpretq_f32_u32(flipped)))
    }
}

/// Horizontally add the two `f64` lanes of `a` then `b`. Matches `_mm_hadd_pd`.
#[inline]
pub fn _mm_hadd_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vpaddq_f64(a.f64(), b.f64()) })
}

/// Horizontally subtract the two `f64` lanes of `a` then `b`. Matches `_mm_hsub_pd`.
#[inline]
pub fn _mm_hsub_pd(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a0 = vgetq_lane_f64(a.f64(), 0);
        let a1 = vgetq_lane_f64(a.f64(), 1);
        let b0 = vgetq_lane_f64(b.f64(), 0);
        let b1 = vgetq_lane_f64(b.f64(), 1);
        crate::sse2::_mm_set_pd(b0 - b1, a0 - a1)
    }
}

/// Add odd lane, subtract even lane: `a0-b0, a1+b1`. Matches `_mm_addsub_pd`.
#[inline]
pub fn _mm_addsub_pd(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a0 = vgetq_lane_f64(a.f64(), 0);
        let a1 = vgetq_lane_f64(a.f64(), 1);
        let b0 = vgetq_lane_f64(b.f64(), 0);
        let b1 = vgetq_lane_f64(b.f64(), 1);
        crate::sse2::_mm_set_pd(a1 + b1, a0 - b0)
    }
}

/// Duplicate the low `f64` lane to both lanes. Matches `_mm_movedup_pd`.
#[inline]
pub fn _mm_movedup_pd(a: __m128d) -> __m128d {
    unsafe {
        let lo = vgetq_lane_f64(a.f64(), 0);
        __m128d::from_f64(vdupq_n_f64(lo))
    }
}

/// Duplicate the odd `f32` lanes: `a1, a1, a3, a3`. Matches `_mm_movehdup_ps`.
#[inline]
pub fn _mm_movehdup_ps(a: __m128) -> __m128 {
    __m128::from_f32(unsafe { vtrn2q_f32(a.f32(), a.f32()) })
}

/// Duplicate the even `f32` lanes: `a0, a0, a2, a2`. Matches `_mm_moveldup_ps`.
#[inline]
pub fn _mm_moveldup_ps(a: __m128) -> __m128 {
    __m128::from_f32(unsafe { vtrn1q_f32(a.f32(), a.f32()) })
}
