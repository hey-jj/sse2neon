//! SSE4.1 intrinsics: blend, rounding, sign and zero extension, dot product,
//! extract and insert, extended min and max, and the all-ones/zeros tests.

use crate::constants::*;
use crate::sse::_mm_move_ss;
use crate::sse2::*;
use crate::types::*;
use core::arch::aarch64::*;

// --- Blend ---

/// Blend eight `i16` lanes by the immediate mask. Matches `_mm_blend_epi16`.
///
/// Bit `n` of `IMM` selects lane `n` from `b`, else from `a`.
#[inline]
pub fn _mm_blend_epi16<const IMM: i32>(a: __m128i, b: __m128i) -> __m128i {
    let av = to_i16_array(a);
    let bv = to_i16_array(b);
    let mut out = [0i16; 8];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if (IMM >> i) & 1 == 1 { bv[i] } else { av[i] };
    }
    __m128i::from_s16(unsafe { vld1q_s16(out.as_ptr()) })
}

/// Blend two `f64` lanes by the immediate mask. Matches `_mm_blend_pd`.
#[inline]
pub fn _mm_blend_pd<const IMM: i32>(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a0 = vgetq_lane_f64(a.f64(), 0);
        let a1 = vgetq_lane_f64(a.f64(), 1);
        let b0 = vgetq_lane_f64(b.f64(), 0);
        let b1 = vgetq_lane_f64(b.f64(), 1);
        let e0 = if IMM & 1 == 1 { b0 } else { a0 };
        let e1 = if IMM & 2 == 2 { b1 } else { a1 };
        _mm_set_pd(e1, e0)
    }
}

/// Blend four `f32` lanes by the immediate mask. Matches `_mm_blend_ps`.
#[inline]
pub fn _mm_blend_ps<const IMM: i32>(a: __m128, b: __m128) -> __m128 {
    let mut av = [0.0f32; 4];
    let mut bv = [0.0f32; 4];
    unsafe {
        vst1q_f32(av.as_mut_ptr(), a.f32());
        vst1q_f32(bv.as_mut_ptr(), b.f32());
    }
    let mut out = [0.0f32; 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = if (IMM >> i) & 1 == 1 { bv[i] } else { av[i] };
    }
    __m128::from_f32(unsafe { vld1q_f32(out.as_ptr()) })
}

/// Blend sixteen `i8` lanes by the sign bit of each `mask` lane.
/// Matches `_mm_blendv_epi8`.
#[inline]
pub fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    unsafe {
        // Build an all-ones/zeros selector from each mask byte's sign bit.
        let sel = vcltq_s8(mask.s8(), vdupq_n_s8(0));
        __m128i::from_s8(vbslq_s8(sel, b.s8(), a.s8()))
    }
}

/// Blend four `f32` lanes by the sign bit of each `mask` lane. Matches `_mm_blendv_ps`.
#[inline]
pub fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) -> __m128 {
    unsafe {
        let sel = vcltq_s32(vreinterpretq_s32_f32(mask.f32()), vdupq_n_s32(0));
        __m128::from_f32(vbslq_f32(sel, b.f32(), a.f32()))
    }
}

/// Blend two `f64` lanes by the sign bit of each `mask` lane. Matches `_mm_blendv_pd`.
#[inline]
pub fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    unsafe {
        let sel = vcltq_s64(vreinterpretq_s64_f64(mask.f64()), vdupq_n_s64(0));
        __m128d::from_f64(vbslq_f64(sel, b.f64(), a.f64()))
    }
}

// --- Rounding ---

/// Round four `f32` lanes per the `ROUNDING` mode. Matches `_mm_round_ps`.
///
/// `ROUNDING` uses the `_MM_FROUND_*` constants. The exception bits are ignored.
#[inline]
pub fn _mm_round_ps<const ROUNDING: i32>(a: __m128) -> __m128 {
    let mode = ROUNDING & !(_MM_FROUND_RAISE_EXC | _MM_FROUND_NO_EXC);
    unsafe {
        let f = a.f32();
        let r = match mode {
            _MM_FROUND_TO_NEAREST_INT => vrndnq_f32(f),
            _MM_FROUND_TO_NEG_INF => vrndmq_f32(f),
            _MM_FROUND_TO_POS_INF => vrndpq_f32(f),
            _MM_FROUND_TO_ZERO => vrndq_f32(f),
            _ => vrndiq_f32(f),
        };
        __m128::from_f32(r)
    }
}

/// Round two `f64` lanes per the `ROUNDING` mode. Matches `_mm_round_pd`.
#[inline]
pub fn _mm_round_pd<const ROUNDING: i32>(a: __m128d) -> __m128d {
    let mode = ROUNDING & !(_MM_FROUND_RAISE_EXC | _MM_FROUND_NO_EXC);
    unsafe {
        let f = a.f64();
        let r = match mode {
            _MM_FROUND_TO_NEAREST_INT => vrndnq_f64(f),
            _MM_FROUND_TO_NEG_INF => vrndmq_f64(f),
            _MM_FROUND_TO_POS_INF => vrndpq_f64(f),
            _MM_FROUND_TO_ZERO => vrndq_f64(f),
            _ => vrndiq_f64(f),
        };
        __m128d::from_f64(r)
    }
}

/// Round four `f32` lanes toward negative infinity. Matches `_mm_floor_ps`.
#[inline]
pub fn _mm_floor_ps(a: __m128) -> __m128 {
    _mm_round_ps::<{ _MM_FROUND_TO_NEG_INF }>(a)
}

/// Round four `f32` lanes toward positive infinity. Matches `_mm_ceil_ps`.
#[inline]
pub fn _mm_ceil_ps(a: __m128) -> __m128 {
    _mm_round_ps::<{ _MM_FROUND_TO_POS_INF }>(a)
}

/// Round two `f64` lanes toward negative infinity. Matches `_mm_floor_pd`.
#[inline]
pub fn _mm_floor_pd(a: __m128d) -> __m128d {
    _mm_round_pd::<{ _MM_FROUND_TO_NEG_INF }>(a)
}

/// Round two `f64` lanes toward positive infinity. Matches `_mm_ceil_pd`.
#[inline]
pub fn _mm_ceil_pd(a: __m128d) -> __m128d {
    _mm_round_pd::<{ _MM_FROUND_TO_POS_INF }>(a)
}

/// Round lane 0 per `ROUNDING`, copy upper lanes from `a`. Matches `_mm_round_ss`.
#[inline]
pub fn _mm_round_ss<const ROUNDING: i32>(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_round_ps::<ROUNDING>(b))
}

/// Floor lane 0 of `b`, upper lanes from `a`. Matches `_mm_floor_ss`.
#[inline]
pub fn _mm_floor_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_floor_ps(b))
}

/// Ceil lane 0 of `b`, upper lanes from `a`. Matches `_mm_ceil_ss`.
#[inline]
pub fn _mm_ceil_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_ceil_ps(b))
}

/// Round lane 0 per `ROUNDING`, high lane from `a`. Matches `_mm_round_sd`.
#[inline]
pub fn _mm_round_sd<const ROUNDING: i32>(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_round_pd::<ROUNDING>(b))
}

/// Floor lane 0 of `b`, high lane from `a`. Matches `_mm_floor_sd`.
#[inline]
pub fn _mm_floor_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_floor_pd(b))
}

/// Ceil lane 0 of `b`, high lane from `a`. Matches `_mm_ceil_sd`.
#[inline]
pub fn _mm_ceil_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_ceil_pd(b))
}

// --- Sign / zero extension ---

/// Sign-extend the low eight `i8` to `i16`. Matches `_mm_cvtepi8_epi16`.
#[inline]
pub fn _mm_cvtepi8_epi16(a: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vmovl_s8(vget_low_s8(a.s8())) })
}

/// Sign-extend the low four `i8` to `i32`. Matches `_mm_cvtepi8_epi32`.
#[inline]
pub fn _mm_cvtepi8_epi32(a: __m128i) -> __m128i {
    unsafe {
        let w = vmovl_s8(vget_low_s8(a.s8()));
        __m128i::from_s32(vmovl_s16(vget_low_s16(w)))
    }
}

/// Sign-extend the low four `i16` to `i32`. Matches `_mm_cvtepi16_epi32`.
#[inline]
pub fn _mm_cvtepi16_epi32(a: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vmovl_s16(vget_low_s16(a.s16())) })
}

/// Sign-extend the low two `i32` to `i64`. Matches `_mm_cvtepi32_epi64`.
#[inline]
pub fn _mm_cvtepi32_epi64(a: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vmovl_s32(vget_low_s32(a.s32())) })
}

/// Zero-extend the low eight `u8` to `u16`. Matches `_mm_cvtepu8_epi16`.
#[inline]
pub fn _mm_cvtepu8_epi16(a: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vmovl_u8(vget_low_u8(a.u8())) })
}

/// Zero-extend the low four `u8` to `u32`. Matches `_mm_cvtepu8_epi32`.
#[inline]
pub fn _mm_cvtepu8_epi32(a: __m128i) -> __m128i {
    unsafe {
        let w = vmovl_u8(vget_low_u8(a.u8()));
        __m128i::from_u32(vmovl_u16(vget_low_u16(w)))
    }
}

/// Zero-extend the low four `u16` to `u32`. Matches `_mm_cvtepu16_epi32`.
#[inline]
pub fn _mm_cvtepu16_epi32(a: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vmovl_u16(vget_low_u16(a.u16())) })
}

/// Zero-extend the low two `u32` to `u64`. Matches `_mm_cvtepu32_epi64`.
#[inline]
pub fn _mm_cvtepu32_epi64(a: __m128i) -> __m128i {
    __m128i::from_u64(unsafe { vmovl_u32(vget_low_u32(a.u32())) })
}

// --- Min / max ---

/// Packed minimum of sixteen `i8` lanes. Matches `_mm_min_epi8`.
#[inline]
pub fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vminq_s8(a.s8(), b.s8()) })
}

/// Packed maximum of sixteen `i8` lanes. Matches `_mm_max_epi8`.
#[inline]
pub fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vmaxq_s8(a.s8(), b.s8()) })
}

/// Packed minimum of four `i32` lanes. Matches `_mm_min_epi32`.
#[inline]
pub fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vminq_s32(a.s32(), b.s32()) })
}

/// Packed maximum of four `i32` lanes. Matches `_mm_max_epi32`.
#[inline]
pub fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vmaxq_s32(a.s32(), b.s32()) })
}

/// Packed minimum of eight `u16` lanes. Matches `_mm_min_epu16`.
#[inline]
pub fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vminq_u16(a.u16(), b.u16()) })
}

/// Packed maximum of eight `u16` lanes. Matches `_mm_max_epu16`.
#[inline]
pub fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vmaxq_u16(a.u16(), b.u16()) })
}

/// Packed minimum of four `u32` lanes. Matches `_mm_min_epu32`.
#[inline]
pub fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vminq_u32(a.u32(), b.u32()) })
}

/// Packed maximum of four `u32` lanes. Matches `_mm_max_epu32`.
#[inline]
pub fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vmaxq_u32(a.u32(), b.u32()) })
}

// --- Multiply / pack ---

/// Multiply four `i32` lanes, keep the low 32 bits. Matches `_mm_mullo_epi32`.
#[inline]
pub fn _mm_mullo_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vmulq_s32(a.s32(), b.s32()) })
}

/// Multiply the even `i32` lanes to `i64`. Matches `_mm_mul_epi32`.
#[inline]
pub fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let al = vmovn_s64(a.s64());
        let bl = vmovn_s64(b.s64());
        __m128i::from_s64(vmull_s32(al, bl))
    }
}

/// Pack four `i32` from each input to eight `u16` with unsigned saturation.
/// Matches `_mm_packus_epi32`.
#[inline]
pub fn _mm_packus_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vcombine_u16(vqmovun_s32(a.s32()), vqmovun_s32(b.s32())) })
}

/// Compare two `i64` lanes for equality. Matches `_mm_cmpeq_epi64`.
#[inline]
pub fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u64(unsafe { vceqq_s64(a.s64(), b.s64()) })
}

// --- Min position ---

/// Find the minimum `u16` lane and its index. Matches `_mm_minpos_epu16`.
///
/// The result holds the minimum value in lane 0 and its index in lane 1.
#[inline]
pub fn _mm_minpos_epu16(a: __m128i) -> __m128i {
    let mut vals = [0u16; 8];
    unsafe { vst1q_u16(vals.as_mut_ptr(), a.u16()) };
    let mut min = vals[0];
    let mut idx = 0u16;
    for (i, &v) in vals.iter().enumerate().skip(1) {
        if v < min {
            min = v;
            idx = i as u16;
        }
    }
    _mm_set_epi16(0, 0, 0, 0, 0, 0, idx as i16, min as i16)
}

// --- Dot product ---

/// Dot product of four `f32` lanes with mask select. Matches `_mm_dp_ps`.
///
/// Bits 4-7 of `IMM` pick which products contribute. Bits 0-3 pick which output
/// lanes receive the sum.
#[inline]
pub fn _mm_dp_ps<const IMM: i32>(a: __m128, b: __m128) -> __m128 {
    let mut av = [0.0f32; 4];
    let mut bv = [0.0f32; 4];
    unsafe {
        vst1q_f32(av.as_mut_ptr(), a.f32());
        vst1q_f32(bv.as_mut_ptr(), b.f32());
    }
    let mut sum = 0.0f32;
    for i in 0..4 {
        if (IMM >> (4 + i)) & 1 == 1 {
            sum += av[i] * bv[i];
        }
    }
    let mut out = [0.0f32; 4];
    for (i, o) in out.iter_mut().enumerate() {
        if (IMM >> i) & 1 == 1 {
            *o = sum;
        }
    }
    __m128::from_f32(unsafe { vld1q_f32(out.as_ptr()) })
}

/// Dot product of two `f64` lanes with mask select. Matches `_mm_dp_pd`.
#[inline]
pub fn _mm_dp_pd<const IMM: i32>(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a0 = vgetq_lane_f64(a.f64(), 0);
        let a1 = vgetq_lane_f64(a.f64(), 1);
        let b0 = vgetq_lane_f64(b.f64(), 0);
        let b1 = vgetq_lane_f64(b.f64(), 1);
        let mut sum = 0.0f64;
        if (IMM >> 4) & 1 == 1 {
            sum += a0 * b0;
        }
        if (IMM >> 5) & 1 == 1 {
            sum += a1 * b1;
        }
        let e0 = if IMM & 1 == 1 { sum } else { 0.0 };
        let e1 = if IMM & 2 == 2 { sum } else { 0.0 };
        _mm_set_pd(e1, e0)
    }
}

// --- Extract / insert ---

/// Extract one `i8` lane, zero-extended. Matches `_mm_extract_epi8`.
#[inline]
pub fn _mm_extract_epi8<const IMM: i32>(a: __m128i) -> i32 {
    to_u8_array(a)[(IMM & 0xf) as usize] as i32
}

/// Extract one `i32` lane. Matches `_mm_extract_epi32`.
#[inline]
pub fn _mm_extract_epi32<const IMM: i32>(a: __m128i) -> i32 {
    to_i32_array(a)[(IMM & 0x3) as usize]
}

/// Extract one `i64` lane. Matches `_mm_extract_epi64`.
#[inline]
pub fn _mm_extract_epi64<const IMM: i32>(a: __m128i) -> i64 {
    let mut vals = [0i64; 2];
    unsafe { vst1q_s64(vals.as_mut_ptr(), a.s64()) };
    vals[(IMM & 0x1) as usize]
}

/// Extract one `i16` lane, zero-extended. Matches `_mm_extract_epi16`.
#[inline]
pub fn _mm_extract_epi16<const IMM: i32>(a: __m128i) -> i32 {
    to_i16_array(a)[(IMM & 0x7) as usize] as u16 as i32
}

/// Insert `i` into one `i8` lane. Matches `_mm_insert_epi8`.
#[inline]
pub fn _mm_insert_epi8<const IMM: i32>(a: __m128i, i: i32) -> __m128i {
    let mut vals = to_u8_array(a);
    vals[(IMM & 0xf) as usize] = i as u8;
    __m128i::from_u8(unsafe { vld1q_u8(vals.as_ptr()) })
}

/// Insert `i` into one `i32` lane. Matches `_mm_insert_epi32`.
#[inline]
pub fn _mm_insert_epi32<const IMM: i32>(a: __m128i, i: i32) -> __m128i {
    let mut vals = to_i32_array(a);
    vals[(IMM & 0x3) as usize] = i;
    __m128i::from_s32(unsafe { vld1q_s32(vals.as_ptr()) })
}

/// Insert `i` into one `i64` lane. Matches `_mm_insert_epi64`.
#[inline]
pub fn _mm_insert_epi64<const IMM: i32>(a: __m128i, i: i64) -> __m128i {
    let mut vals = [0i64; 2];
    unsafe { vst1q_s64(vals.as_mut_ptr(), a.s64()) };
    vals[(IMM & 0x1) as usize] = i;
    __m128i::from_s64(unsafe { vld1q_s64(vals.as_ptr()) })
}

/// Insert `i` into one `i16` lane. Matches `_mm_insert_epi16`.
#[inline]
pub fn _mm_insert_epi16<const IMM: i32>(a: __m128i, i: i32) -> __m128i {
    let mut vals = to_i16_array(a);
    vals[(IMM & 0x7) as usize] = i as i16;
    __m128i::from_s16(unsafe { vld1q_s16(vals.as_ptr()) })
}

// --- Tests ---

/// True if `(a & b) == 0`. Matches `_mm_testz_si128`.
#[inline]
pub fn _mm_testz_si128(a: __m128i, b: __m128i) -> i32 {
    let r = _mm_and_si128(a, b);
    i32::from(is_all_zero(r))
}

/// True if `((!a) & b) == 0`. Matches `_mm_testc_si128`.
#[inline]
pub fn _mm_testc_si128(a: __m128i, b: __m128i) -> i32 {
    let r = _mm_andnot_si128(a, b);
    i32::from(is_all_zero(r))
}

/// True if both `a & b` and `(!a) & b` are nonzero. Matches `_mm_testnzc_si128`.
#[inline]
pub fn _mm_testnzc_si128(a: __m128i, b: __m128i) -> i32 {
    let zf = _mm_testz_si128(a, b);
    let cf = _mm_testc_si128(a, b);
    i32::from(zf == 0 && cf == 0)
}

/// True if all bits are one. Matches `_mm_test_all_ones`.
#[inline]
pub fn _mm_test_all_ones(a: __m128i) -> i32 {
    i32::from(is_all_ones(a))
}

/// True if `(a & mask) == 0`. Matches `_mm_test_all_zeros`.
#[inline]
pub fn _mm_test_all_zeros(a: __m128i, mask: __m128i) -> i32 {
    _mm_testz_si128(a, mask)
}

/// True if `a & mask` has both set and clear bits. Matches `_mm_test_mix_ones_zeros`.
#[inline]
pub fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) -> i32 {
    _mm_testnzc_si128(a, mask)
}

fn is_all_zero(a: __m128i) -> bool {
    unsafe { vmaxvq_u32(a.u32()) == 0 }
}

fn is_all_ones(a: __m128i) -> bool {
    unsafe { vminvq_u32(a.u32()) == u32::MAX }
}
