//! MMX `__m64` intrinsics and the SSE MMX-assisted helpers.
//!
//! These operate on 64-bit vectors: eight `i8`, four `i16`, or two `i32` lanes.
//! Lane ordering matches the wider types, least significant first. The suite is
//! little-endian only.

use crate::types::*;
use core::arch::aarch64::*;

/// Set two `i32` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_pi32`.
#[inline]
pub fn _mm_set_pi32(e1: i32, e0: i32) -> __m64 {
    let data = [e0, e1];
    __m64::from_s32(unsafe { vld1_s32(data.as_ptr()) })
}

/// Set four `i16` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_pi16`.
#[inline]
pub fn _mm_set_pi16(e3: i16, e2: i16, e1: i16, e0: i16) -> __m64 {
    let data = [e0, e1, e2, e3];
    __m64::from_s16(unsafe { vld1_s16(data.as_ptr()) })
}

/// Set eight `i8` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_pi8`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn _mm_set_pi8(e7: i8, e6: i8, e5: i8, e4: i8, e3: i8, e2: i8, e1: i8, e0: i8) -> __m64 {
    let data = [e0, e1, e2, e3, e4, e5, e6, e7];
    __m64::from_s8(unsafe { vld1_s8(data.as_ptr()) })
}

/// Broadcast one `i32` to both lanes. Matches `_mm_set1_pi32`.
#[inline]
pub fn _mm_set1_pi32(w: i32) -> __m64 {
    __m64::from_s32(unsafe { vdup_n_s32(w) })
}

/// Return a zeroed `__m64`. Matches `_mm_setzero_si64`.
#[inline]
pub fn _mm_setzero_si64() -> __m64 {
    __m64::from_s32(unsafe { vdup_n_s32(0) })
}

/// Extract lane 0 as `i32`. Matches `_mm_cvtsi64_si32`.
#[inline]
pub fn _mm_cvtsi64_si32(a: __m64) -> i32 {
    unsafe { vget_lane_s32(a.s32(), 0) }
}

/// Widen a `__m64` into the low 64 bits of a `__m128i`, zeroing the high half.
/// Matches `_mm_movpi64_epi64`.
#[inline]
pub fn _mm_movpi64_epi64(a: __m64) -> __m128i {
    __m128i::from_s64(unsafe { vcombine_s64(a.0, vdup_n_s64(0)) })
}

/// Take the low 64 bits of a `__m128i` as a `__m64`. Matches `_mm_movepi64_pi64`.
#[inline]
pub fn _mm_movepi64_pi64(a: __m128i) -> __m64 {
    __m64(unsafe { vget_low_s64(a.s64()) })
}

/// Add four `i16` lanes with wraparound. Matches `_mm_add_pi16`.
#[inline]
pub fn _mm_add_pi16(a: __m64, b: __m64) -> __m64 {
    __m64::from_s16(unsafe { vadd_s16(a.s16(), b.s16()) })
}

/// Add two `i32` lanes with wraparound. Matches `_mm_add_pi32`.
#[inline]
pub fn _mm_add_pi32(a: __m64, b: __m64) -> __m64 {
    __m64::from_s32(unsafe { vadd_s32(a.s32(), b.s32()) })
}

/// Subtract four `i16` lanes with wraparound. Matches `_mm_sub_pi16`.
#[inline]
pub fn _mm_sub_pi16(a: __m64, b: __m64) -> __m64 {
    __m64::from_s16(unsafe { vsub_s16(a.s16(), b.s16()) })
}

/// Packed maximum of four `i16` lanes. Matches `_mm_max_pi16`.
#[inline]
pub fn _mm_max_pi16(a: __m64, b: __m64) -> __m64 {
    __m64::from_s16(unsafe { vmax_s16(a.s16(), b.s16()) })
}

/// Packed minimum of four `i16` lanes. Matches `_mm_min_pi16`.
#[inline]
pub fn _mm_min_pi16(a: __m64, b: __m64) -> __m64 {
    __m64::from_s16(unsafe { vmin_s16(a.s16(), b.s16()) })
}

/// Packed maximum of eight `u8` lanes. Matches `_mm_max_pu8`.
#[inline]
pub fn _mm_max_pu8(a: __m64, b: __m64) -> __m64 {
    __m64::from_u8(unsafe { vmax_u8(a.u8(), b.u8()) })
}

/// Packed minimum of eight `u8` lanes. Matches `_mm_min_pu8`.
#[inline]
pub fn _mm_min_pu8(a: __m64, b: __m64) -> __m64 {
    __m64::from_u8(unsafe { vmin_u8(a.u8(), b.u8()) })
}

/// Rounded average of eight `u8` lanes. Matches `_mm_avg_pu8`.
#[inline]
pub fn _mm_avg_pu8(a: __m64, b: __m64) -> __m64 {
    __m64::from_u8(unsafe { vrhadd_u8(a.u8(), b.u8()) })
}

/// Rounded average of four `u16` lanes. Matches `_mm_avg_pu16`.
#[inline]
pub fn _mm_avg_pu16(a: __m64, b: __m64) -> __m64 {
    __m64::from_u16(unsafe { vrhadd_u16(a.u16(), b.u16()) })
}

/// Gather the sign bit of each of the eight `i8` lanes into 8 bits.
/// Matches `_mm_movemask_pi8`.
#[inline]
pub fn _mm_movemask_pi8(a: __m64) -> i32 {
    unsafe {
        let msbs = vshr_n_u8::<7>(a.u8());
        let shift_table: [i8; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
        let shifts = vld1_s8(shift_table.as_ptr());
        let positioned = vshl_u8(msbs, shifts);
        vaddv_u8(positioned) as i32
    }
}

/// Sum absolute differences of eight `u8` lanes into a single `u16` total in
/// lane 0. Matches `_mm_sad_pu8`.
#[inline]
pub fn _mm_sad_pu8(a: __m64, b: __m64) -> __m64 {
    unsafe {
        let diff = vabd_u8(a.u8(), b.u8());
        let sum = vaddlv_u8(diff) as u16;
        __m64::from_u16(vset_lane_u16(sum, vdup_n_u16(0), 0))
    }
}
