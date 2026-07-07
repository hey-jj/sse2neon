//! SSE2 intrinsics: packed integers and `f64` vectors.
//!
//! Covers integer arithmetic with and without saturation, bitwise logic,
//! comparisons, shifts, pack and unpack, set/load/store, movemask, and the
//! double-precision arithmetic and compare families.

use crate::types::*;
use core::arch::aarch64::*;

// --- Integer arithmetic ---

/// Add sixteen `i8` lanes with wraparound. Matches `_mm_add_epi8`.
#[inline]
pub fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vaddq_s8(a.s8(), b.s8()) })
}

/// Add eight `i16` lanes with wraparound. Matches `_mm_add_epi16`.
#[inline]
pub fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vaddq_s16(a.s16(), b.s16()) })
}

/// Add four `i32` lanes with wraparound. Matches `_mm_add_epi32`.
#[inline]
pub fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vaddq_s32(a.s32(), b.s32()) })
}

/// Add two `i64` lanes with wraparound. Matches `_mm_add_epi64`.
#[inline]
pub fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vaddq_s64(a.s64(), b.s64()) })
}

/// Subtract sixteen `i8` lanes with wraparound. Matches `_mm_sub_epi8`.
#[inline]
pub fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vsubq_s8(a.s8(), b.s8()) })
}

/// Subtract eight `i16` lanes with wraparound. Matches `_mm_sub_epi16`.
#[inline]
pub fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vsubq_s16(a.s16(), b.s16()) })
}

/// Subtract four `i32` lanes with wraparound. Matches `_mm_sub_epi32`.
#[inline]
pub fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vsubq_s32(a.s32(), b.s32()) })
}

/// Subtract two `i64` lanes with wraparound. Matches `_mm_sub_epi64`.
#[inline]
pub fn _mm_sub_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vsubq_s64(a.s64(), b.s64()) })
}

/// Add eight `i16` lanes with signed saturation. Matches `_mm_adds_epi16`.
#[inline]
pub fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vqaddq_s16(a.s16(), b.s16()) })
}

/// Add sixteen `i8` lanes with signed saturation. Matches `_mm_adds_epi8`.
#[inline]
pub fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vqaddq_s8(a.s8(), b.s8()) })
}

/// Add sixteen `u8` lanes with unsigned saturation. Matches `_mm_adds_epu8`.
#[inline]
pub fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vqaddq_u8(a.u8(), b.u8()) })
}

/// Add eight `u16` lanes with unsigned saturation. Matches `_mm_adds_epu16`.
#[inline]
pub fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vqaddq_u16(a.u16(), b.u16()) })
}

/// Subtract eight `i16` lanes with signed saturation. Matches `_mm_subs_epi16`.
#[inline]
pub fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vqsubq_s16(a.s16(), b.s16()) })
}

/// Subtract sixteen `i8` lanes with signed saturation. Matches `_mm_subs_epi8`.
#[inline]
pub fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vqsubq_s8(a.s8(), b.s8()) })
}

/// Subtract sixteen `u8` lanes with unsigned saturation. Matches `_mm_subs_epu8`.
#[inline]
pub fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vqsubq_u8(a.u8(), b.u8()) })
}

/// Subtract eight `u16` lanes with unsigned saturation. Matches `_mm_subs_epu16`.
#[inline]
pub fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vqsubq_u16(a.u16(), b.u16()) })
}

/// Multiply eight `i16` lanes, keep the low 16 bits. Matches `_mm_mullo_epi16`.
#[inline]
pub fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vmulq_s16(a.s16(), b.s16()) })
}

/// Multiply eight `i16` lanes, keep the high 16 bits. Matches `_mm_mulhi_epi16`.
#[inline]
pub fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let al = vget_low_s16(a.s16());
        let ah = vget_high_s16(a.s16());
        let bl = vget_low_s16(b.s16());
        let bh = vget_high_s16(b.s16());
        let lo = vmull_s16(al, bl);
        let hi = vmull_s16(ah, bh);
        // Take the high halfword of each 32-bit product.
        let r = vuzp2q_s16(vreinterpretq_s16_s32(lo), vreinterpretq_s16_s32(hi));
        __m128i::from_s16(r)
    }
}

/// Multiply eight `u16` lanes, keep the high 16 bits. Matches `_mm_mulhi_epu16`.
#[inline]
pub fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let al = vget_low_u16(a.u16());
        let ah = vget_high_u16(a.u16());
        let bl = vget_low_u16(b.u16());
        let bh = vget_high_u16(b.u16());
        let lo = vmull_u16(al, bl);
        let hi = vmull_u16(ah, bh);
        let r = vuzp2q_u16(vreinterpretq_u16_u32(lo), vreinterpretq_u16_u32(hi));
        __m128i::from_u16(r)
    }
}

/// Multiply pairs of `i16`, add adjacent products to `i32`. Matches `_mm_madd_epi16`.
#[inline]
pub fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let al = vget_low_s16(a.s16());
        let ah = vget_high_s16(a.s16());
        let bl = vget_low_s16(b.s16());
        let bh = vget_high_s16(b.s16());
        let lo = vmull_s16(al, bl);
        let hi = vmull_s16(ah, bh);
        // Pairwise add the 32-bit products: (p0+p1, p2+p3, ...).
        __m128i::from_s32(vpaddq_s32(lo, hi))
    }
}

/// Multiply the low `u32` of each 64-bit lane, produce `u64`. Matches `_mm_mul_epu32`.
#[inline]
pub fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        // Zero the odd 32-bit lanes, then widen-multiply the even ones.
        let al = vmovn_u64(a.u64());
        let bl = vmovn_u64(b.u64());
        __m128i::from_u64(vmull_u32(al, bl))
    }
}

// --- Bitwise ---

/// Bitwise AND of two integer vectors. Matches `_mm_and_si128`.
#[inline]
pub fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vandq_s64(a.s64(), b.s64()) })
}

/// Bitwise OR of two integer vectors. Matches `_mm_or_si128`.
#[inline]
pub fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vorrq_s64(a.s64(), b.s64()) })
}

/// Bitwise XOR of two integer vectors. Matches `_mm_xor_si128`.
#[inline]
pub fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { veorq_s64(a.s64(), b.s64()) })
}

/// Bitwise `(!a) & b`. The NOT applies to `a`. Matches `_mm_andnot_si128`.
#[inline]
pub fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vbicq_s64(b.s64(), a.s64()) })
}

// --- Averages ---

/// Rounded average of sixteen `u8` lanes. Matches `_mm_avg_epu8`.
#[inline]
pub fn _mm_avg_epu8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vrhaddq_u8(a.u8(), b.u8()) })
}

/// Rounded average of eight `u16` lanes. Matches `_mm_avg_epu16`.
#[inline]
pub fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vrhaddq_u16(a.u16(), b.u16()) })
}

// --- Min / max ---

/// Packed minimum of eight `i16` lanes. Matches `_mm_min_epi16`.
#[inline]
pub fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vminq_s16(a.s16(), b.s16()) })
}

/// Packed maximum of eight `i16` lanes. Matches `_mm_max_epi16`.
#[inline]
pub fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vmaxq_s16(a.s16(), b.s16()) })
}

/// Packed minimum of sixteen `u8` lanes. Matches `_mm_min_epu8`.
#[inline]
pub fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vminq_u8(a.u8(), b.u8()) })
}

/// Packed maximum of sixteen `u8` lanes. Matches `_mm_max_epu8`.
#[inline]
pub fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vmaxq_u8(a.u8(), b.u8()) })
}

// --- Compares ---

/// Lanes where sixteen `i8` are equal become all-ones. Matches `_mm_cmpeq_epi8`.
#[inline]
pub fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vceqq_s8(a.s8(), b.s8()) })
}

/// Lanes where eight `i16` are equal become all-ones. Matches `_mm_cmpeq_epi16`.
#[inline]
pub fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vceqq_s16(a.s16(), b.s16()) })
}

/// Lanes where four `i32` are equal become all-ones. Matches `_mm_cmpeq_epi32`.
#[inline]
pub fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vceqq_s32(a.s32(), b.s32()) })
}

/// Signed `a > b` for sixteen `i8` lanes. Matches `_mm_cmpgt_epi8`.
#[inline]
pub fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vcgtq_s8(a.s8(), b.s8()) })
}

/// Signed `a > b` for eight `i16` lanes. Matches `_mm_cmpgt_epi16`.
#[inline]
pub fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vcgtq_s16(a.s16(), b.s16()) })
}

/// Signed `a > b` for four `i32` lanes. Matches `_mm_cmpgt_epi32`.
#[inline]
pub fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vcgtq_s32(a.s32(), b.s32()) })
}

/// Signed `a < b` for sixteen `i8` lanes. Matches `_mm_cmplt_epi8`.
#[inline]
pub fn _mm_cmplt_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vcltq_s8(a.s8(), b.s8()) })
}

/// Signed `a < b` for eight `i16` lanes. Matches `_mm_cmplt_epi16`.
#[inline]
pub fn _mm_cmplt_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u16(unsafe { vcltq_s16(a.s16(), b.s16()) })
}

/// Signed `a < b` for four `i32` lanes. Matches `_mm_cmplt_epi32`.
#[inline]
pub fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u32(unsafe { vcltq_s32(a.s32(), b.s32()) })
}

// --- Shifts by immediate ---

/// Shift eight `i16` lanes left by `IMM`. Count >= 16 yields zero.
/// Matches `_mm_slli_epi16`.
#[inline]
pub fn _mm_slli_epi16<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..16).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_s16(unsafe { vshlq_s16(a.s16(), vdupq_n_s16(IMM as i16)) })
}

/// Shift four `i32` lanes left by `IMM`. Count >= 32 yields zero.
/// Matches `_mm_slli_epi32`.
#[inline]
pub fn _mm_slli_epi32<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..32).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_s32(unsafe { vshlq_s32(a.s32(), vdupq_n_s32(IMM)) })
}

/// Shift two `i64` lanes left by `IMM`. Count >= 64 yields zero.
/// Matches `_mm_slli_epi64`.
#[inline]
pub fn _mm_slli_epi64<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..64).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_s64(unsafe { vshlq_s64(a.s64(), vdupq_n_s64(IMM as i64)) })
}

/// Logical shift eight `u16` lanes right by `IMM`. Matches `_mm_srli_epi16`.
#[inline]
pub fn _mm_srli_epi16<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..16).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_u16(unsafe { vshlq_u16(a.u16(), vdupq_n_s16(-(IMM as i16))) })
}

/// Logical shift four `u32` lanes right by `IMM`. Matches `_mm_srli_epi32`.
#[inline]
pub fn _mm_srli_epi32<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..32).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_u32(unsafe { vshlq_u32(a.u32(), vdupq_n_s32(-IMM)) })
}

/// Logical shift two `u64` lanes right by `IMM`. Matches `_mm_srli_epi64`.
#[inline]
pub fn _mm_srli_epi64<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if !(0..64).contains(&IMM) {
        return _mm_setzero_si128();
    }
    __m128i::from_u64(unsafe { vshlq_u64(a.u64(), vdupq_n_s64(-(IMM as i64))) })
}

/// Arithmetic shift eight `i16` lanes right by `IMM`. Count >= 16 fills sign.
/// Matches `_mm_srai_epi16`.
#[inline]
pub fn _mm_srai_epi16<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let sh = IMM.clamp(0, 15);
    __m128i::from_s16(unsafe { vshlq_s16(a.s16(), vdupq_n_s16(-(sh as i16))) })
}

/// Arithmetic shift four `i32` lanes right by `IMM`. Count >= 32 fills sign.
/// Matches `_mm_srai_epi32`.
#[inline]
pub fn _mm_srai_epi32<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let sh = IMM.clamp(0, 31);
    __m128i::from_s32(unsafe { vshlq_s32(a.s32(), vdupq_n_s32(-sh)) })
}

/// Shift the whole vector left by `IMM` bytes. Matches `_mm_slli_si128`.
#[inline]
pub fn _mm_slli_si128<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if IMM == 0 {
        return a;
    }
    if !(0..16).contains(&IMM) {
        return _mm_setzero_si128();
    }
    let mut bytes = [0u8; 16];
    let av = to_u8_array(a);
    for i in 0..(16 - IMM as usize) {
        bytes[i + IMM as usize] = av[i];
    }
    __m128i::from_u8(unsafe { vld1q_u8(bytes.as_ptr()) })
}

/// Shift the whole vector right by `IMM` bytes. Matches `_mm_srli_si128`.
#[inline]
pub fn _mm_srli_si128<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    if IMM == 0 {
        return a;
    }
    if !(0..16).contains(&IMM) {
        return _mm_setzero_si128();
    }
    let mut bytes = [0u8; 16];
    let av = to_u8_array(a);
    for i in 0..(16 - IMM as usize) {
        bytes[i] = av[i + IMM as usize];
    }
    __m128i::from_u8(unsafe { vld1q_u8(bytes.as_ptr()) })
}

/// Alias of [`_mm_slli_si128`]. Matches `_mm_bslli_si128`.
#[inline]
pub fn _mm_bslli_si128<const IMM: i32>(a: __m128i) -> __m128i {
    _mm_slli_si128::<IMM>(a)
}

/// Alias of [`_mm_srli_si128`]. Matches `_mm_bsrli_si128`.
#[inline]
pub fn _mm_bsrli_si128<const IMM: i32>(a: __m128i) -> __m128i {
    _mm_srli_si128::<IMM>(a)
}

// --- Pack / unpack ---

/// Pack eight `i16` from each input to sixteen `i8` with signed saturation.
/// Matches `_mm_packs_epi16`.
#[inline]
pub fn _mm_packs_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vcombine_s8(vqmovn_s16(a.s16()), vqmovn_s16(b.s16())) })
}

/// Pack four `i32` from each input to eight `i16` with signed saturation.
/// Matches `_mm_packs_epi32`.
#[inline]
pub fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vcombine_s16(vqmovn_s32(a.s32()), vqmovn_s32(b.s32())) })
}

/// Pack eight `i16` from each input to sixteen `u8` with unsigned saturation.
/// Matches `_mm_packus_epi16`.
#[inline]
pub fn _mm_packus_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u8(unsafe { vcombine_u8(vqmovun_s16(a.s16()), vqmovun_s16(b.s16())) })
}

/// Interleave the high eight `i8` lanes. Matches `_mm_unpackhi_epi8`.
#[inline]
pub fn _mm_unpackhi_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vzip2q_s8(a.s8(), b.s8()) })
}

/// Interleave the high four `i16` lanes. Matches `_mm_unpackhi_epi16`.
#[inline]
pub fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vzip2q_s16(a.s16(), b.s16()) })
}

/// Interleave the high two `i32` lanes. Matches `_mm_unpackhi_epi32`.
#[inline]
pub fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vzip2q_s32(a.s32(), b.s32()) })
}

/// Interleave the high `i64` lane. Matches `_mm_unpackhi_epi64`.
#[inline]
pub fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vzip2q_s64(a.s64(), b.s64()) })
}

/// Interleave the low eight `i8` lanes. Matches `_mm_unpacklo_epi8`.
#[inline]
pub fn _mm_unpacklo_epi8(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s8(unsafe { vzip1q_s8(a.s8(), b.s8()) })
}

/// Interleave the low four `i16` lanes. Matches `_mm_unpacklo_epi16`.
#[inline]
pub fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s16(unsafe { vzip1q_s16(a.s16(), b.s16()) })
}

/// Interleave the low two `i32` lanes. Matches `_mm_unpacklo_epi32`.
#[inline]
pub fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s32(unsafe { vzip1q_s32(a.s32(), b.s32()) })
}

/// Interleave the low `i64` lane. Matches `_mm_unpacklo_epi64`.
#[inline]
pub fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_s64(unsafe { vzip1q_s64(a.s64(), b.s64()) })
}

// --- Move / shuffle ---

/// Copy the low `i64` lane, zero the high lane. Matches `_mm_move_epi64`.
#[inline]
pub fn _mm_move_epi64(a: __m128i) -> __m128i {
    unsafe {
        let lo = vgetq_lane_s64(a.s64(), 0);
        __m128i::from_s64(vsetq_lane_s64(0, vdupq_n_s64(lo), 1))
    }
}

/// Shuffle four `i32` lanes by `IMM`. Matches `_mm_shuffle_epi32`.
///
/// Each 2-bit field of `IMM` picks a source lane. `IMM` is a compile-time
/// constant in `0..256`.
#[inline]
pub fn _mm_shuffle_epi32<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let av = to_i32_array(a);
    let out = [
        av[(IMM & 0x3) as usize],
        av[((IMM >> 2) & 0x3) as usize],
        av[((IMM >> 4) & 0x3) as usize],
        av[((IMM >> 6) & 0x3) as usize],
    ];
    __m128i::from_s32(unsafe { vld1q_s32(out.as_ptr()) })
}

/// Shuffle the high four `i16` lanes by `IMM`, keep the low four.
/// Matches `_mm_shufflehi_epi16`. `IMM` is a compile-time constant in `0..256`.
#[inline]
pub fn _mm_shufflehi_epi16<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let av = to_i16_array(a);
    let mut out = av;
    out[4] = av[4 + (IMM & 0x3) as usize];
    out[5] = av[4 + ((IMM >> 2) & 0x3) as usize];
    out[6] = av[4 + ((IMM >> 4) & 0x3) as usize];
    out[7] = av[4 + ((IMM >> 6) & 0x3) as usize];
    __m128i::from_s16(unsafe { vld1q_s16(out.as_ptr()) })
}

/// Shuffle the low four `i16` lanes by `IMM`, keep the high four.
/// Matches `_mm_shufflelo_epi16`. `IMM` is a compile-time constant in `0..256`.
#[inline]
pub fn _mm_shufflelo_epi16<const IMM: i32>(a: __m128i) -> __m128i {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let av = to_i16_array(a);
    let mut out = av;
    out[0] = av[(IMM & 0x3) as usize];
    out[1] = av[((IMM >> 2) & 0x3) as usize];
    out[2] = av[((IMM >> 4) & 0x3) as usize];
    out[3] = av[((IMM >> 6) & 0x3) as usize];
    __m128i::from_s16(unsafe { vld1q_s16(out.as_ptr()) })
}

// --- Movemask ---

/// Gather the sign bit of each of the sixteen `i8` lanes into 16 bits.
/// Matches `_mm_movemask_epi8`.
#[inline]
pub fn _mm_movemask_epi8(a: __m128i) -> i32 {
    unsafe {
        // Isolate each byte's MSB, then shift left by its position within the
        // half and horizontally add the two 8-byte halves.
        let msbs = vshrq_n_u8::<7>(a.u8());
        let shift_table: [i8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7];
        let shifts = vld1q_s8(shift_table.as_ptr());
        let positioned = vshlq_u8(msbs, shifts);
        let lo = vaddv_u8(vget_low_u8(positioned)) as i32;
        let hi = vaddv_u8(vget_high_u8(positioned)) as i32;
        lo | (hi << 8)
    }
}

// --- Sum of absolute differences ---

/// Sum absolute differences of `u8` lanes into two `u16` totals.
/// Matches `_mm_sad_epu8`.
#[inline]
pub fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        let t = vpaddlq_u8(vabdq_u8(a.u8(), b.u8()));
        __m128i::from_u64(vpaddlq_u32(vpaddlq_u16(t)))
    }
}

// --- Set / load / store ---

/// Return a zeroed integer vector. Matches `_mm_setzero_si128`.
#[inline]
pub fn _mm_setzero_si128() -> __m128i {
    __m128i::from_s64(unsafe { vdupq_n_s64(0) })
}

/// Broadcast one `i8` to all lanes. Matches `_mm_set1_epi8`.
#[inline]
pub fn _mm_set1_epi8(w: i8) -> __m128i {
    __m128i::from_s8(unsafe { vdupq_n_s8(w) })
}

/// Broadcast one `i16` to all lanes. Matches `_mm_set1_epi16`.
#[inline]
pub fn _mm_set1_epi16(w: i16) -> __m128i {
    __m128i::from_s16(unsafe { vdupq_n_s16(w) })
}

/// Broadcast one `i32` to all lanes. Matches `_mm_set1_epi32`.
#[inline]
pub fn _mm_set1_epi32(w: i32) -> __m128i {
    __m128i::from_s32(unsafe { vdupq_n_s32(w) })
}

/// Broadcast one `i64` to both lanes. Matches `_mm_set1_epi64x`.
#[inline]
pub fn _mm_set1_epi64x(w: i64) -> __m128i {
    __m128i::from_s64(unsafe { vdupq_n_s64(w) })
}

/// Set four `i32` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_epi32`.
#[inline]
pub fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    let data = [e0, e1, e2, e3];
    __m128i::from_s32(unsafe { vld1q_s32(data.as_ptr()) })
}

/// Set four `i32` lanes in order. Lane 0 is `e3`. Matches `_mm_setr_epi32`.
#[inline]
pub fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    _mm_set_epi32(e0, e1, e2, e3)
}

/// Set two `i64` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_epi64x`.
#[inline]
pub fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i {
    let data = [e0, e1];
    __m128i::from_s64(unsafe { vld1q_s64(data.as_ptr()) })
}

/// Set eight `i16` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_epi16`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn _mm_set_epi16(
    e7: i16,
    e6: i16,
    e5: i16,
    e4: i16,
    e3: i16,
    e2: i16,
    e1: i16,
    e0: i16,
) -> __m128i {
    let data = [e0, e1, e2, e3, e4, e5, e6, e7];
    __m128i::from_s16(unsafe { vld1q_s16(data.as_ptr()) })
}

/// Set sixteen `i8` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_epi8`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn _mm_set_epi8(
    e15: i8,
    e14: i8,
    e13: i8,
    e12: i8,
    e11: i8,
    e10: i8,
    e9: i8,
    e8: i8,
    e7: i8,
    e6: i8,
    e5: i8,
    e4: i8,
    e3: i8,
    e2: i8,
    e1: i8,
    e0: i8,
) -> __m128i {
    let data = [
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    ];
    __m128i::from_s8(unsafe { vld1q_s8(data.as_ptr()) })
}

/// Load a 128-bit integer vector from a 16-byte aligned pointer. Matches `_mm_load_si128`.
///
/// # Safety
/// `p` must point to 16 readable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_load_si128(p: *const __m128i) -> __m128i {
    __m128i::from_s64(vld1q_s64(p as *const i64))
}

/// Load a 128-bit integer vector from an unaligned pointer. Matches `_mm_loadu_si128`.
///
/// # Safety
/// `p` must point to 16 readable bytes.
#[inline]
pub unsafe fn _mm_loadu_si128(p: *const __m128i) -> __m128i {
    __m128i::from_s64(vld1q_s64(p as *const i64))
}

/// Load the low 64 bits, zero the high 64 bits. Matches `_mm_loadl_epi64`.
///
/// # Safety
/// `p` must point to 8 readable bytes.
#[inline]
pub unsafe fn _mm_loadl_epi64(p: *const __m128i) -> __m128i {
    let lo = core::ptr::read_unaligned(p as *const i64);
    __m128i::from_s64(vsetq_lane_s64(0, vdupq_n_s64(lo), 1))
}

/// Store a 128-bit integer vector to a 16-byte aligned pointer. Matches `_mm_store_si128`.
///
/// # Safety
/// `p` must point to 16 writable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_store_si128(p: *mut __m128i, a: __m128i) {
    vst1q_s64(p as *mut i64, a.s64());
}

/// Store a 128-bit integer vector to an unaligned pointer. Matches `_mm_storeu_si128`.
///
/// # Safety
/// `p` must point to 16 writable bytes.
#[inline]
pub unsafe fn _mm_storeu_si128(p: *mut __m128i, a: __m128i) {
    vst1q_s64(p as *mut i64, a.s64());
}

/// Store the low 64 bits to memory. Matches `_mm_storel_epi64`.
///
/// # Safety
/// `p` must point to 8 writable bytes.
#[inline]
pub unsafe fn _mm_storel_epi64(p: *mut __m128i, a: __m128i) {
    core::ptr::write_unaligned(p as *mut i64, vgetq_lane_s64(a.s64(), 0));
}

/// Extract the low `i32` lane. Matches `_mm_cvtsi128_si32`.
#[inline]
pub fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
    unsafe { vgetq_lane_s32(a.s32(), 0) }
}

/// Extract the low `i64` lane. Matches `_mm_cvtsi128_si64`.
#[inline]
pub fn _mm_cvtsi128_si64(a: __m128i) -> i64 {
    unsafe { vgetq_lane_s64(a.s64(), 0) }
}

/// Set lane 0 to `w`, zero the rest. Matches `_mm_cvtsi32_si128`.
#[inline]
pub fn _mm_cvtsi32_si128(w: i32) -> __m128i {
    __m128i::from_s32(unsafe { vsetq_lane_s32(w, vdupq_n_s32(0), 0) })
}

/// Set lane 0 to `w`, zero the high lane. Matches `_mm_cvtsi64_si128`.
#[inline]
pub fn _mm_cvtsi64_si128(w: i64) -> __m128i {
    __m128i::from_s64(unsafe { vsetq_lane_s64(w, vdupq_n_s64(0), 0) })
}

// --- Internal array views ---

pub(crate) fn to_u8_array(a: __m128i) -> [u8; 16] {
    let mut out = [0u8; 16];
    unsafe { vst1q_u8(out.as_mut_ptr(), a.u8()) };
    out
}

pub(crate) fn to_i16_array(a: __m128i) -> [i16; 8] {
    let mut out = [0i16; 8];
    unsafe { vst1q_s16(out.as_mut_ptr(), a.s16()) };
    out
}

pub(crate) fn to_i32_array(a: __m128i) -> [i32; 4] {
    let mut out = [0i32; 4];
    unsafe { vst1q_s32(out.as_mut_ptr(), a.s32()) };
    out
}

// ============================================================================
// Double-precision (`__m128d`) family
// ============================================================================

/// Add two `f64` lanes. Matches `_mm_add_pd`.
#[inline]
pub fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vaddq_f64(a.f64(), b.f64()) })
}

/// Subtract two `f64` lanes, `a - b`. Matches `_mm_sub_pd`.
#[inline]
pub fn _mm_sub_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vsubq_f64(a.f64(), b.f64()) })
}

/// Multiply two `f64` lanes. Matches `_mm_mul_pd`.
#[inline]
pub fn _mm_mul_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vmulq_f64(a.f64(), b.f64()) })
}

/// Divide two `f64` lanes, `a / b`. Matches `_mm_div_pd`.
#[inline]
pub fn _mm_div_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vdivq_f64(a.f64(), b.f64()) })
}

/// Square root of two `f64` lanes. Matches `_mm_sqrt_pd`.
#[inline]
pub fn _mm_sqrt_pd(a: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vsqrtq_f64(a.f64()) })
}

/// Packed maximum of two `f64` lanes.
/// Uses NEON max: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_max_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vmaxq_f64(a.f64(), b.f64()) })
}

/// Packed minimum of two `f64` lanes.
/// Uses NEON min: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_min_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vminq_f64(a.f64(), b.f64()) })
}

/// Add the low lanes, copy the high lane from `a`. Matches `_mm_add_sd`.
#[inline]
pub fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_add_pd(a, b))
}

/// Subtract the low lanes, copy the high lane from `a`. Matches `_mm_sub_sd`.
#[inline]
pub fn _mm_sub_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_sub_pd(a, b))
}

/// Multiply the low lanes, copy the high lane from `a`. Matches `_mm_mul_sd`.
#[inline]
pub fn _mm_mul_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_mul_pd(a, b))
}

/// Divide the low lanes, copy the high lane from `a`. Matches `_mm_div_sd`.
#[inline]
pub fn _mm_div_sd(a: __m128d, b: __m128d) -> __m128d {
    _mm_move_sd(a, _mm_div_pd(a, b))
}

/// Bitwise AND of two `f64` vectors. Matches `_mm_and_pd`.
#[inline]
pub fn _mm_and_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vandq_u64(a.u64(), b.u64()) })
}

/// Bitwise OR of two `f64` vectors. Matches `_mm_or_pd`.
#[inline]
pub fn _mm_or_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vorrq_u64(a.u64(), b.u64()) })
}

/// Bitwise XOR of two `f64` vectors. Matches `_mm_xor_pd`.
#[inline]
pub fn _mm_xor_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { veorq_u64(a.u64(), b.u64()) })
}

/// Bitwise `(!a) & b`. The NOT applies to `a`. Matches `_mm_andnot_pd`.
#[inline]
pub fn _mm_andnot_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vbicq_u64(b.u64(), a.u64()) })
}

/// Compare `a == b` lane-wise. Matches `_mm_cmpeq_pd`.
#[inline]
pub fn _mm_cmpeq_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vceqq_f64(a.f64(), b.f64()) })
}

/// Compare `a < b` lane-wise. Matches `_mm_cmplt_pd`.
#[inline]
pub fn _mm_cmplt_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vcltq_f64(a.f64(), b.f64()) })
}

/// Compare `a <= b` lane-wise. Matches `_mm_cmple_pd`.
#[inline]
pub fn _mm_cmple_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vcleq_f64(a.f64(), b.f64()) })
}

/// Compare `a > b` lane-wise. Matches `_mm_cmpgt_pd`.
#[inline]
pub fn _mm_cmpgt_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vcgtq_f64(a.f64(), b.f64()) })
}

/// Compare `a >= b` lane-wise. Matches `_mm_cmpge_pd`.
#[inline]
pub fn _mm_cmpge_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe { vcgeq_f64(a.f64(), b.f64()) })
}

/// Compare `a != b` lane-wise. Matches `_mm_cmpneq_pd`.
#[inline]
pub fn _mm_cmpneq_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_u64(unsafe {
        let eq = vceqq_f64(a.f64(), b.f64());
        vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(eq)))
    })
}

/// Ordered compare: all-ones where neither lane is NaN. Matches `_mm_cmpord_pd`.
#[inline]
pub fn _mm_cmpord_pd(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a_ord = vceqq_f64(a.f64(), a.f64());
        let b_ord = vceqq_f64(b.f64(), b.f64());
        __m128d::from_u64(vandq_u64(a_ord, b_ord))
    }
}

/// Unordered compare: all-ones where either lane is NaN. Matches `_mm_cmpunord_pd`.
#[inline]
pub fn _mm_cmpunord_pd(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let a_ord = vceqq_f64(a.f64(), a.f64());
        let b_ord = vceqq_f64(b.f64(), b.f64());
        let ord = vandq_u64(a_ord, b_ord);
        __m128d::from_u64(vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(ord))))
    }
}

/// Copy the low lane from `b`, the high lane from `a`. Matches `_mm_move_sd`.
#[inline]
pub fn _mm_move_sd(a: __m128d, b: __m128d) -> __m128d {
    unsafe {
        let lo = vgetq_lane_f64(b.f64(), 0);
        __m128d::from_f64(vsetq_lane_f64(lo, a.f64(), 0))
    }
}

/// Interleave the high `f64` lanes. Matches `_mm_unpackhi_pd`.
#[inline]
pub fn _mm_unpackhi_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vzip2q_f64(a.f64(), b.f64()) })
}

/// Interleave the low `f64` lanes. Matches `_mm_unpacklo_pd`.
#[inline]
pub fn _mm_unpacklo_pd(a: __m128d, b: __m128d) -> __m128d {
    __m128d::from_f64(unsafe { vzip1q_f64(a.f64(), b.f64()) })
}

/// Gather the sign bit of each `f64` lane into 2 bits. Matches `_mm_movemask_pd`.
#[inline]
pub fn _mm_movemask_pd(a: __m128d) -> i32 {
    unsafe {
        let bits = vshrq_n_u64::<63>(a.u64());
        let lo = vgetq_lane_u64(bits, 0) as i32;
        let hi = vgetq_lane_u64(bits, 1) as i32;
        lo | (hi << 1)
    }
}

/// Return a zeroed `f64` vector. Matches `_mm_setzero_pd`.
#[inline]
pub fn _mm_setzero_pd() -> __m128d {
    __m128d::from_f64(unsafe { vdupq_n_f64(0.0) })
}

/// Broadcast one `f64` to both lanes. Matches `_mm_set1_pd`.
#[inline]
pub fn _mm_set1_pd(w: f64) -> __m128d {
    __m128d::from_f64(unsafe { vdupq_n_f64(w) })
}

/// Set two `f64` lanes, most significant first. Lane 0 is `e0`. Matches `_mm_set_pd`.
#[inline]
pub fn _mm_set_pd(e1: f64, e0: f64) -> __m128d {
    let data = [e0, e1];
    __m128d::from_f64(unsafe { vld1q_f64(data.as_ptr()) })
}

/// Set two `f64` lanes in order. Lane 0 is `e1`. Matches `_mm_setr_pd`.
#[inline]
pub fn _mm_setr_pd(e1: f64, e0: f64) -> __m128d {
    _mm_set_pd(e0, e1)
}

/// Set lane 0 to `w`, zero the high lane. Matches `_mm_set_sd`.
#[inline]
pub fn _mm_set_sd(w: f64) -> __m128d {
    let data = [w, 0.0];
    __m128d::from_f64(unsafe { vld1q_f64(data.as_ptr()) })
}

/// Extract lane 0 as `f64`. Matches `_mm_cvtsd_f64`.
#[inline]
pub fn _mm_cvtsd_f64(a: __m128d) -> f64 {
    unsafe { vgetq_lane_f64(a.f64(), 0) }
}

/// Load two `f64` from a 16-byte aligned pointer. Matches `_mm_load_pd`.
///
/// # Safety
/// `p` must point to 16 readable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_load_pd(p: *const f64) -> __m128d {
    __m128d::from_f64(vld1q_f64(p))
}

/// Load two `f64` from an unaligned pointer. Matches `_mm_loadu_pd`.
///
/// # Safety
/// `p` must point to 16 readable bytes.
#[inline]
pub unsafe fn _mm_loadu_pd(p: *const f64) -> __m128d {
    __m128d::from_f64(vld1q_f64(p))
}

/// Store two `f64` to a 16-byte aligned pointer. Matches `_mm_store_pd`.
///
/// # Safety
/// `p` must point to 16 writable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_store_pd(p: *mut f64, a: __m128d) {
    vst1q_f64(p, a.f64());
}

/// Store two `f64` to an unaligned pointer. Matches `_mm_storeu_pd`.
///
/// # Safety
/// `p` must point to 16 writable bytes.
#[inline]
pub unsafe fn _mm_storeu_pd(p: *mut f64, a: __m128d) {
    vst1q_f64(p, a.f64());
}

// ============================================================================
// Conversions between int, float, and double
// ============================================================================

/// Threshold above which an `f32` no longer fits `i32`. Values at or above this,
/// plus NaN, become the x86 "integer indefinite" `i32::MIN`.
const OVERFLOW_F32: f32 = 2147483648.0;

/// Blend `i32::MIN` into lanes where the source `f32` overflowed or is NaN.
///
/// NEON saturates out-of-range conversions and maps NaN to zero. x86 returns
/// `0x8000_0000` for both. This restores the x86 result.
#[inline]
fn cvtps_epi32_fixup(f: float32x4_t, cvt: int32x4_t) -> int32x4_t {
    unsafe {
        let overflow = vcgeq_f32(f, vdupq_n_f32(OVERFLOW_F32));
        let is_nan = vmvnq_u32(vceqq_f32(f, f));
        let need = vorrq_u32(overflow, is_nan);
        vbslq_s32(need, vdupq_n_s32(i32::MIN), cvt)
    }
}

/// Convert four `f32` to `i32` using the current MXCSR rounding mode.
///
/// The default mode rounds to nearest, ties to even. Out-of-range and NaN lanes
/// become `i32::MIN`. Matches `_mm_cvtps_epi32`.
#[inline]
pub fn _mm_cvtps_epi32(a: __m128) -> __m128i {
    use crate::mxcsr::_MM_GET_ROUNDING_MODE;
    unsafe {
        let f = a.f32();
        let cvt = match _MM_GET_ROUNDING_MODE() {
            crate::constants::_MM_ROUND_NEAREST => vcvtnq_s32_f32(f),
            crate::constants::_MM_ROUND_DOWN => vcvtmq_s32_f32(f),
            crate::constants::_MM_ROUND_UP => vcvtpq_s32_f32(f),
            _ => vcvtq_s32_f32(f),
        };
        __m128i::from_s32(cvtps_epi32_fixup(f, cvt))
    }
}

/// Convert four `f32` to `i32` by truncation toward zero.
///
/// Out-of-range and NaN lanes become `i32::MIN`. Matches `_mm_cvttps_epi32`.
#[inline]
pub fn _mm_cvttps_epi32(a: __m128) -> __m128i {
    unsafe {
        let f = a.f32();
        let cvt = vcvtq_s32_f32(f);
        __m128i::from_s32(cvtps_epi32_fixup(f, cvt))
    }
}

/// Convert four `i32` to `f32`. Matches `_mm_cvtepi32_ps`.
#[inline]
pub fn _mm_cvtepi32_ps(a: __m128i) -> __m128 {
    __m128::from_f32(unsafe { vcvtq_f32_s32(a.s32()) })
}

/// Convert two `i32` (low lanes) to two `f64`. Matches `_mm_cvtepi32_pd`.
#[inline]
pub fn _mm_cvtepi32_pd(a: __m128i) -> __m128d {
    unsafe {
        let lo = vget_low_s32(a.s32());
        __m128d::from_f64(vcvtq_f64_s64(vmovl_s32(lo)))
    }
}

/// Convert two `f64` to two `f32` in the low lanes, zero the high two.
/// Matches `_mm_cvtpd_ps`.
#[inline]
pub fn _mm_cvtpd_ps(a: __m128d) -> __m128 {
    unsafe {
        let lo = vcvt_f32_f64(a.f64());
        __m128::from_f32(vcombine_f32(lo, vdup_n_f32(0.0)))
    }
}

/// Convert two `f32` (low lanes) to two `f64`. Matches `_mm_cvtps_pd`.
#[inline]
pub fn _mm_cvtps_pd(a: __m128) -> __m128d {
    unsafe {
        let lo = vget_low_f32(a.f32());
        __m128d::from_f64(vcvt_f64_f32(lo))
    }
}

/// Convert two `f64` to two `i32` in the low lanes using the MXCSR rounding mode.
///
/// The default mode rounds to nearest, ties to even. Out-of-range and NaN become
/// `i32::MIN`. Matches `_mm_cvtpd_epi32`.
#[inline]
pub fn _mm_cvtpd_epi32(a: __m128d) -> __m128i {
    let lo = cvtd_s32(round_cur(_mm_cvtsd_f64(a)));
    let hi = cvtd_s32(round_cur(unsafe { vgetq_lane_f64(a.f64(), 1) }));
    _mm_set_epi32(0, 0, hi, lo)
}

/// Convert two `f64` to two `i32` in the low lanes by truncation.
///
/// Out-of-range and NaN become `i32::MIN`. Matches `_mm_cvttpd_epi32`.
#[inline]
pub fn _mm_cvttpd_epi32(a: __m128d) -> __m128i {
    let lo = cvtd_s32(_mm_cvtsd_f64(a));
    let hi = cvtd_s32(unsafe { vgetq_lane_f64(a.f64(), 1) });
    _mm_set_epi32(0, 0, hi, lo)
}

/// Convert lane 0 of `a` to `i32` using the MXCSR rounding mode.
/// Matches `_mm_cvtsd_si32`.
#[inline]
pub fn _mm_cvtsd_si32(a: __m128d) -> i32 {
    cvtd_s32(round_cur(_mm_cvtsd_f64(a)))
}

/// Convert lane 0 of `a` to `i32` by truncation. Matches `_mm_cvttsd_si32`.
#[inline]
pub fn _mm_cvttsd_si32(a: __m128d) -> i32 {
    cvtd_s32(_mm_cvtsd_f64(a))
}

/// Round an `f64` to an integral `f64` using the current FPCR rounding mode.
///
/// `vrndi_f64` reads the FPCR RMode field, so this tracks the MXCSR mode set
/// through `_MM_SET_ROUNDING_MODE`. NaN and infinity pass through unchanged.
#[inline]
fn round_cur(v: f64) -> f64 {
    unsafe { vget_lane_f64(vrndi_f64(vdup_n_f64(v)), 0) }
}

/// Range-check an integral-valued `f64` and cast to `i32` with the x86 rule.
///
/// The caller has already applied any rounding. NaN, infinity, and values
/// outside `i32` range give `i32::MIN`. In range, the `as` cast is exact because
/// the input is integral. Matches the x86 "integer indefinite" result.
#[inline]
fn cvtd_s32(v: f64) -> i32 {
    if v.is_nan() || v.is_infinite() || !(-2147483648.0..2147483648.0).contains(&v) {
        i32::MIN
    } else {
        v as i32
    }
}

/// Convert lane 0 of `a` (`f64`) to `f32` in lane 0, upper lanes from `b`.
/// Matches `_mm_cvtsd_ss`.
///
/// `vcvt_f32_f64` narrows using the current FPCR rounding mode, so the low bit
/// matches x86 `cvtsd2ss` under directed rounding.
#[inline]
pub fn _mm_cvtsd_ss(a: __m128, b: __m128d) -> __m128 {
    let v = unsafe { vget_lane_f32(vcvt_f32_f64(vdupq_n_f64(_mm_cvtsd_f64(b))), 0) };
    unsafe { __m128::from_f32(vsetq_lane_f32(v, a.f32(), 0)) }
}

/// Convert lane 0 of `b` (`f32`) to `f64` in lane 0, high lane from `a`.
/// Matches `_mm_cvtss_sd`.
#[inline]
pub fn _mm_cvtss_sd(a: __m128d, b: __m128) -> __m128d {
    let v = crate::sse::_mm_cvtss_f32(b) as f64;
    unsafe { __m128d::from_f64(vsetq_lane_f64(v, a.f64(), 0)) }
}

/// Set lane 0 to `b` converted to `f64`, high lane from `a`. Matches `_mm_cvtsi32_sd`.
#[inline]
pub fn _mm_cvtsi32_sd(a: __m128d, b: i32) -> __m128d {
    unsafe { __m128d::from_f64(vsetq_lane_f64(b as f64, a.f64(), 0)) }
}

/// Set lane 0 to `b` converted to `f64`, high lane from `a`. Matches `_mm_cvtsi64_sd`.
#[inline]
pub fn _mm_cvtsi64_sd(a: __m128d, b: i64) -> __m128d {
    unsafe { __m128d::from_f64(vsetq_lane_f64(b as f64, a.f64(), 0)) }
}
