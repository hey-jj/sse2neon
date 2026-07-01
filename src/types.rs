//! Vector types and zero-cost reinterpret helpers.
//!
//! Each type wraps a NEON register. The wrappers are `Copy` and transparent, so
//! they lower to plain registers. Reinterpret helpers bit-cast between NEON lane
//! views. They emit no instructions.

use core::arch::aarch64::*;

/// 128-bit vector of four `f32` lanes.
///
/// Matches the Intel `__m128` type. Lane 0 is the least significant.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct __m128(pub(crate) float32x4_t);

/// 128-bit vector of two `f64` lanes.
///
/// Matches the Intel `__m128d` type. Lane 0 is the least significant.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct __m128d(pub(crate) float64x2_t);

/// 128-bit integer vector.
///
/// Matches the Intel `__m128i` type. Holds sixteen `i8`, eight `i16`, four
/// `i32`, or two `i64` lanes depending on the operation.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct __m128i(pub(crate) int64x2_t);

/// 64-bit integer vector.
///
/// Matches the Intel `__m64` MMX type. Holds eight `i8`, four `i16`, or two
/// `i32` lanes depending on the operation.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
#[allow(non_camel_case_types)]
pub struct __m64(pub(crate) int64x1_t);

// Reinterpret helpers. Every cast below is a bit-preserving view change with no
// runtime cost. They centralize the transmute so intrinsic bodies stay short.

impl __m128 {
    #[inline(always)]
    pub(crate) fn f32(self) -> float32x4_t {
        self.0
    }
    #[inline(always)]
    pub(crate) fn from_f32(v: float32x4_t) -> Self {
        __m128(v)
    }
    #[inline(always)]
    pub(crate) fn u32(self) -> uint32x4_t {
        unsafe { vreinterpretq_u32_f32(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u32(v: uint32x4_t) -> Self {
        __m128(unsafe { vreinterpretq_f32_u32(v) })
    }
}

impl __m128i {
    #[inline(always)]
    pub(crate) fn s64(self) -> int64x2_t {
        self.0
    }
    #[inline(always)]
    pub(crate) fn from_s64(v: int64x2_t) -> Self {
        __m128i(v)
    }
    #[inline(always)]
    pub(crate) fn u64(self) -> uint64x2_t {
        unsafe { vreinterpretq_u64_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u64(v: uint64x2_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_u64(v) })
    }
    #[inline(always)]
    pub(crate) fn s8(self) -> int8x16_t {
        unsafe { vreinterpretq_s8_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_s8(v: int8x16_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_s8(v) })
    }
    #[inline(always)]
    pub(crate) fn u8(self) -> uint8x16_t {
        unsafe { vreinterpretq_u8_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u8(v: uint8x16_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_u8(v) })
    }
    #[inline(always)]
    pub(crate) fn s16(self) -> int16x8_t {
        unsafe { vreinterpretq_s16_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_s16(v: int16x8_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_s16(v) })
    }
    #[inline(always)]
    pub(crate) fn u16(self) -> uint16x8_t {
        unsafe { vreinterpretq_u16_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u16(v: uint16x8_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_u16(v) })
    }
    #[inline(always)]
    pub(crate) fn s32(self) -> int32x4_t {
        unsafe { vreinterpretq_s32_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_s32(v: int32x4_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_s32(v) })
    }
    #[inline(always)]
    pub(crate) fn u32(self) -> uint32x4_t {
        unsafe { vreinterpretq_u32_s64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u32(v: uint32x4_t) -> Self {
        __m128i(unsafe { vreinterpretq_s64_u32(v) })
    }
}

impl __m128d {
    #[inline(always)]
    pub(crate) fn f64(self) -> float64x2_t {
        self.0
    }
    #[inline(always)]
    pub(crate) fn from_f64(v: float64x2_t) -> Self {
        __m128d(v)
    }
    #[inline(always)]
    pub(crate) fn u64(self) -> uint64x2_t {
        unsafe { vreinterpretq_u64_f64(self.0) }
    }
    #[inline(always)]
    pub(crate) fn from_u64(v: uint64x2_t) -> Self {
        __m128d(unsafe { vreinterpretq_f64_u64(v) })
    }
}

/// Cast between vector types without changing bits.
///
/// The result holds the same 128 bits reinterpreted as the destination type.
/// Matches `_mm_castps_si128`.
#[inline]
pub fn _mm_castps_si128(a: __m128) -> __m128i {
    __m128i(unsafe { vreinterpretq_s64_f32(a.0) })
}

/// Cast between vector types without changing bits. Matches `_mm_castsi128_ps`.
#[inline]
pub fn _mm_castsi128_ps(a: __m128i) -> __m128 {
    __m128(unsafe { vreinterpretq_f32_s64(a.0) })
}

/// Cast between vector types without changing bits. Matches `_mm_castps_pd`.
#[inline]
pub fn _mm_castps_pd(a: __m128) -> __m128d {
    __m128d(unsafe { vreinterpretq_f64_f32(a.0) })
}

/// Cast between vector types without changing bits. Matches `_mm_castpd_ps`.
#[inline]
pub fn _mm_castpd_ps(a: __m128d) -> __m128 {
    __m128(unsafe { vreinterpretq_f32_f64(a.0) })
}

/// Cast between vector types without changing bits. Matches `_mm_castsi128_pd`.
#[inline]
pub fn _mm_castsi128_pd(a: __m128i) -> __m128d {
    __m128d(unsafe { vreinterpretq_f64_s64(a.0) })
}

/// Cast between vector types without changing bits. Matches `_mm_castpd_si128`.
#[inline]
pub fn _mm_castpd_si128(a: __m128d) -> __m128i {
    __m128i(unsafe { vreinterpretq_s64_f64(a.0) })
}
