//! Public constant macros and enums.
//!
//! These mirror the `_MM_*` and `_SIDD_*` values from the Intel headers. Shuffle
//! immediates use [`_MM_SHUFFLE`] to pack four 2-bit lane selectors.
//!
//! The `_MM_HINT_*` and `_SIDD_*` values exist for source parity. Their matching
//! functions (`_mm_prefetch` and the `_mm_cmpistr*`/`_mm_cmpestr*` string
//! compares) are not provided, so these constants have no effect here.

/// Pack four 2-bit lane selectors into a shuffle immediate.
///
/// `fp3` occupies bits 6-7, `fp2` bits 4-5, `fp1` bits 2-3, `fp0` bits 0-1.
/// Use with `_mm_shuffle_ps`, `_mm_shuffle_epi32`, and related functions.
#[inline]
#[must_use]
pub const fn _MM_SHUFFLE(fp3: u32, fp2: u32, fp1: u32, fp0: u32) -> i32 {
    ((fp3 << 6) | (fp2 << 4) | (fp1 << 2) | fp0) as i32
}

/// Pack two 1-bit lane selectors into a shuffle immediate for `_mm_shuffle_pd`.
#[inline]
#[must_use]
pub const fn _MM_SHUFFLE2(x: u32, y: u32) -> i32 {
    ((x << 1) | y) as i32
}

/// Round to nearest, ties to even.
pub const _MM_FROUND_TO_NEAREST_INT: i32 = 0x00;
/// Round toward negative infinity.
pub const _MM_FROUND_TO_NEG_INF: i32 = 0x01;
/// Round toward positive infinity.
pub const _MM_FROUND_TO_POS_INF: i32 = 0x02;
/// Round toward zero (truncate).
pub const _MM_FROUND_TO_ZERO: i32 = 0x03;
/// Use the current MXCSR rounding mode.
pub const _MM_FROUND_CUR_DIRECTION: i32 = 0x04;
/// Suppress floating-point exceptions.
pub const _MM_FROUND_NO_EXC: i32 = 0x08;
/// Raise floating-point exceptions (the default, value zero).
pub const _MM_FROUND_RAISE_EXC: i32 = 0x00;
/// Round to nearest and raise exceptions.
pub const _MM_FROUND_NINT: i32 = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_RAISE_EXC;
/// Round toward negative infinity and raise exceptions.
pub const _MM_FROUND_FLOOR: i32 = _MM_FROUND_TO_NEG_INF | _MM_FROUND_RAISE_EXC;
/// Round toward positive infinity and raise exceptions.
pub const _MM_FROUND_CEIL: i32 = _MM_FROUND_TO_POS_INF | _MM_FROUND_RAISE_EXC;
/// Round toward zero and raise exceptions.
pub const _MM_FROUND_TRUNC: i32 = _MM_FROUND_TO_ZERO | _MM_FROUND_RAISE_EXC;
/// Round with the current direction and raise exceptions.
pub const _MM_FROUND_RINT: i32 = _MM_FROUND_CUR_DIRECTION | _MM_FROUND_RAISE_EXC;
/// Round with the current direction and suppress exceptions.
pub const _MM_FROUND_NEARBYINT: i32 = _MM_FROUND_CUR_DIRECTION | _MM_FROUND_NO_EXC;

/// MXCSR rounding mode: nearest.
pub const _MM_ROUND_NEAREST: u32 = 0x0000;
/// MXCSR rounding mode: toward negative infinity.
pub const _MM_ROUND_DOWN: u32 = 0x2000;
/// MXCSR rounding mode: toward positive infinity.
pub const _MM_ROUND_UP: u32 = 0x4000;
/// MXCSR rounding mode: toward zero.
pub const _MM_ROUND_TOWARD_ZERO: u32 = 0x6000;
/// Mask selecting the MXCSR rounding-mode bits.
pub const _MM_ROUND_MASK: u32 = 0x6000;

/// Mask selecting the flush-to-zero bit.
pub const _MM_FLUSH_ZERO_MASK: u32 = 0x8000;
/// Flush-to-zero enabled.
pub const _MM_FLUSH_ZERO_ON: u32 = 0x8000;
/// Flush-to-zero disabled.
pub const _MM_FLUSH_ZERO_OFF: u32 = 0x0000;

/// Mask selecting the denormals-are-zero bit.
pub const _MM_DENORMALS_ZERO_MASK: u32 = 0x0040;
/// Denormals-are-zero enabled.
pub const _MM_DENORMALS_ZERO_ON: u32 = 0x0040;
/// Denormals-are-zero disabled.
pub const _MM_DENORMALS_ZERO_OFF: u32 = 0x0000;

/// Cache hint: non-temporal.
pub const _MM_HINT_NTA: i32 = 0;
/// Cache hint: prefetch to all levels.
pub const _MM_HINT_T0: i32 = 1;
/// Cache hint: prefetch to L2 and higher.
pub const _MM_HINT_T1: i32 = 2;
/// Cache hint: prefetch to L3 and higher.
pub const _MM_HINT_T2: i32 = 3;

// SSE4.2 string-compare control bits. Grouped by field per the Intel encoding.

/// String data format: unsigned bytes.
pub const _SIDD_UBYTE_OPS: i32 = 0x00;
/// String data format: unsigned words.
pub const _SIDD_UWORD_OPS: i32 = 0x01;
/// String data format: signed bytes.
pub const _SIDD_SBYTE_OPS: i32 = 0x02;
/// String data format: signed words.
pub const _SIDD_SWORD_OPS: i32 = 0x03;

/// Comparison: any element of b equals any element of a.
pub const _SIDD_CMP_EQUAL_ANY: i32 = 0x00;
/// Comparison: elements of b fall inside ranges from a.
pub const _SIDD_CMP_RANGES: i32 = 0x04;
/// Comparison: elements compare equal lane by lane.
pub const _SIDD_CMP_EQUAL_EACH: i32 = 0x08;
/// Comparison: substring search.
pub const _SIDD_CMP_EQUAL_ORDERED: i32 = 0x0C;

/// Polarity: keep the match bits as computed.
pub const _SIDD_POSITIVE_POLARITY: i32 = 0x00;
/// Polarity: negate the match bits.
pub const _SIDD_NEGATIVE_POLARITY: i32 = 0x10;
/// Polarity: keep, masked to valid b elements.
pub const _SIDD_MASKED_POSITIVE_POLARITY: i32 = 0x20;
/// Polarity: negate, masked to valid b elements.
pub const _SIDD_MASKED_NEGATIVE_POLARITY: i32 = 0x30;

/// Index output: least significant set bit.
pub const _SIDD_LEAST_SIGNIFICANT: i32 = 0x00;
/// Index output: most significant set bit.
pub const _SIDD_MOST_SIGNIFICANT: i32 = 0x40;

/// Mask output: one bit per element.
pub const _SIDD_BIT_MASK: i32 = 0x00;
/// Mask output: one full element per bit.
pub const _SIDD_UNIT_MASK: i32 = 0x40;
