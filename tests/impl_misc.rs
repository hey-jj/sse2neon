//! MXCSR mode control, rounding-mode-aware conversion, and the MMX `__m64`
//! family.

mod common;

use common::*;
use sse2neon::*;

fn m64_words(a: __m64) -> [i32; 2] {
    // Widen the __m64 into the low half of a 128-bit vector, then read lanes.
    let wide = _mm_movpi64_epi64(a);
    let mut out = [0i32; 4];
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, wide) };
    [out[0], out[1]]
}

#[test]
fn m64_set_and_extract() {
    let a = _mm_set_pi32(7, 3);
    assert_eq!(_mm_cvtsi64_si32(a), 3);
    let words = m64_words(a);
    assert_eq!(words, [3, 7]);
}

#[test]
fn m64_arithmetic() {
    let a = _mm_set_pi16(4, 3, 2, 1);
    let b = _mm_set_pi16(40, 30, 20, 10);
    let r = _mm_add_pi16(a, b);
    // Convert to i16 lanes via widening path.
    let wide = _mm_movpi64_epi64(r);
    let mut out = [0i16; 8];
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, wide) };
    assert_eq!(&out[..4], &[11, 22, 33, 44]);
}

#[test]
fn m64_movemask_pi8() {
    let a = _mm_set_pi8(-1, 0, -1, 0, -1, 0, -1, 0);
    // lane0..7 = e0..e7 = 0,-1,0,-1,0,-1,0,-1 -> bits at odd positions = 0b10101010
    assert_eq!(_mm_movemask_pi8(a), 0b1010_1010);
}

#[test]
fn m64_min_max_avg() {
    let a = _mm_set_pi16(1, 2, 3, 4);
    let b = _mm_set_pi16(4, 3, 2, 1);
    let mn = _mm_movpi64_epi64(_mm_min_pi16(a, b));
    let mut out = [0i16; 8];
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, mn) };
    // lane0 min(4,1)=1, lane1 min(3,2)=2, lane2 min(2,3)=2, lane3 min(1,4)=1
    assert_eq!(&out[..4], &[1, 2, 2, 1]);
}

#[test]
fn m64_sad_pu8() {
    let a = _mm_set_pi8(10, 10, 10, 10, 10, 10, 10, 10);
    let b = _mm_set_pi8(3, 3, 3, 3, 3, 3, 3, 3);
    // |10-3|=7 times 8 = 56, in lane 0
    let r = _mm_movpi64_epi64(_mm_sad_pu8(a, b));
    let mut out = [0u16; 8];
    unsafe { _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, r) };
    assert_eq!(out[0], 56);
}

// --- MXCSR round-trip and rounding-mode-aware conversion ---

#[test]
fn rounding_mode_roundtrip() {
    let original = _MM_GET_ROUNDING_MODE();
    for mode in [
        _MM_ROUND_NEAREST,
        _MM_ROUND_DOWN,
        _MM_ROUND_UP,
        _MM_ROUND_TOWARD_ZERO,
    ] {
        _MM_SET_ROUNDING_MODE(mode);
        assert_eq!(_MM_GET_ROUNDING_MODE(), mode);
    }
    _MM_SET_ROUNDING_MODE(original);
    assert_eq!(_MM_GET_ROUNDING_MODE(), original);
}

#[test]
fn flush_zero_roundtrip() {
    let original = _MM_GET_FLUSH_ZERO_MODE();
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    assert_eq!(_MM_GET_FLUSH_ZERO_MODE(), _MM_FLUSH_ZERO_ON);
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
    assert_eq!(_MM_GET_FLUSH_ZERO_MODE(), _MM_FLUSH_ZERO_OFF);
    _MM_SET_FLUSH_ZERO_MODE(original);
}

#[test]
fn getcsr_setcsr_roundtrip() {
    let original = _mm_getcsr();
    _mm_setcsr(_MM_ROUND_UP | _MM_FLUSH_ZERO_ON);
    assert_eq!(_mm_getcsr() & _MM_ROUND_MASK, _MM_ROUND_UP);
    assert_eq!(_mm_getcsr() & _MM_FLUSH_ZERO_MASK, _MM_FLUSH_ZERO_ON);
    _mm_setcsr(original);
    assert_eq!(_mm_getcsr(), original);
}

#[test]
fn cvtps_epi32_respects_rounding_mode() {
    let original = _MM_GET_ROUNDING_MODE();
    let a = _mm_set1_ps(2.5);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    assert_eq!(i32x4(_mm_cvtps_epi32(a))[0], 2); // ties to even

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    assert_eq!(i32x4(_mm_cvtps_epi32(a))[0], 2);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    assert_eq!(i32x4(_mm_cvtps_epi32(a))[0], 3);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    let neg = _mm_set1_ps(-2.7);
    assert_eq!(i32x4(_mm_cvtps_epi32(neg))[0], -2);

    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn cvtpd_epi32_respects_rounding_mode() {
    let original = _MM_GET_ROUNDING_MODE();
    let a = _mm_set1_pd(2.5);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    assert_eq!(i32x4(_mm_cvtpd_epi32(a))[0], 2); // ties to even

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    assert_eq!(i32x4(_mm_cvtpd_epi32(a))[0], 2);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    assert_eq!(i32x4(_mm_cvtpd_epi32(a))[0], 3);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
    assert_eq!(i32x4(_mm_cvtpd_epi32(_mm_set1_pd(-2.7)))[0], -2);

    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn cvtsd_si32_respects_rounding_mode() {
    let original = _MM_GET_ROUNDING_MODE();
    let a = _mm_set_pd(9.0, 2.5);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    assert_eq!(_mm_cvtsd_si32(a), 2); // ties to even

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    assert_eq!(_mm_cvtsd_si32(a), 3);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    assert_eq!(_mm_cvtsd_si32(a), 2);

    // Truncating variant ignores the mode.
    assert_eq!(_mm_cvttsd_si32(a), 2);

    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn cvtsd_ss_respects_rounding_mode() {
    // 1 + 2^-24 has no exact f32. Round-up and round-down give adjacent f32
    // values, so the low bit of the narrowing exposes the mode.
    let original = _MM_GET_ROUNDING_MODE();
    let val = 1.0 + 2.0f64.powi(-24);
    let upper = _mm_set_ps(4.0, 3.0, 2.0, 9.0);
    let src = _mm_set1_pd(val);

    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    let up = f32x4(_mm_cvtsd_ss(upper, src))[0];

    _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
    let down = f32x4(_mm_cvtsd_ss(upper, src))[0];

    assert!(up > down, "round-up {up} must exceed round-down {down}");
    assert_eq!(down, 1.0f32); // truncates the extra bit
                              // Upper three lanes come from `a` unchanged.
    let r = f32x4(_mm_cvtsd_ss(upper, src));
    assert_eq!(&r[1..], &[2.0, 3.0, 4.0]);

    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn setcsr_honors_daz_bit() {
    let original = _mm_getcsr();

    // DAZ set, FTZ clear: FPCR bit 24 must still turn on flush.
    _mm_setcsr(_MM_DENORMALS_ZERO_ON);
    assert_eq!(_MM_GET_FLUSH_ZERO_MODE(), _MM_FLUSH_ZERO_ON);
    assert_eq!(_MM_GET_DENORMALS_ZERO_MODE(), _MM_DENORMALS_ZERO_ON);

    // A DAZ-only word round-trips through getcsr without losing the bit.
    let word = _mm_getcsr();
    assert_eq!(word & _MM_DENORMALS_ZERO_MASK, _MM_DENORMALS_ZERO_ON);

    // Neither bit set clears flush.
    _mm_setcsr(0);
    assert_eq!(_MM_GET_FLUSH_ZERO_MODE(), _MM_FLUSH_ZERO_OFF);

    _mm_setcsr(original);
}

#[test]
fn scalar_sd_upper_lane_copy() {
    let a = _mm_set_pd(2.0, 1.0);
    let b = _mm_set_pd(4.0, 3.0);

    assert_eq!(f64x2(_mm_add_sd(a, b)), [4.0, 2.0]); // lane0 1+3, high from a
    assert_eq!(f64x2(_mm_sub_sd(a, b)), [-2.0, 2.0]);
    assert_eq!(f64x2(_mm_mul_sd(a, b)), [3.0, 2.0]);
    assert_eq!(f64x2(_mm_div_sd(a, b)), [1.0 / 3.0, 2.0]);
    assert_eq!(f64x2(_mm_move_sd(a, b)), [3.0, 2.0]); // lane0 from b, high from a

    let c = _mm_set_pd(40.0, 2.7);
    assert_eq!(f64x2(_mm_round_sd::<_MM_FROUND_TO_ZERO>(a, c)), [2.0, 2.0]);
    assert_eq!(f64x2(_mm_floor_sd(a, c)), [2.0, 2.0]);
    assert_eq!(f64x2(_mm_ceil_sd(a, c)), [3.0, 2.0]);
}

#[test]
fn invalid_rounding_mode_falls_back_to_toward_zero() {
    let original = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(0x1234); // not a valid mode
    assert_eq!(_MM_GET_ROUNDING_MODE(), _MM_ROUND_TOWARD_ZERO);
    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn little_endian_only() {
    // The library assumes little-endian lane-to-memory order. Confirm the host
    // matches so the lane readers are valid.
    const { assert!(cfg!(target_endian = "little")) }
}
