//! NaN handling across arithmetic, comparisons, bitwise ops, and conversions.

mod common;

use common::*;
use sse2neon::*;

fn qnan() -> f32 {
    f32::from_bits(0x7FC0_0000)
}

fn qnan64() -> f64 {
    f64::from_bits(0x7FF8_0000_0000_0000)
}

#[test]
fn nan_first_or_second_operand() {
    let nan = _mm_set1_ps(qnan());
    let x = _mm_set1_ps(3.0);
    // NaN in either position yields NaN for add/sub/mul/div.
    for (a, b) in [(nan, x), (x, nan)] {
        assert!(f32x4(_mm_add_ps(a, b))[0].is_nan());
        assert!(f32x4(_mm_sub_ps(a, b))[0].is_nan());
        assert!(f32x4(_mm_mul_ps(a, b))[0].is_nan());
        assert!(f32x4(_mm_div_ps(a, b))[0].is_nan());
    }
}

#[test]
fn nan_in_double_arithmetic() {
    let nan = _mm_set1_pd(qnan64());
    let x = _mm_set1_pd(2.0);
    assert!(f64x2(_mm_add_pd(nan, x))[0].is_nan());
    assert!(f64x2(_mm_mul_pd(x, nan))[0].is_nan());
}

#[test]
fn all_compares_with_nan() {
    let nan = _mm_set1_ps(qnan());
    let x = _mm_set1_ps(1.0);
    // Every ordered comparison is false with a NaN operand.
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpeq_ps(nan, x)))[0], 0);
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmplt_ps(nan, x)))[0], 0);
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmple_ps(nan, x)))[0], 0);
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpgt_ps(nan, x)))[0], 0);
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpge_ps(nan, x)))[0], 0);
    // Not-equal is true with a NaN operand.
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpneq_ps(nan, x)))[0], -1);
    // The negated forms are true with a NaN operand.
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpnlt_ps(nan, x)))[0], -1);
    assert_eq!(i32x4(_mm_castps_si128(_mm_cmpnge_ps(nan, x)))[0], -1);
}

#[test]
fn nan_payload_survives_bitwise_and_shuffle() {
    // A NaN bit pattern passes unchanged through a bitwise identity and a
    // shuffle that only moves lanes.
    let payload = f32::from_bits(0x7FC1_2345);
    let a = _mm_set1_ps(payload);
    let ident = _mm_castsi128_ps(_mm_set1_epi32(-1));
    let anded = f32x4(_mm_and_ps(a, ident));
    assert_eq!(anded[0].to_bits(), 0x7FC1_2345);
    let shuffled = f32x4(_mm_shuffle_ps::<{ _MM_SHUFFLE(3, 2, 1, 0) }>(a, a));
    assert_eq!(shuffled[0].to_bits(), 0x7FC1_2345);
}

#[test]
fn nan_generation() {
    // 0/0, inf-inf, inf*0, sqrt(-1), inf/inf all make NaN.
    let zero = _mm_set1_ps(0.0);
    let inf = _mm_set1_ps(f32::INFINITY);
    assert!(f32x4(_mm_div_ps(zero, zero))[0].is_nan());
    assert!(f32x4(_mm_sub_ps(inf, inf))[0].is_nan());
    assert!(f32x4(_mm_mul_ps(inf, zero))[0].is_nan());
    assert!(f32x4(_mm_sqrt_ps(_mm_set1_ps(-1.0)))[0].is_nan());
    assert!(f32x4(_mm_div_ps(inf, inf))[0].is_nan());
}

#[test]
fn cvt_with_nan() {
    let nan = _mm_set1_ps(qnan());
    assert_eq!(i32x4(_mm_cvtps_epi32(nan))[0], i32::MIN);
    assert_eq!(i32x4(_mm_cvttps_epi32(nan))[0], i32::MIN);
    let nan64 = _mm_set1_pd(qnan64());
    assert_eq!(i32x4(_mm_cvtpd_epi32(nan64))[0], i32::MIN);
    assert_eq!(_mm_cvtsd_si32(nan64), i32::MIN);
}

#[test]
fn blendv_with_nan_operands_and_selector() {
    let a = _mm_set1_ps(qnan());
    let b = _mm_set1_ps(2.0);
    // Selector positive picks a, negative picks b. NaN passes through unchanged.
    let mask_pos = _mm_set1_ps(1.0);
    let r = f32x4(_mm_blendv_ps(a, b, mask_pos));
    assert!(r[0].is_nan()); // sign bit clear -> a
    let mask_neg = _mm_set1_ps(-1.0);
    let r2 = f32x4(_mm_blendv_ps(a, b, mask_neg));
    assert_eq!(r2[0], 2.0); // sign bit set -> b
}

#[test]
fn comi_with_nan() {
    let nan = _mm_set_ss(qnan());
    let x = _mm_set_ss(1.0);
    // Ordered compares return 0 with NaN, except not-equal which returns 1.
    assert_eq!(_mm_comieq_ss(nan, x), 0);
    assert_eq!(_mm_comilt_ss(nan, x), 0);
    assert_eq!(_mm_comigt_ss(nan, x), 0);
    assert_eq!(_mm_comineq_ss(nan, x), 1);
}

#[test]
fn min_max_never_invent_nan_from_finite_inputs() {
    // The default min/max path must not fabricate a NaN when both inputs are
    // finite. This is the core of the DIFF_NAN_HANDLING contract.
    let a = _mm_set1_ps(3.0);
    let b = _mm_set1_ps(5.0);
    assert_eq!(f32x4(_mm_min_ps(a, b))[0], 3.0);
    assert_eq!(f32x4(_mm_max_ps(a, b))[0], 5.0);
}

#[test]
fn min_max_default_propagates_nan() {
    // The default (non-precise) path uses the NEON min/max that propagate a NaN
    // operand. The precise path would instead return the finite operand.
    let nan = _mm_set1_ps(qnan());
    let x = _mm_set1_ps(5.0);
    assert!(f32x4(_mm_min_ps(nan, x))[0].is_nan());
    assert!(f32x4(_mm_max_ps(nan, x))[0].is_nan());
}

#[test]
fn min_max_pd_default_propagates_nan() {
    // Double-precision min/max use the NEON default path, which propagates a NaN
    // operand. This diverges from x86 MINPD/MAXPD, which return the second
    // operand on an unordered compare. The choice matches the _ps path.
    let nan = _mm_set1_pd(qnan64());
    let x = _mm_set1_pd(5.0);
    assert!(f64x2(_mm_min_pd(nan, x))[0].is_nan());
    assert!(f64x2(_mm_max_pd(nan, x))[0].is_nan());
}

#[test]
fn min_max_pd_never_invent_nan_from_finite() {
    let a = _mm_set1_pd(3.0);
    let b = _mm_set1_pd(5.0);
    assert_eq!(f64x2(_mm_min_pd(a, b))[0], 3.0);
    assert_eq!(f64x2(_mm_max_pd(a, b))[0], 5.0);
}
