//! IEEE-754 edge cases: NaN, infinity, signed zero, and conversion saturation.

mod common;

use common::*;
use sse2neon::*;

fn qnan() -> f32 {
    f32::from_bits(0x7FC0_0000)
}

#[test]
fn nan_propagates_through_arithmetic() {
    let a = _mm_set1_ps(qnan());
    let b = _mm_set1_ps(1.0);
    assert!(f32x4(_mm_add_ps(a, b))[0].is_nan());
    assert!(f32x4(_mm_sub_ps(a, b))[0].is_nan());
    assert!(f32x4(_mm_mul_ps(a, b))[0].is_nan());
    assert!(f32x4(_mm_div_ps(a, b))[0].is_nan());
}

#[test]
fn infinity_arithmetic() {
    let inf = _mm_set1_ps(f32::INFINITY);
    let one = _mm_set1_ps(1.0);
    assert_eq!(f32x4(_mm_add_ps(inf, one))[0], f32::INFINITY);
    // inf - inf = NaN
    assert!(f32x4(_mm_sub_ps(inf, inf))[0].is_nan());
    // 0 * inf = NaN
    let zero = _mm_set1_ps(0.0);
    assert!(f32x4(_mm_mul_ps(zero, inf))[0].is_nan());
    // inf / inf = NaN
    assert!(f32x4(_mm_div_ps(inf, inf))[0].is_nan());
}

#[test]
fn signed_zero_preserved() {
    let neg = _mm_set1_ps(-0.0);
    // -0.0 keeps its sign bit through a store.
    assert_eq!(f32x4(neg)[0].to_bits(), 0x8000_0000);
    // +0.0 + -0.0 = +0.0
    let pos = _mm_set1_ps(0.0);
    assert_eq!(f32x4(_mm_add_ps(pos, neg))[0].to_bits(), 0);
}

#[test]
fn signed_zero_in_double() {
    let neg = _mm_set1_pd(-0.0);
    assert_eq!(f64x2(neg)[0].to_bits(), 0x8000_0000_0000_0000);
}

#[test]
fn cmpord_and_cmpunord() {
    let nan = _mm_set1_ps(qnan());
    let one = _mm_set1_ps(1.0);
    // ordered: false where any NaN
    let ord = i32x4(_mm_castps_si128(_mm_cmpord_ps(nan, one)));
    assert_eq!(ord[0], 0);
    let ord2 = i32x4(_mm_castps_si128(_mm_cmpord_ps(one, one)));
    assert_eq!(ord2[0], -1);
    // unordered: true where any NaN
    let un = i32x4(_mm_castps_si128(_mm_cmpunord_ps(nan, one)));
    assert_eq!(un[0], -1);
}

#[test]
fn cmpeq_with_nan_is_false() {
    let nan = _mm_set1_ps(qnan());
    let eq = i32x4(_mm_castps_si128(_mm_cmpeq_ps(nan, nan)));
    assert_eq!(eq[0], 0);
    let neq = i32x4(_mm_castps_si128(_mm_cmpneq_ps(nan, nan)));
    assert_eq!(neq[0], -1);
}

#[test]
fn cvttps_epi32_special_values() {
    // NaN, +Inf, -Inf, and overflow all map to i32::MIN (integer indefinite).
    let nan = _mm_set1_ps(qnan());
    assert_eq!(i32x4(_mm_cvttps_epi32(nan))[0], i32::MIN);
    let pinf = _mm_set1_ps(f32::INFINITY);
    assert_eq!(i32x4(_mm_cvttps_epi32(pinf))[0], i32::MIN);
    let ninf = _mm_set1_ps(f32::NEG_INFINITY);
    assert_eq!(i32x4(_mm_cvttps_epi32(ninf))[0], i32::MIN);
    let over = _mm_set1_ps(3.0e9);
    assert_eq!(i32x4(_mm_cvttps_epi32(over))[0], i32::MIN);
    // in-range truncates toward zero
    let ok = _mm_set1_ps(-2.9);
    assert_eq!(i32x4(_mm_cvttps_epi32(ok))[0], -2);
}

#[test]
fn cvtps_epi32_rounds_half_to_even() {
    let a = _mm_set_ps(4.5, 3.5, 2.5, 0.5);
    // half to even: 0.5->0, 2.5->2, 3.5->4, 4.5->4
    assert_eq!(i32x4(_mm_cvtps_epi32(a)), [0, 2, 4, 4]);
}

#[test]
fn sqrt_of_negative_is_nan() {
    let a = _mm_set1_ps(-4.0);
    assert!(f32x4(_mm_sqrt_ps(a))[0].is_nan());
}

#[test]
fn round_special_values() {
    let inf = _mm_set1_ps(f32::INFINITY);
    assert_eq!(f32x4(_mm_floor_ps(inf))[0], f32::INFINITY);
    assert_eq!(f32x4(_mm_ceil_ps(inf))[0], f32::INFINITY);
    let nan = _mm_set1_ps(qnan());
    assert!(f32x4(_mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(nan))[0].is_nan());
}
