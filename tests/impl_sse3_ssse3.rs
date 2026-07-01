//! SSE3 and SSSE3 conformance.

mod common;

use common::*;
use sse2neon::*;

#[test]
fn hadd_ps() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
    // result: a0+a1, a2+a3, b0+b1, b2+b3 = 3, 7, 11, 15
    assert_eq!(f32x4(_mm_hadd_ps(a, b)), [3.0, 7.0, 11.0, 15.0]);
}

#[test]
fn hsub_ps() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
    // result: a0-a1, a2-a3, b0-b1, b2-b3 = -1, -1, -1, -1
    assert_eq!(f32x4(_mm_hsub_ps(a, b)), [-1.0, -1.0, -1.0, -1.0]);
}

#[test]
fn addsub_ps() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(1.0, 1.0, 1.0, 1.0);
    // lane0=a0-b0, lane1=a1+b1, lane2=a2-b2, lane3=a3+b3 = 0, 3, 2, 5
    assert_eq!(f32x4(_mm_addsub_ps(a, b)), [0.0, 3.0, 2.0, 5.0]);
}

#[test]
fn hadd_pd_and_addsub_pd() {
    let a = _mm_set_pd(2.0, 1.0);
    let b = _mm_set_pd(4.0, 3.0);
    assert_eq!(f64x2(_mm_hadd_pd(a, b)), [3.0, 7.0]);
    // addsub_pd: lane0=a0-b0, lane1=a1+b1 = -2, 6
    assert_eq!(f64x2(_mm_addsub_pd(a, b)), [-2.0, 6.0]);
}

#[test]
fn movedup_and_duplicates() {
    let a = _mm_set_pd(2.0, 1.0);
    assert_eq!(f64x2(_mm_movedup_pd(a)), [1.0, 1.0]);
    let f = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    assert_eq!(f32x4(_mm_movehdup_ps(f)), [2.0, 2.0, 4.0, 4.0]);
    assert_eq!(f32x4(_mm_moveldup_ps(f)), [1.0, 1.0, 3.0, 3.0]);
}

// --- SSSE3 ---

#[test]
fn abs_variants() {
    let a = _mm_set_epi8(-8, 7, -6, 5, -4, 3, -2, 1, -8, 7, -6, 5, -4, 3, -2, 1);
    let r = i8x16(_mm_abs_epi8(a));
    assert_eq!(r[0], 1);
    assert_eq!(r[1], 2);
    let b = _mm_set_epi16(-4, 3, -2, 1, -4, 3, -2, 1);
    assert_eq!(i16x8(_mm_abs_epi16(b))[0], 1);
    assert_eq!(i16x8(_mm_abs_epi16(b))[1], 2);
    let c = _mm_set_epi32(-2, 1, -2, 1);
    let cr = i32x4(_mm_abs_epi32(c));
    assert_eq!(cr[0], 1);
    assert_eq!(cr[1], 2);
}

#[test]
fn sign_epi8() {
    let a = _mm_set1_epi8(5);
    // b positive keeps, zero zeroes, negative negates
    let b = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1);
    let r = i8x16(_mm_sign_epi8(a, b));
    assert_eq!(r[0], 5); // b=1 keep
    assert_eq!(r[1], 0); // b=0 zero
    assert_eq!(r[2], -5); // b=-1 negate
}

#[test]
fn shuffle_epi8_indexes_and_zeroes() {
    let a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let idx = _mm_set_epi8(-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    let r = u8x16(_mm_shuffle_epi8(a, idx));
    // lane0 index=15 -> a[15]=15
    assert_eq!(r[0], 15);
    // lane15 index=-1 (top bit set) -> 0
    assert_eq!(r[15], 0);
}

#[test]
fn hadd_epi16() {
    let a = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    let b = _mm_set_epi16(80, 70, 60, 50, 40, 30, 20, 10);
    // a: 1+2,3+4,5+6,7+8 then b: 10+20,30+40,50+60,70+80
    assert_eq!(
        i16x8(_mm_hadd_epi16(a, b)),
        [3, 7, 11, 15, 30, 70, 110, 150]
    );
}

#[test]
fn maddubs_epi16() {
    // a unsigned, b signed. a=[2,2,...], b=[3,-4,...]
    let a = _mm_set1_epi8(2);
    let b = _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -4, 3);
    let r = i16x8(_mm_maddubs_epi16(a, b));
    // lane0 = 2*3 + 2*(-4) = 6 - 8 = -2
    assert_eq!(r[0], -2);
}

#[test]
fn mulhrs_epi16() {
    let a = _mm_set1_epi16(0x4000);
    let b = _mm_set1_epi16(0x4000);
    // (0x4000 * 0x4000 + 0x4000) >> 15 = (0x10000000 + 0x4000) >> 15 = 0x2000
    assert_eq!(i16x8(_mm_mulhrs_epi16(a, b))[0], 0x2000);
}

#[test]
fn alignr_epi8() {
    let a = _mm_set_epi8(
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    );
    let b = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    // concat b:a, shift right 3 bytes. low lane picks b[3]=3
    let r = u8x16(_mm_alignr_epi8::<3>(a, b));
    assert_eq!(r[0], 3);
    assert_eq!(r[13], 16); // b[16]=a[0]=16
                           // imm >= 32 -> zero
    assert_eq!(u8x16(_mm_alignr_epi8::<32>(a, b)), [0u8; 16]);
}
