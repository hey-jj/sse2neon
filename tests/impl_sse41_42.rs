//! SSE4.1 and SSE4.2 conformance, plus independent CRC32C and popcnt checks.

mod common;

use common::*;
use sse2neon::*;

#[test]
fn blend_epi16_mask() {
    let a = _mm_set1_epi16(1);
    let b = _mm_set1_epi16(2);
    // imm 0b10101010: lanes 1,3,5,7 from b
    let r = i16x8(_mm_blend_epi16::<0b1010_1010>(a, b));
    assert_eq!(r, [1, 2, 1, 2, 1, 2, 1, 2]);
}

#[test]
fn blend_ps_mask() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    // imm 0b0101: lanes 0 and 2 from b
    let r = f32x4(_mm_blend_ps::<0b0101>(a, b));
    assert_eq!(r, [10.0, 2.0, 30.0, 4.0]);
}

#[test]
fn blendv_epi8_by_sign() {
    let a = _mm_set1_epi8(1);
    let b = _mm_set1_epi8(2);
    let mask = _mm_set_epi8(-1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0);
    let r = i8x16(_mm_blendv_epi8(a, b, mask));
    // negative mask byte selects b; lane0 mask=0 -> a
    assert_eq!(r[0], 1);
    assert_eq!(r[1], 2);
}

#[test]
fn rounding_modes() {
    let a = _mm_set_ps(-2.5, 2.5, 1.4, 1.6);
    let nearest = f32x4(_mm_round_ps::<{ _MM_FROUND_TO_NEAREST_INT }>(a));
    // round half to even: 1.6->2, 1.4->1, 2.5->2, -2.5->-2
    assert_eq!(nearest, [2.0, 1.0, 2.0, -2.0]);
    let floor = f32x4(_mm_floor_ps(a));
    assert_eq!(floor, [1.0, 1.0, 2.0, -3.0]);
    let ceil = f32x4(_mm_ceil_ps(a));
    assert_eq!(ceil, [2.0, 2.0, 3.0, -2.0]);
    let trunc = f32x4(_mm_round_ps::<{ _MM_FROUND_TO_ZERO }>(a));
    assert_eq!(trunc, [1.0, 1.0, 2.0, -2.0]);
}

#[test]
fn round_pd_and_floor_ceil() {
    let a = _mm_set_pd(-1.5, 2.5);
    assert_eq!(f64x2(_mm_floor_pd(a)), [2.0, -2.0]);
    assert_eq!(f64x2(_mm_ceil_pd(a)), [3.0, -1.0]);
    let nearest = f64x2(_mm_round_pd::<{ _MM_FROUND_TO_NEAREST_INT }>(a));
    assert_eq!(nearest, [2.0, -2.0]); // half to even
}

#[test]
fn sign_zero_extend() {
    let a = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
    // sign extend low byte -1 to i16
    assert_eq!(i16x8(_mm_cvtepi8_epi16(a))[0], -1);
    // zero extend low byte 0xFF to u16 = 255
    let b = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
    assert_eq!(u16x8(_mm_cvtepu8_epi16(b))[0], 255);
    let c = _mm_set_epi32(0, 0, 0, -5);
    assert_eq!(i64x2(_mm_cvtepi32_epi64(c))[0], -5);
}

#[test]
fn extended_minmax() {
    let a = _mm_set1_epi8(-100);
    let b = _mm_set1_epi8(50);
    assert_eq!(i8x16(_mm_min_epi8(a, b))[0], -100);
    assert_eq!(i8x16(_mm_max_epi8(a, b))[0], 50);
    let c = _mm_set1_epi32(200000);
    let d = _mm_set1_epi32(100000);
    assert_eq!(u32x4(_mm_min_epu32(c, d))[0], 100000);
    assert_eq!(u32x4(_mm_max_epu32(c, d))[0], 200000);
}

#[test]
fn dot_product_ps() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(5.0, 6.0, 7.0, 8.0);
    // imm 0xFF: all products, all output lanes
    // sum = 1*8 + 2*7 + 3*6 + 4*5 = 8+14+18+20 = 60
    let r = f32x4(_mm_dp_ps::<0xFF>(a, b));
    assert_eq!(r, [60.0, 60.0, 60.0, 60.0]);
    // imm 0x71: products all, output only lane 0
    let r2 = f32x4(_mm_dp_ps::<0xF1>(a, b));
    assert_eq!(r2, [60.0, 0.0, 0.0, 0.0]);
}

#[test]
fn dot_product_pd() {
    let a = _mm_set_pd(2.0, 1.0);
    let b = _mm_set_pd(4.0, 3.0);
    // imm 0x31: both products, output lane 0. sum = 1*3 + 2*4 = 11
    let r = f64x2(_mm_dp_pd::<0x31>(a, b));
    assert_eq!(r, [11.0, 0.0]);
}

#[test]
fn extract_insert() {
    let a = _mm_set_epi32(4, 3, 2, 1);
    assert_eq!(_mm_extract_epi32::<0>(a), 1);
    assert_eq!(_mm_extract_epi32::<3>(a), 4);
    let b = _mm_insert_epi32::<1>(a, 99);
    assert_eq!(i32x4(b), [1, 99, 3, 4]);
    // extract_epi8 zero-extends
    let c = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
    assert_eq!(_mm_extract_epi8::<0>(c), 255);
}

#[test]
fn minpos() {
    let a = _mm_set_epi16(9, 8, 7, 6, 5, 4, 3, 2);
    // min value 2 at index 0
    let r = u16x8(_mm_minpos_epu16(a));
    assert_eq!(r[0], 2);
    assert_eq!(r[1], 0);
    let b = _mm_set_epi16(9, 8, 1, 6, 5, 4, 3, 2);
    // min value 1 at index 5
    let rb = u16x8(_mm_minpos_epu16(b));
    assert_eq!(rb[0], 1);
    assert_eq!(rb[1], 5);
}

#[test]
fn tests_all_ones_zeros_mix() {
    let ones = _mm_set1_epi32(-1);
    let zeros = _mm_setzero_si128();
    assert_eq!(_mm_test_all_ones(ones), 1);
    assert_eq!(_mm_test_all_ones(zeros), 0);
    assert_eq!(_mm_test_all_zeros(zeros, ones), 1);
    assert_eq!(_mm_testz_si128(zeros, ones), 1);
    assert_eq!(_mm_testz_si128(ones, ones), 0);
    assert_eq!(_mm_testc_si128(ones, ones), 1);
    let mixed = _mm_set_epi32(0, 0, -1, -1);
    assert_eq!(_mm_test_mix_ones_zeros(mixed, ones), 1);
}

#[test]
fn cmpeq_and_cmpgt_epi64() {
    let a = _mm_set_epi64x(5, 3);
    let b = _mm_set_epi64x(5, 4);
    assert_eq!(i64x2(_mm_cmpeq_epi64(a, b)), [0, -1]);
    let c = _mm_set_epi64x(10, -5);
    let d = _mm_set_epi64x(1, 1);
    // lane0: -5 > 1? no -> 0. lane1: 10 > 1? yes -> -1
    assert_eq!(i64x2(_mm_cmpgt_epi64(c, d)), [0, -1]);
}

// --- CRC32C independent known-answer tests ---

#[test]
fn crc32c_known_answers() {
    // CRC-32C of "123456789" is 0xE3069283.
    let data = b"123456789";
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = _mm_crc32_u8(crc, b);
    }
    crc ^= 0xFFFF_FFFF;
    assert_eq!(crc, 0xE306_9283);
}

#[test]
fn crc32c_wider_widths_agree() {
    // Feeding bytes one at a time must match feeding a u32 word.
    let bytes = [0x12u8, 0x34, 0x56, 0x78];
    let mut by_byte = 0u32;
    for &b in &bytes {
        by_byte = _mm_crc32_u8(by_byte, b);
    }
    let word = u32::from_le_bytes(bytes);
    let by_word = _mm_crc32_u32(0, word);
    assert_eq!(by_byte, by_word);

    // u64 splits into two u32 lanes correctly.
    let v: u64 = 0x0102_0304_0506_0708;
    let lo = _mm_crc32_u32(0, (v & 0xffff_ffff) as u32);
    let full = _mm_crc32_u32(lo, (v >> 32) as u32);
    assert_eq!(_mm_crc32_u64(0, v), full as u64);
}

#[test]
fn popcnt_known_answers() {
    assert_eq!(_mm_popcnt_u32(0), 0);
    assert_eq!(_mm_popcnt_u32(0xFFFF_FFFF), 32);
    assert_eq!(_mm_popcnt_u32(0xAAAA_AAAA), 16);
    assert_eq!(_mm_popcnt_u64(0), 0);
    assert_eq!(_mm_popcnt_u64(u64::MAX), 64);
    assert_eq!(_mm_popcnt_u64(0x0F0F_0F0F_0F0F_0F0F), 32);
}
