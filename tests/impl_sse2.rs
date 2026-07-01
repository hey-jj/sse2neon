//! SSE2 integer and double conformance against scalar oracles.

mod common;

use common::*;
use sse2neon::*;

fn iwin(ints: &[i32], i: usize) -> __m128i {
    _mm_set_epi32(ints[i + 3], ints[i + 2], ints[i + 1], ints[i])
}

// --- 32-bit integer arithmetic ---

macro_rules! epi32_binop {
    ($name:ident, $intr:ident, $op:expr) => {
        #[test]
        fn $name() {
            let (_, ints) = test_streams();
            for i in (0..(10000 - 8)).step_by(7) {
                let a = iwin(&ints, i);
                let b = iwin(&ints, i + 4);
                let r = i32x4($intr(a, b));
                let av = i32x4(a);
                let bv = i32x4(b);
                for lane in 0..4 {
                    let want: i32 = $op(av[lane], bv[lane]);
                    assert_eq!(r[lane], want, "{} lane {}", stringify!($intr), lane);
                }
            }
        }
    };
}

epi32_binop!(add_epi32, _mm_add_epi32, |a: i32, b: i32| a.wrapping_add(b));
epi32_binop!(sub_epi32, _mm_sub_epi32, |a: i32, b: i32| a.wrapping_sub(b));
epi32_binop!(and_si128, _mm_and_si128, |a: i32, b: i32| a & b);
epi32_binop!(or_si128, _mm_or_si128, |a: i32, b: i32| a | b);
epi32_binop!(xor_si128, _mm_xor_si128, |a: i32, b: i32| a ^ b);
epi32_binop!(andnot_si128, _mm_andnot_si128, |a: i32, b: i32| !a & b);
epi32_binop!(min_epi32_via41, _mm_min_epi32, |a: i32, b: i32| a.min(b));
epi32_binop!(max_epi32_via41, _mm_max_epi32, |a: i32, b: i32| a.max(b));
epi32_binop!(mullo_epi32, _mm_mullo_epi32, |a: i32, b: i32| a
    .wrapping_mul(b));

macro_rules! epi32_cmp {
    ($name:ident, $intr:ident, $op:expr) => {
        #[test]
        fn $name() {
            let (_, ints) = test_streams();
            for i in (0..(10000 - 8)).step_by(9) {
                let a = iwin(&ints, i);
                let b = iwin(&ints, i + 4);
                let r = i32x4($intr(a, b));
                let av = i32x4(a);
                let bv = i32x4(b);
                for lane in 0..4 {
                    let cond: bool = $op(av[lane], bv[lane]);
                    assert_eq!(r[lane], if cond { -1 } else { 0 });
                }
            }
        }
    };
}

epi32_cmp!(cmpeq_epi32, _mm_cmpeq_epi32, |a: i32, b: i32| a == b);
epi32_cmp!(cmpgt_epi32, _mm_cmpgt_epi32, |a: i32, b: i32| a > b);
epi32_cmp!(cmplt_epi32, _mm_cmplt_epi32, |a: i32, b: i32| a < b);

// --- 8 and 16 bit saturating arithmetic ---

#[test]
fn adds_epi8_saturates() {
    let a = _mm_set1_epi8(100);
    let b = _mm_set1_epi8(100);
    // 100 + 100 = 200 -> saturates to 127
    assert_eq!(i8x16(_mm_adds_epi8(a, b))[0], 127);
    let c = _mm_set1_epi8(-100);
    assert_eq!(i8x16(_mm_adds_epi8(c, c))[0], -128);
}

#[test]
fn adds_epu8_saturates() {
    let a = _mm_set1_epi8(200u8 as i8);
    let b = _mm_set1_epi8(100u8 as i8);
    // 200 + 100 = 300 -> saturates to 255
    assert_eq!(u8x16(_mm_adds_epu8(a, b))[0], 255);
}

#[test]
fn subs_epu8_floors_at_zero() {
    let a = _mm_set1_epi8(10);
    let b = _mm_set1_epi8(50);
    assert_eq!(u8x16(_mm_subs_epu8(a, b))[0], 0);
}

#[test]
fn adds_epi16_full_range() {
    for lane_a in [i16::MIN, -1, 0, 1, i16::MAX, 30000] {
        for lane_b in [i16::MIN, -1, 0, 1, i16::MAX, 30000] {
            let a = _mm_set1_epi16(lane_a);
            let b = _mm_set1_epi16(lane_b);
            let want =
                (lane_a as i32 + lane_b as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            assert_eq!(i16x8(_mm_adds_epi16(a, b))[0], want);
        }
    }
}

// --- pack / unpack ---

#[test]
fn packs_and_packus_boundaries() {
    let hi = _mm_set1_epi16(300);
    let lo = _mm_set1_epi16(-300);
    // packs_epi16: signed saturate to i8. 300->127, -300->-128
    let r = i8x16(_mm_packs_epi16(hi, lo));
    assert_eq!(r[0], 127);
    assert_eq!(r[8], -128);
    // packus_epi16: unsigned saturate to u8. 300->255, -300->0
    let ru = u8x16(_mm_packus_epi16(hi, lo));
    assert_eq!(ru[0], 255);
    assert_eq!(ru[8], 0);
    // packs_epi32
    let big = _mm_set1_epi32(100000);
    let rs = i16x8(_mm_packs_epi32(big, big));
    assert_eq!(rs[0], i16::MAX);
}

#[test]
fn unpack_epi8() {
    let a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    let b = _mm_set_epi8(
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
    );
    let lo = i8x16(_mm_unpacklo_epi8(a, b));
    assert_eq!(lo[0], 0);
    assert_eq!(lo[1], 16);
    assert_eq!(lo[2], 1);
    assert_eq!(lo[3], 17);
    let hi = i8x16(_mm_unpackhi_epi8(a, b));
    assert_eq!(hi[0], 8);
    assert_eq!(hi[1], 24);
}

// --- shifts ---

#[test]
fn shifts_by_immediate() {
    let a = _mm_set1_epi32(1);
    assert_eq!(i32x4(_mm_slli_epi32::<4>(a))[0], 16);
    assert_eq!(i32x4(_mm_slli_epi32::<32>(a))[0], 0); // count >= width -> 0
    let b = _mm_set1_epi32(-16);
    assert_eq!(i32x4(_mm_srai_epi32::<2>(b))[0], -4); // arithmetic
    assert_eq!(i32x4(_mm_srai_epi32::<40>(b))[0], -1); // saturates to sign
    let c = _mm_set1_epi32(0x40);
    assert_eq!(u32x4(_mm_srli_epi32::<4>(c))[0], 4);
}

#[test]
fn byte_shifts() {
    let a = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    // shift left by 1 byte: lane0 becomes 0, lanes move up
    let l = u8x16(_mm_slli_si128::<1>(a));
    assert_eq!(l[0], 0);
    assert_eq!(l[1], 0);
    let r = u8x16(_mm_srli_si128::<1>(a));
    assert_eq!(r[0], 1);
    assert_eq!(r[15], 0);
    assert_eq!(u8x16(_mm_slli_si128::<16>(a)), [0u8; 16]);
    // imm 0 is identity
    assert_eq!(u8x16(_mm_slli_si128::<0>(a)), u8x16(a));
}

// --- shuffle ---

#[test]
fn shuffle_epi32_selects() {
    let a = _mm_set_epi32(4, 3, 2, 1);
    // reverse: (0,1,2,3) picks lane0=a[0], lane1=a[1], lane2=a[2], lane3=a[3]
    let r = i32x4(_mm_shuffle_epi32::<{ _MM_SHUFFLE(0, 1, 2, 3) }>(a));
    assert_eq!(r, [4, 3, 2, 1]);
    // broadcast lane 0
    let b = i32x4(_mm_shuffle_epi32::<{ _MM_SHUFFLE(0, 0, 0, 0) }>(a));
    assert_eq!(b, [1, 1, 1, 1]);
}

// --- movemask ---

#[test]
fn movemask_epi8_gathers_signs() {
    let a = _mm_set1_epi8(-1);
    assert_eq!(_mm_movemask_epi8(a), 0xFFFF);
    let z = _mm_setzero_si128();
    assert_eq!(_mm_movemask_epi8(z), 0);
    // Only lane 0 negative
    let one = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1);
    assert_eq!(_mm_movemask_epi8(one), 1);
}

// --- madd / mulhi / sad ---

#[test]
fn madd_epi16() {
    let a = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    let b = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
    // lane0 pair: 1*1+2*2=5; then 3*3+4*4=25; 5*5+6*6=61; 7*7+8*8=113
    assert_eq!(i32x4(_mm_madd_epi16(a, b)), [5, 25, 61, 113]);
}

#[test]
fn sad_epu8() {
    let a = _mm_set1_epi8(10);
    let b = _mm_set1_epi8(3);
    // |10-3| = 7, times 8 per 64-bit half = 56
    let r = u64x2(_mm_sad_epu8(a, b));
    assert_eq!(r[0], 56);
    assert_eq!(r[1], 56);
}

// --- doubles ---

#[test]
fn double_arithmetic() {
    let a = _mm_set_pd(3.0, 1.0);
    let b = _mm_set_pd(4.0, 2.0);
    assert_eq!(f64x2(_mm_add_pd(a, b)), [3.0, 7.0]);
    assert_eq!(f64x2(_mm_sub_pd(a, b)), [-1.0, -1.0]);
    assert_eq!(f64x2(_mm_mul_pd(a, b)), [2.0, 12.0]);
    assert_eq!(f64x2(_mm_div_pd(a, b)), [0.5, 0.75]);
}

#[test]
fn double_compares_and_movemask() {
    let a = _mm_set_pd(-1.0, 2.0);
    let b = _mm_set_pd(1.0, 2.0);
    let eq = u64x2(_mm_castpd_si128(_mm_cmpeq_pd(a, b)));
    assert_eq!(eq, [u64::MAX, 0]);
    // signs: lane0=2.0(+), lane1=-1.0(sign) -> 0b10
    assert_eq!(_mm_movemask_pd(a), 0b10);
}

#[test]
fn cvt_roundtrip_int_double() {
    let original = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
    let i = _mm_set_epi32(0, 0, 7, 3);
    let d = f64x2(_mm_cvtepi32_pd(i));
    assert_eq!(d, [3.0, 7.0]);
    let back = i32x4(_mm_cvtpd_epi32(_mm_set_pd(7.5, 2.5)));
    // round half to even: 2.5->2, 7.5->8
    assert_eq!(back[0], 2);
    assert_eq!(back[1], 8);
    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn cvttpd_truncates_toward_zero() {
    // Truncation ignores the rounding mode and always drops the fraction.
    let original = _MM_GET_ROUNDING_MODE();
    _MM_SET_ROUNDING_MODE(_MM_ROUND_UP);
    let r = i32x4(_mm_cvttpd_epi32(_mm_set_pd(-2.7, 2.7)));
    assert_eq!(r[0], 2); // 2.7 -> 2
    assert_eq!(r[1], -2); // -2.7 -> -2, toward zero
    _MM_SET_ROUNDING_MODE(original);
}

#[test]
fn cvt_saturation_indefinite() {
    let big = _mm_set1_ps(4e9);
    assert_eq!(i32x4(_mm_cvttps_epi32(big))[0], i32::MIN);
    let inf = _mm_set1_ps(f32::INFINITY);
    assert_eq!(i32x4(_mm_cvtps_epi32(inf))[0], i32::MIN);
    let nan = _mm_set1_ps(f32::NAN);
    assert_eq!(i32x4(_mm_cvttps_epi32(nan))[0], i32::MIN);
}

// --- Saturating add/sub across widths (both saturation directions) ---

#[test]
fn saturating_subs_and_adds_extra_widths() {
    // Signed 8-bit underflow clamps to i8::MIN.
    assert_eq!(
        i8x16(_mm_subs_epi8(_mm_set1_epi8(-100), _mm_set1_epi8(100)))[0],
        -128
    );
    // Signed 8-bit overflow clamps to i8::MAX.
    assert_eq!(
        i8x16(_mm_subs_epi8(_mm_set1_epi8(100), _mm_set1_epi8(-100)))[0],
        127
    );

    // Signed 16-bit overflow clamps to i16::MAX.
    assert_eq!(
        i16x8(_mm_subs_epi16(
            _mm_set1_epi16(30000),
            _mm_set1_epi16(-30000)
        ))[0],
        32767
    );
    // Signed 16-bit underflow clamps to i16::MIN.
    assert_eq!(
        i16x8(_mm_subs_epi16(
            _mm_set1_epi16(-30000),
            _mm_set1_epi16(30000)
        ))[0],
        -32768
    );

    // Unsigned 16-bit subtract floors at zero.
    assert_eq!(
        u16x8(_mm_subs_epu16(_mm_set1_epi16(10), _mm_set1_epi16(50)))[0],
        0
    );
    // Unsigned 16-bit add ceils at u16::MAX.
    let sixty_k = _mm_set1_epi16(60000u16 as i16);
    let ten_k = _mm_set1_epi16(10000);
    assert_eq!(u16x8(_mm_adds_epu16(sixty_k, ten_k))[0], 65535);
}

// --- Widening and high-half multiplies ---

#[test]
fn widening_multiplies() {
    // mul_epu32 reads epi32 lanes 0 and 2 of each input.
    let a = _mm_set_epi32(0, 7, 0, 0xFFFFFFFFu32 as i32);
    let b = _mm_set_epi32(0, 3, 0, 2);
    let r = u64x2(_mm_mul_epu32(a, b));
    assert_eq!(r, [0x1_FFFF_FFFE, 21]); // 0xFFFFFFFF*2, 7*3

    // mul_epi32 signed, same lane selection.
    let a = _mm_set_epi32(0, 7, 0, -1);
    let b = _mm_set_epi32(0, 3, 0, 3);
    assert_eq!(i64x2(_mm_mul_epi32(a, b)), [-3, 21]);
}

#[test]
fn high_half_multiplies_signed_vs_unsigned() {
    // 0x4000 * 0x4000 = 0x1000_0000, high 16 bits = 0x1000.
    let a = _mm_set1_epi16(0x4000);
    assert_eq!(u16x8(_mm_mulhi_epi16(a, a))[0], 0x1000);

    // 0xFFFF * 0xFFFF = 0xFFFE_0001. Unsigned high = 0xFFFE.
    let ones = _mm_set1_epi16(-1); // 0xFFFF
    assert_eq!(u16x8(_mm_mulhi_epu16(ones, ones))[0], 0xFFFE);
    // Signed: (-1)*(-1) = 1, high 16 bits = 0.
    assert_eq!(i16x8(_mm_mulhi_epi16(ones, ones))[0], 0);

    // mullo keeps the low 16 bits.
    let x = _mm_set1_epi16(0x1234);
    let y = _mm_set1_epi16(0x0010);
    assert_eq!(u16x8(_mm_mullo_epi16(x, y))[0], 0x2340);
}

// --- Half-lane shuffles leave the other half alone ---

#[test]
fn shuffle_hi_lo_epi16() {
    let s = _mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1); // lane0..7 = 1..8
                                                   // Reverse the low four, keep the high four.
    let lo = i16x8(_mm_shufflelo_epi16::<{ _MM_SHUFFLE(0, 1, 2, 3) }>(s));
    assert_eq!(lo, [4, 3, 2, 1, 5, 6, 7, 8]);
    // Reverse the high four, keep the low four.
    let hi = i16x8(_mm_shufflehi_epi16::<{ _MM_SHUFFLE(0, 1, 2, 3) }>(s));
    assert_eq!(hi, [1, 2, 3, 4, 8, 7, 6, 5]);
}

#[test]
fn valid_immediates_compile_and_run() {
    // Const-generic immediates carry a compile-time range assert. Confirm that
    // valid values in range still monomorphize and produce the right lane.
    let a = _mm_set_epi32(4, 3, 2, 1);
    assert_eq!(i32x4(_mm_shuffle_epi32::<255>(a))[0], 4); // top of 0..256
    assert_eq!(i32x4(_mm_slli_epi32::<0>(a))[0], 1); // bottom of 0..256
    assert_eq!(i32x4(_mm_slli_epi32::<255>(a))[0], 0); // count >= width -> 0
    assert_eq!(i32x4(_mm_srai_epi32::<255>(_mm_set1_epi32(-16)))[0], -1);
    assert_eq!(_mm_extract_epi32::<3>(a), 4); // top lane index
}
