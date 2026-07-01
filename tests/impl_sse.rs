//! SSE float conformance. Drives the deterministic stream through each
//! intrinsic and compares against a scalar oracle.

mod common;

use common::*;
use sse2neon::*;

/// Load a 4-float window at index `i` from the stream.
fn win(floats: &[f32], i: usize) -> __m128 {
    _mm_set_ps(floats[i + 3], floats[i + 2], floats[i + 1], floats[i])
}

macro_rules! ps_binop_test {
    ($name:ident, $intr:ident, $op:expr) => {
        #[test]
        fn $name() {
            let (f, _) = test_streams();
            for i in (0..(10000 - 8)).step_by(7) {
                let a = win(&f, i);
                let b = win(&f, i + 4);
                let r = f32x4($intr(a, b));
                let av = f32x4(a);
                let bv = f32x4(b);
                for lane in 0..4 {
                    let want: f32 = $op(av[lane], bv[lane]);
                    assert_eq!(
                        r[lane].to_bits(),
                        want.to_bits(),
                        "{} lane {} a={} b={}",
                        stringify!($intr),
                        lane,
                        av[lane],
                        bv[lane]
                    );
                }
            }
        }
    };
}

ps_binop_test!(add_ps, _mm_add_ps, |a: f32, b: f32| a + b);
ps_binop_test!(sub_ps, _mm_sub_ps, |a: f32, b: f32| a - b);
ps_binop_test!(mul_ps, _mm_mul_ps, |a: f32, b: f32| a * b);
ps_binop_test!(div_ps, _mm_div_ps, |a: f32, b: f32| a / b);
ps_binop_test!(min_ps, _mm_min_ps, |a: f32, b: f32| if a < b {
    a
} else {
    b
});
ps_binop_test!(max_ps, _mm_max_ps, |a: f32, b: f32| if a > b {
    a
} else {
    b
});

macro_rules! ps_cmp_test {
    ($name:ident, $intr:ident, $op:expr) => {
        #[test]
        fn $name() {
            let (f, _) = test_streams();
            for i in (0..(10000 - 8)).step_by(11) {
                let a = win(&f, i);
                let b = win(&f, i + 4);
                let r = i32x4(_mm_castps_si128($intr(a, b)));
                let av = f32x4(a);
                let bv = f32x4(b);
                for lane in 0..4 {
                    let cond: bool = $op(av[lane], bv[lane]);
                    let want = if cond { -1i32 } else { 0i32 };
                    assert_eq!(r[lane], want, "{} lane {}", stringify!($intr), lane);
                }
            }
        }
    };
}

ps_cmp_test!(cmpeq_ps, _mm_cmpeq_ps, |a: f32, b: f32| a == b);
ps_cmp_test!(cmplt_ps, _mm_cmplt_ps, |a: f32, b: f32| a < b);
ps_cmp_test!(cmple_ps, _mm_cmple_ps, |a: f32, b: f32| a <= b);
ps_cmp_test!(cmpgt_ps, _mm_cmpgt_ps, |a: f32, b: f32| a > b);
ps_cmp_test!(cmpge_ps, _mm_cmpge_ps, |a: f32, b: f32| a >= b);
ps_cmp_test!(cmpneq_ps, _mm_cmpneq_ps, |a: f32, b: f32| a != b);

#[test]
fn bitwise_ps() {
    let (f, _) = test_streams();
    for i in (0..(10000 - 8)).step_by(13) {
        let a = win(&f, i);
        let b = win(&f, i + 4);
        let av = f32x4(a);
        let bv = f32x4(b);
        let and = f32x4(_mm_and_ps(a, b));
        let or = f32x4(_mm_or_ps(a, b));
        let xor = f32x4(_mm_xor_ps(a, b));
        let andnot = f32x4(_mm_andnot_ps(a, b));
        for lane in 0..4 {
            let x = av[lane].to_bits();
            let y = bv[lane].to_bits();
            assert_eq!(and[lane].to_bits(), x & y);
            assert_eq!(or[lane].to_bits(), x | y);
            assert_eq!(xor[lane].to_bits(), x ^ y);
            assert_eq!(andnot[lane].to_bits(), !x & y);
        }
    }
}

#[test]
fn scalar_ss_upper_lanes() {
    // _ss ops operate on lane 0 and copy lanes 1-3 from a.
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    let r = f32x4(_mm_add_ss(a, b));
    assert_eq!(r, [11.0, 2.0, 3.0, 4.0]);
    let r = f32x4(_mm_mul_ss(a, b));
    assert_eq!(r, [10.0, 2.0, 3.0, 4.0]);
}

#[test]
fn scalar_ss_arithmetic_upper_lanes() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    // lane0 gets the op, lanes 1-3 come from a.
    assert_eq!(f32x4(_mm_sub_ss(a, b)), [-9.0, 2.0, 3.0, 4.0]);
    assert_eq!(f32x4(_mm_div_ss(a, b)), [0.1, 2.0, 3.0, 4.0]);

    let s = _mm_set_ps(4.0, 3.0, 2.0, 16.0);
    assert_eq!(f32x4(_mm_sqrt_ss(s)), [4.0, 2.0, 3.0, 4.0]);
}

#[test]
fn scalar_ss_min_max_with_nan() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ss(f32::NAN);
    // x86 min_ss/max_ss return the second operand on an unordered compare.
    assert!(f32x4(_mm_min_ss(a, b))[0].is_nan());
    assert_eq!(&f32x4(_mm_min_ss(a, b))[1..], &[2.0, 3.0, 4.0]);
    assert!(f32x4(_mm_max_ss(a, b))[0].is_nan());
    assert_eq!(&f32x4(_mm_max_ss(a, b))[1..], &[2.0, 3.0, 4.0]);
}

#[test]
fn scalar_ss_compare_upper_lanes() {
    // cmpord_ss: lane0 all-ones if both ordered, else zero. Upper from a.
    let n = _mm_set_ps(4.0, 3.0, 2.0, f32::NAN);
    let m = _mm_set_ps(40.0, 30.0, 20.0, 10.0);
    let r = f32x4(_mm_cmpord_ss(n, m));
    assert_eq!(r[0].to_bits(), 0); // NaN makes lane0 unordered
    assert_eq!(&r[1..], &[2.0, 3.0, 4.0]);

    // cmpeq_ss: lane0 equal -> all-ones bit pattern.
    let a = _mm_set_ps(4.0, 3.0, 2.0, 5.0);
    let b = _mm_set_ps(40.0, 30.0, 20.0, 5.0);
    let r = f32x4(_mm_cmpeq_ss(a, b));
    assert_eq!(r[0].to_bits(), 0xFFFF_FFFF);
    assert_eq!(&r[1..], &[2.0, 3.0, 4.0]);
}

#[test]
fn round_ss_upper_lanes() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(40.0, 30.0, 20.0, 2.7);
    // lane0 truncates 2.7 to 2, upper lanes from a.
    assert_eq!(
        f32x4(_mm_round_ss::<_MM_FROUND_TO_ZERO>(a, b)),
        [2.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(f32x4(_mm_floor_ss(a, b)), [2.0, 2.0, 3.0, 4.0]);
    assert_eq!(f32x4(_mm_ceil_ss(a, b)), [3.0, 2.0, 3.0, 4.0]);
}

#[test]
fn move_masks() {
    let a = _mm_set_ps(-1.0, 2.0, -3.0, 4.0);
    // lane0=4(+), lane1=-3(sign), lane2=2(+), lane3=-1(sign) -> bits 1 and 3 = 0b1010
    assert_eq!(_mm_movemask_ps(a), 0b1010);
}

#[test]
fn moves_and_unpacks() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
    assert_eq!(f32x4(_mm_movehl_ps(a, b)), [7.0, 8.0, 3.0, 4.0]);
    assert_eq!(f32x4(_mm_movelh_ps(a, b)), [1.0, 2.0, 5.0, 6.0]);
    assert_eq!(f32x4(_mm_unpacklo_ps(a, b)), [1.0, 5.0, 2.0, 6.0]);
    assert_eq!(f32x4(_mm_unpackhi_ps(a, b)), [3.0, 7.0, 4.0, 8.0]);
}

#[test]
fn shuffle_ps_selects_lanes() {
    let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
    let b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
    // imm = (3,2,1,0): lane0=a[0], lane1=a[1], lane2=b[2], lane3=b[3]
    let r = f32x4(_mm_shuffle_ps::<{ _MM_SHUFFLE(3, 2, 1, 0) }>(a, b));
    assert_eq!(r, [1.0, 2.0, 7.0, 8.0]);
}

#[test]
fn sqrt_matches_scalar() {
    let (f, _) = test_streams();
    for i in (0..(10000 - 4)).step_by(17) {
        let a = win(&f, i);
        let r = f32x4(_mm_sqrt_ps(a));
        let av = f32x4(a);
        for lane in 0..4 {
            assert_eq!(r[lane].to_bits(), av[lane].sqrt().to_bits());
        }
    }
}

#[test]
fn rcp_within_tolerance() {
    let a = _mm_set_ps(4.0, 2.0, 1.0, 0.5);
    let r = f32x4(_mm_rcp_ps(a));
    let want = [2.0, 1.0, 0.5, 0.25];
    for lane in 0..4 {
        assert!((r[lane] - want[lane]).abs() < 1.5e-3, "lane {}", lane);
    }
}

#[test]
fn rsqrt_zero_gives_signed_inf() {
    // Lanes read least significant first: [1.0, 4.0, +0.0, -0.0].
    let a = _mm_set_ps(-0.0, 0.0, 4.0, 1.0);
    let r = f32x4(_mm_rsqrt_ps(a));
    assert!((r[0] - 1.0).abs() < 1e-2); // rsqrt(1) ~ 1
    assert!((r[1] - 0.5).abs() < 1e-2); // rsqrt(4) ~ 0.5
    assert_eq!(r[2], f32::INFINITY); // rsqrt(+0) = +Inf
    assert_eq!(r[3], f32::NEG_INFINITY); // rsqrt(-0) = -Inf
}

#[test]
fn comi_returns_int_and_nan_rules() {
    let a = _mm_set_ss(1.0);
    let b = _mm_set_ss(2.0);
    assert_eq!(_mm_comilt_ss(a, b), 1);
    assert_eq!(_mm_comigt_ss(a, b), 0);
    assert_eq!(_mm_comieq_ss(a, a), 1);
    let nan = _mm_set_ss(f32::NAN);
    assert_eq!(_mm_comieq_ss(nan, a), 0);
    assert_eq!(_mm_comineq_ss(nan, a), 1);
}
