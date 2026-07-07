//! SSE intrinsics over `f32` vectors, plus the MMX helpers they rely on.
//!
//! Covers arithmetic, bitwise logic, comparisons, ordered scalar compares,
//! set/load/store, min/max, shuffle, movemask, and the reciprocal and square
//! root estimates.

use crate::types::*;
use core::arch::aarch64::*;

// --- Arithmetic ---

/// Add four packed `f32` lanes. Matches `_mm_add_ps`.
#[inline]
pub fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vaddq_f32(a.f32(), b.f32()) })
}

/// Subtract packed `f32` lanes, `a - b`. Matches `_mm_sub_ps`.
#[inline]
pub fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vsubq_f32(a.f32(), b.f32()) })
}

/// Multiply packed `f32` lanes. Matches `_mm_mul_ps`.
#[inline]
pub fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vmulq_f32(a.f32(), b.f32()) })
}

/// Divide packed `f32` lanes, `a / b`. Matches `_mm_div_ps`.
#[inline]
pub fn _mm_div_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vdivq_f32(a.f32(), b.f32()) })
}

/// Add the low lanes and copy the upper three from `a`. Matches `_mm_add_ss`.
#[inline]
pub fn _mm_add_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_add_ps(a, b))
}

/// Subtract the low lanes and copy the upper three from `a`. Matches `_mm_sub_ss`.
#[inline]
pub fn _mm_sub_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_sub_ps(a, b))
}

/// Multiply the low lanes and copy the upper three from `a`. Matches `_mm_mul_ss`.
#[inline]
pub fn _mm_mul_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_mul_ps(a, b))
}

/// Divide the low lanes and copy the upper three from `a`. Matches `_mm_div_ss`.
#[inline]
pub fn _mm_div_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_div_ps(a, b))
}

// --- Bitwise ---

/// Bitwise AND of two `f32` vectors. Matches `_mm_and_ps`.
#[inline]
pub fn _mm_and_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vandq_u32(a.u32(), b.u32()) })
}

/// Bitwise OR of two `f32` vectors. Matches `_mm_or_ps`.
#[inline]
pub fn _mm_or_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vorrq_u32(a.u32(), b.u32()) })
}

/// Bitwise XOR of two `f32` vectors. Matches `_mm_xor_ps`.
#[inline]
pub fn _mm_xor_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { veorq_u32(a.u32(), b.u32()) })
}

/// Bitwise `(!a) & b`. The NOT applies to `a`. Matches `_mm_andnot_ps`.
#[inline]
pub fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 {
    // vbicq computes b & !a in NEON, which is the x86 andnot(a, b) semantics.
    __m128::from_u32(unsafe { vbicq_u32(b.u32(), a.u32()) })
}

// --- Comparisons ---

/// Lanes where `a == b` become all-ones, else all-zeros. Matches `_mm_cmpeq_ps`.
#[inline]
pub fn _mm_cmpeq_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vceqq_f32(a.f32(), b.f32()) })
}

/// Compare `a >= b` lane-wise. Matches `_mm_cmpge_ps`.
#[inline]
pub fn _mm_cmpge_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vcgeq_f32(a.f32(), b.f32()) })
}

/// Compare `a > b` lane-wise. Matches `_mm_cmpgt_ps`.
#[inline]
pub fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vcgtq_f32(a.f32(), b.f32()) })
}

/// Compare `a <= b` lane-wise. Matches `_mm_cmple_ps`.
#[inline]
pub fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vcleq_f32(a.f32(), b.f32()) })
}

/// Compare `a < b` lane-wise. Matches `_mm_cmplt_ps`.
#[inline]
pub fn _mm_cmplt_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vcltq_f32(a.f32(), b.f32()) })
}

/// Compare `a != b` lane-wise. Matches `_mm_cmpneq_ps`.
#[inline]
pub fn _mm_cmpneq_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vmvnq_u32(vceqq_f32(a.f32(), b.f32())) })
}

/// Compare `!(a >= b)` lane-wise. Matches `_mm_cmpnge_ps`.
#[inline]
pub fn _mm_cmpnge_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vmvnq_u32(vcgeq_f32(a.f32(), b.f32())) })
}

/// Compare `!(a > b)` lane-wise. Matches `_mm_cmpngt_ps`.
#[inline]
pub fn _mm_cmpngt_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vmvnq_u32(vcgtq_f32(a.f32(), b.f32())) })
}

/// Compare `!(a <= b)` lane-wise. Matches `_mm_cmpnle_ps`.
#[inline]
pub fn _mm_cmpnle_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vmvnq_u32(vcleq_f32(a.f32(), b.f32())) })
}

/// Compare `!(a < b)` lane-wise. Matches `_mm_cmpnlt_ps`.
#[inline]
pub fn _mm_cmpnlt_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_u32(unsafe { vmvnq_u32(vcltq_f32(a.f32(), b.f32())) })
}

/// Ordered compare: all-ones where neither lane is NaN. Matches `_mm_cmpord_ps`.
#[inline]
pub fn _mm_cmpord_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        // A lane is ordered when a == a and b == b (no NaN).
        let a_ord = vceqq_f32(a.f32(), a.f32());
        let b_ord = vceqq_f32(b.f32(), b.f32());
        __m128::from_u32(vandq_u32(a_ord, b_ord))
    }
}

/// Unordered compare: all-ones where either lane is NaN. Matches `_mm_cmpunord_ps`.
#[inline]
pub fn _mm_cmpunord_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a_ord = vceqq_f32(a.f32(), a.f32());
        let b_ord = vceqq_f32(b.f32(), b.f32());
        __m128::from_u32(vmvnq_u32(vandq_u32(a_ord, b_ord)))
    }
}

macro_rules! ss_cmp {
    ($name:ident, $full:ident, $doc:literal) => {
        #[doc = $doc]
        #[inline]
        pub fn $name(a: __m128, b: __m128) -> __m128 {
            _mm_move_ss(a, $full(a, b))
        }
    };
}

ss_cmp!(
    _mm_cmpeq_ss,
    _mm_cmpeq_ps,
    "Scalar `==` on lane 0. Matches `_mm_cmpeq_ss`."
);
ss_cmp!(
    _mm_cmpge_ss,
    _mm_cmpge_ps,
    "Scalar `>=` on lane 0. Matches `_mm_cmpge_ss`."
);
ss_cmp!(
    _mm_cmpgt_ss,
    _mm_cmpgt_ps,
    "Scalar `>` on lane 0. Matches `_mm_cmpgt_ss`."
);
ss_cmp!(
    _mm_cmple_ss,
    _mm_cmple_ps,
    "Scalar `<=` on lane 0. Matches `_mm_cmple_ss`."
);
ss_cmp!(
    _mm_cmplt_ss,
    _mm_cmplt_ps,
    "Scalar `<` on lane 0. Matches `_mm_cmplt_ss`."
);
ss_cmp!(
    _mm_cmpneq_ss,
    _mm_cmpneq_ps,
    "Scalar `!=` on lane 0. Matches `_mm_cmpneq_ss`."
);
ss_cmp!(
    _mm_cmpnge_ss,
    _mm_cmpnge_ps,
    "Scalar `!(>=)` on lane 0. Matches `_mm_cmpnge_ss`."
);
ss_cmp!(
    _mm_cmpngt_ss,
    _mm_cmpngt_ps,
    "Scalar `!(>)` on lane 0. Matches `_mm_cmpngt_ss`."
);
ss_cmp!(
    _mm_cmpnle_ss,
    _mm_cmpnle_ps,
    "Scalar `!(<=)` on lane 0. Matches `_mm_cmpnle_ss`."
);
ss_cmp!(
    _mm_cmpnlt_ss,
    _mm_cmpnlt_ps,
    "Scalar `!(<)` on lane 0. Matches `_mm_cmpnlt_ss`."
);
ss_cmp!(
    _mm_cmpord_ss,
    _mm_cmpord_ps,
    "Scalar ordered test on lane 0. Matches `_mm_cmpord_ss`."
);
ss_cmp!(
    _mm_cmpunord_ss,
    _mm_cmpunord_ps,
    "Scalar unordered test on lane 0. Matches `_mm_cmpunord_ss`."
);

// --- Ordered scalar compares returning int ---

/// Compare lane 0 for equality. Returns 1 or 0. NaN yields 0. Matches `_mm_comieq_ss`.
#[inline]
pub fn _mm_comieq_ss(a: __m128, b: __m128) -> i32 {
    let x = _mm_cvtss_f32(a);
    let y = _mm_cvtss_f32(b);
    i32::from(x == y)
}

/// Compare lane 0 with `>=`. Returns 1 or 0. Matches `_mm_comige_ss`.
#[inline]
pub fn _mm_comige_ss(a: __m128, b: __m128) -> i32 {
    i32::from(_mm_cvtss_f32(a) >= _mm_cvtss_f32(b))
}

/// Compare lane 0 with `>`. Returns 1 or 0. Matches `_mm_comigt_ss`.
#[inline]
pub fn _mm_comigt_ss(a: __m128, b: __m128) -> i32 {
    i32::from(_mm_cvtss_f32(a) > _mm_cvtss_f32(b))
}

/// Compare lane 0 with `<=`. Returns 1 or 0. Matches `_mm_comile_ss`.
#[inline]
pub fn _mm_comile_ss(a: __m128, b: __m128) -> i32 {
    i32::from(_mm_cvtss_f32(a) <= _mm_cvtss_f32(b))
}

/// Compare lane 0 with `<`. Returns 1 or 0. Matches `_mm_comilt_ss`.
#[inline]
pub fn _mm_comilt_ss(a: __m128, b: __m128) -> i32 {
    i32::from(_mm_cvtss_f32(a) < _mm_cvtss_f32(b))
}

/// Compare lane 0 for inequality. NaN yields 1. Matches `_mm_comineq_ss`.
#[inline]
pub fn _mm_comineq_ss(a: __m128, b: __m128) -> i32 {
    i32::from(_mm_cvtss_f32(a) != _mm_cvtss_f32(b))
}

/// Alias of [`_mm_comieq_ss`]. Matches `_mm_ucomieq_ss`.
#[inline]
pub fn _mm_ucomieq_ss(a: __m128, b: __m128) -> i32 {
    _mm_comieq_ss(a, b)
}
/// Alias of [`_mm_comige_ss`]. Matches `_mm_ucomige_ss`.
#[inline]
pub fn _mm_ucomige_ss(a: __m128, b: __m128) -> i32 {
    _mm_comige_ss(a, b)
}
/// Alias of [`_mm_comigt_ss`]. Matches `_mm_ucomigt_ss`.
#[inline]
pub fn _mm_ucomigt_ss(a: __m128, b: __m128) -> i32 {
    _mm_comigt_ss(a, b)
}
/// Alias of [`_mm_comile_ss`]. Matches `_mm_ucomile_ss`.
#[inline]
pub fn _mm_ucomile_ss(a: __m128, b: __m128) -> i32 {
    _mm_comile_ss(a, b)
}
/// Alias of [`_mm_comilt_ss`]. Matches `_mm_ucomilt_ss`.
#[inline]
pub fn _mm_ucomilt_ss(a: __m128, b: __m128) -> i32 {
    _mm_comilt_ss(a, b)
}
/// Alias of [`_mm_comineq_ss`]. Matches `_mm_ucomineq_ss`.
#[inline]
pub fn _mm_ucomineq_ss(a: __m128, b: __m128) -> i32 {
    _mm_comineq_ss(a, b)
}

// --- Min / max ---

/// Packed maximum.
/// Uses NEON max: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vmaxq_f32(a.f32(), b.f32()) })
}

/// Packed minimum.
/// Uses NEON min: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_min_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vminq_f32(a.f32(), b.f32()) })
}

/// Scalar maximum on lane 0. Upper lanes come from `a`.
/// Uses NEON max on lane 0: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_max_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_max_ps(a, b))
}

/// Scalar minimum on lane 0. Upper lanes come from `a`.
/// Uses NEON min on lane 0: a NaN operand propagates, and `-0.0` is below `+0.0`.
#[inline]
pub fn _mm_min_ss(a: __m128, b: __m128) -> __m128 {
    _mm_move_ss(a, _mm_min_ps(a, b))
}

// --- Move / shuffle / unpack ---

/// Copy lane 0 from `b`, lanes 1-3 from `a`. Matches `_mm_move_ss`.
#[inline]
pub fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let lane0 = vgetq_lane_f32(b.f32(), 0);
        __m128::from_f32(vsetq_lane_f32(lane0, a.f32(), 0))
    }
}

/// Move the two high lanes of `b` into the two low lanes, high from `a`.
/// Matches `_mm_movehl_ps`.
#[inline]
pub fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a_hi = vget_high_f32(a.f32());
        let b_hi = vget_high_f32(b.f32());
        __m128::from_f32(vcombine_f32(b_hi, a_hi))
    }
}

/// Move the two low lanes of `b` into the two high lanes, low from `a`.
/// Matches `_mm_movelh_ps`.
#[inline]
pub fn _mm_movelh_ps(a: __m128, b: __m128) -> __m128 {
    unsafe {
        let a_lo = vget_low_f32(a.f32());
        let b_lo = vget_low_f32(b.f32());
        __m128::from_f32(vcombine_f32(a_lo, b_lo))
    }
}

/// Interleave the high two lanes of `a` and `b`. Matches `_mm_unpackhi_ps`.
#[inline]
pub fn _mm_unpackhi_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vzip2q_f32(a.f32(), b.f32()) })
}

/// Interleave the low two lanes of `a` and `b`. Matches `_mm_unpacklo_ps`.
#[inline]
pub fn _mm_unpacklo_ps(a: __m128, b: __m128) -> __m128 {
    __m128::from_f32(unsafe { vzip1q_f32(a.f32(), b.f32()) })
}

/// Shuffle four `f32` lanes. `IMM` picks lanes 0-1 from `a`, 2-3 from `b`.
///
/// Matches `_mm_shuffle_ps`. `IMM` is a compile-time constant in `0..256`.
#[inline]
pub fn _mm_shuffle_ps<const IMM: i32>(a: __m128, b: __m128) -> __m128 {
    const { assert!(IMM >= 0 && IMM < 256, "IMM must be in 0..256") };
    let mut out = [0.0f32; 4];
    let av = to_f32_array(a);
    let bv = to_f32_array(b);
    out[0] = av[(IMM & 0x3) as usize];
    out[1] = av[((IMM >> 2) & 0x3) as usize];
    out[2] = bv[((IMM >> 4) & 0x3) as usize];
    out[3] = bv[((IMM >> 6) & 0x3) as usize];
    __m128::from_f32(unsafe { vld1q_f32(out.as_ptr()) })
}

#[inline]
fn to_f32_array(a: __m128) -> [f32; 4] {
    let mut out = [0.0f32; 4];
    unsafe { vst1q_f32(out.as_mut_ptr(), a.f32()) };
    out
}

// --- Movemask ---

/// Gather the sign bit of each of the four lanes into the low 4 bits.
/// Matches `_mm_movemask_ps`.
#[inline]
pub fn _mm_movemask_ps(a: __m128) -> i32 {
    unsafe {
        // Shift each 32-bit lane right by 31 to isolate the sign bit, then
        // weight lanes by 1, 2, 4, 8 and horizontally add.
        let signs = vshrq_n_u32::<31>(a.u32());
        let weights: uint32x4_t = {
            let w = [1u32, 2, 4, 8];
            vld1q_u32(w.as_ptr())
        };
        vaddvq_u32(vmulq_u32(signs, weights)) as i32
    }
}

// --- rcp / rsqrt / sqrt ---

/// Approximate reciprocal of each lane. About 12 bits. Matches `_mm_rcp_ps`.
#[inline]
pub fn _mm_rcp_ps(a: __m128) -> __m128 {
    unsafe {
        let x = a.f32();
        let mut e = vrecpeq_f32(x);
        e = vmulq_f32(vrecpsq_f32(x, e), e);
        __m128::from_f32(e)
    }
}

/// Approximate reciprocal of lane 0, upper lanes from `a`. Matches `_mm_rcp_ss`.
#[inline]
pub fn _mm_rcp_ss(a: __m128) -> __m128 {
    _mm_move_ss(a, _mm_rcp_ps(a))
}

/// Approximate reciprocal square root of each lane. Matches `_mm_rsqrt_ps`.
///
/// Restores `+Inf`/`-Inf` for `+0.0`/`-0.0` inputs, matching x86.
#[inline]
pub fn _mm_rsqrt_ps(a: __m128) -> __m128 {
    unsafe {
        let x = a.f32();
        let mut e = vrsqrteq_f32(x);
        e = vmulq_f32(vrsqrtsq_f32(vmulq_f32(x, e), e), e);

        // NEON gives NaN for rsqrt(0). x86 gives +/-Inf. Restore the sign of the
        // input into an infinity where the input was zero.
        let is_zero = vceqq_f32(x, vdupq_n_f32(0.0));
        let sign = vandq_u32(a.u32(), vdupq_n_u32(0x8000_0000));
        let inf = vorrq_u32(sign, vdupq_n_u32(0x7f80_0000));
        let fixed = vbslq_u32(is_zero, inf, vreinterpretq_u32_f32(e));
        __m128::from_u32(fixed)
    }
}

/// Approximate reciprocal square root of lane 0, upper from `a`.
/// Matches `_mm_rsqrt_ss`.
#[inline]
pub fn _mm_rsqrt_ss(a: __m128) -> __m128 {
    _mm_move_ss(a, _mm_rsqrt_ps(a))
}

/// Square root of each lane. Matches `_mm_sqrt_ps`.
#[inline]
pub fn _mm_sqrt_ps(a: __m128) -> __m128 {
    __m128::from_f32(unsafe { vsqrtq_f32(a.f32()) })
}

/// Square root of lane 0, upper lanes from `a`. Matches `_mm_sqrt_ss`.
#[inline]
pub fn _mm_sqrt_ss(a: __m128) -> __m128 {
    _mm_move_ss(a, _mm_sqrt_ps(a))
}

// --- Set / load / store ---

/// Broadcast one `f32` to all four lanes. Matches `_mm_set1_ps`.
#[inline]
pub fn _mm_set1_ps(w: f32) -> __m128 {
    __m128::from_f32(unsafe { vdupq_n_f32(w) })
}

/// Alias of [`_mm_set1_ps`]. Matches `_mm_set_ps1`.
#[inline]
pub fn _mm_set_ps1(w: f32) -> __m128 {
    _mm_set1_ps(w)
}

/// Set four lanes, most significant first. Lane 0 is `x`. Matches `_mm_set_ps`.
#[inline]
pub fn _mm_set_ps(w: f32, z: f32, y: f32, x: f32) -> __m128 {
    let data = [x, y, z, w];
    __m128::from_f32(unsafe { vld1q_f32(data.as_ptr()) })
}

/// Set four lanes in order. Lane 0 is `w`. Matches `_mm_setr_ps`.
#[inline]
pub fn _mm_setr_ps(w: f32, z: f32, y: f32, x: f32) -> __m128 {
    let data = [w, z, y, x];
    __m128::from_f32(unsafe { vld1q_f32(data.as_ptr()) })
}

/// Set lane 0 to `w`, upper lanes to zero. Matches `_mm_set_ss`.
#[inline]
pub fn _mm_set_ss(w: f32) -> __m128 {
    let data = [w, 0.0, 0.0, 0.0];
    __m128::from_f32(unsafe { vld1q_f32(data.as_ptr()) })
}

/// Return a zeroed vector. Matches `_mm_setzero_ps`.
#[inline]
pub fn _mm_setzero_ps() -> __m128 {
    __m128::from_f32(unsafe { vdupq_n_f32(0.0) })
}

/// Extract lane 0 as `f32`. Matches `_mm_cvtss_f32`.
#[inline]
pub fn _mm_cvtss_f32(a: __m128) -> f32 {
    unsafe { vgetq_lane_f32(a.f32(), 0) }
}

/// Load four `f32` from a 16-byte aligned pointer. Matches `_mm_load_ps`.
///
/// # Safety
/// `p` must point to 16 readable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_load_ps(p: *const f32) -> __m128 {
    __m128::from_f32(vld1q_f32(p))
}

/// Load four `f32` from an unaligned pointer. Matches `_mm_loadu_ps`.
///
/// # Safety
/// `p` must point to 16 readable bytes.
#[inline]
pub unsafe fn _mm_loadu_ps(p: *const f32) -> __m128 {
    __m128::from_f32(vld1q_f32(p))
}

/// Broadcast one `f32` from memory to all lanes. Matches `_mm_load1_ps`.
///
/// # Safety
/// `p` must point to a readable `f32`.
#[inline]
pub unsafe fn _mm_load1_ps(p: *const f32) -> __m128 {
    __m128::from_f32(vld1q_dup_f32(p))
}

/// Alias of [`_mm_load1_ps`]. Matches `_mm_load_ps1`.
///
/// # Safety
/// `p` must point to a readable `f32`.
#[inline]
pub unsafe fn _mm_load_ps1(p: *const f32) -> __m128 {
    _mm_load1_ps(p)
}

/// Load one `f32` into lane 0, zero the rest. Matches `_mm_load_ss`.
///
/// # Safety
/// `p` must point to a readable `f32`.
#[inline]
pub unsafe fn _mm_load_ss(p: *const f32) -> __m128 {
    __m128::from_f32(vsetq_lane_f32(*p, vdupq_n_f32(0.0), 0))
}

/// Store four `f32` to a 16-byte aligned pointer. Matches `_mm_store_ps`.
///
/// # Safety
/// `p` must point to 16 writable, 16-byte-aligned bytes.
#[inline]
pub unsafe fn _mm_store_ps(p: *mut f32, a: __m128) {
    vst1q_f32(p, a.f32());
}

/// Store four `f32` to an unaligned pointer. Matches `_mm_storeu_ps`.
///
/// # Safety
/// `p` must point to 16 writable bytes.
#[inline]
pub unsafe fn _mm_storeu_ps(p: *mut f32, a: __m128) {
    vst1q_f32(p, a.f32());
}

/// Store lane 0 to memory. Matches `_mm_store_ss`.
///
/// # Safety
/// `p` must point to a writable `f32`.
#[inline]
pub unsafe fn _mm_store_ss(p: *mut f32, a: __m128) {
    *p = _mm_cvtss_f32(a);
}
