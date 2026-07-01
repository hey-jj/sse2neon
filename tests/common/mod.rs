//! Shared test helpers: deterministic PRNG, lane readers, and scalar oracles.
//!
//! The PRNG is a fixed SplitMix64 stream, so inputs are reproducible across
//! runs. Lane readers pull typed views out of the vector types. Scalar oracles
//! restate the x86 semantics independently of the NEON implementation, so a
//! test compares the intrinsic against a separate source of truth.

#![allow(dead_code)]

use sse2neon::*;

/// SplitMix64 state. Advance with [`SplitMix64::next_u64`].
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Seed the generator.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Return the next 64-bit output and advance the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform `f64` in `[0, 1)`.
    pub fn ranf(&mut self) -> f64 {
        self.next_u64() as f64 / 18446744073709551616.0
    }

    /// Uniform `f64` in `[lo, hi)`.
    pub fn ranf_range(&mut self, lo: f64, hi: f64) -> f64 {
        self.ranf() * (hi - lo) + lo
    }
}

/// Generate the float and int input streams that drive the conformance tests.
///
/// 10000 floats then 10000 ints, each drawn uniformly from `[-100000, 100000)`
/// with seed 123456.
pub fn test_streams() -> (Vec<f32>, Vec<i32>) {
    let mut rng = SplitMix64::new(123456);
    let mut floats = Vec::with_capacity(10000);
    let mut ints = Vec::with_capacity(10000);
    for _ in 0..10000 {
        floats.push(rng.ranf_range(-100000.0, 100000.0) as f32);
    }
    // The int stream draws from the same generator after the floats.
    for _ in 0..10000 {
        ints.push(rng.ranf_range(-100000.0, 100000.0) as i32);
    }
    (floats, ints)
}

// --- Lane readers ---

pub fn f32x4(a: __m128) -> [f32; 4] {
    let mut o = [0.0f32; 4];
    unsafe { _mm_storeu_ps(o.as_mut_ptr(), a) };
    o
}

pub fn f64x2(a: __m128d) -> [f64; 2] {
    let mut o = [0.0f64; 2];
    unsafe { _mm_storeu_pd(o.as_mut_ptr(), a) };
    o
}

pub fn i8x16(a: __m128i) -> [i8; 16] {
    let mut o = [0i8; 16];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn u8x16(a: __m128i) -> [u8; 16] {
    let mut o = [0u8; 16];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn i16x8(a: __m128i) -> [i16; 8] {
    let mut o = [0i16; 8];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn u16x8(a: __m128i) -> [u16; 8] {
    let mut o = [0u16; 8];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn i32x4(a: __m128i) -> [i32; 4] {
    let mut o = [0i32; 4];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn u32x4(a: __m128i) -> [u32; 4] {
    let mut o = [0u32; 4];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn i64x2(a: __m128i) -> [i64; 2] {
    let mut o = [0i64; 2];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

pub fn u64x2(a: __m128i) -> [u64; 2] {
    let mut o = [0u64; 2];
    unsafe { _mm_storeu_si128(o.as_mut_ptr() as *mut __m128i, a) };
    o
}

// --- Scalar saturation oracles ---

pub fn sat_i8(v: i32) -> i8 {
    v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

pub fn sat_i16(v: i32) -> i16 {
    v.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub fn sat_u8(v: i32) -> u8 {
    v.clamp(0, u8::MAX as i32) as u8
}

pub fn sat_u16(v: i32) -> u16 {
    v.clamp(0, u16::MAX as i32) as u16
}

/// Round half to even, matching the SSE default conversion mode.
pub fn bankers_round_f32(v: f32) -> f32 {
    v.round_ties_even()
}
