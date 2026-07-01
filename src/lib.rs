//! x86 SSE-family SIMD intrinsics reimplemented on ARM NEON.
//!
//! This crate mirrors the Intel `_mm_*` intrinsic surface (SSE, SSE2, SSE3,
//! SSSE3, SSE4.1, SSE4.2, AES, CRC32C) on top of `core::arch::aarch64` NEON.
//! Each function reproduces the observable x86 lane semantics under
//! little-endian memory ordering. Code written for x86 SIMD can call these
//! functions and get matching results on AArch64.
//!
//! # Types
//!
//! Four vector types match the Intel ABI. Each wraps a NEON register:
//!
//! - [`__m128`] holds four `f32` lanes over `float32x4_t`.
//! - [`__m128d`] holds two `f64` lanes over `float64x2_t`.
//! - [`__m128i`] holds a 128-bit integer vector over `int64x2_t`.
//! - [`__m64`] holds a 64-bit integer vector over `int64x1_t`.
//!
//! # Lane ordering
//!
//! `_mm_set_ps(w, z, y, x)` places `x` in lane 0 and `w` in lane 3. Arguments
//! read most-significant lane first. Memory stores least-significant lane
//! first. `_mm_setr_*` reverses the argument order. Every `set`/`setr`
//! function preserves this exact mapping.
//!
//! # Platform
//!
//! AArch64 only. The intrinsics use NEON registers that exist on ARMv8. The
//! target must be little-endian, which every AArch64 target is.
//!
//! # Safety
//!
//! The public functions are safe. They wrap `unsafe` NEON intrinsic calls that
//! carry no memory-safety preconditions when the target supports NEON, which is
//! guaranteed on AArch64. Load and store functions that take raw pointers are
//! marked `unsafe` and document their alignment and length requirements.

#![cfg(target_arch = "aarch64")]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

mod aes;
mod constants;
mod sse;
mod sse2;
mod sse3;
mod sse41;
mod sse42;
mod ssse3;
mod types;

pub use aes::*;
pub use constants::*;
pub use sse::*;
pub use sse2::*;
pub use sse3::*;
pub use sse41::*;
pub use sse42::*;
pub use ssse3::*;
pub use types::*;
