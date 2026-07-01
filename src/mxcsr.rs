//! MXCSR mode control mapped onto the ARM FPCR register.
//!
//! The rounding mode and the flush-to-zero and denormals-are-zero flags map to
//! FPCR fields. Exception flags and masks are not emulated. Their getters return
//! fixed values and their setters do nothing, matching the x86 API surface
//! without tracking real exception state.

use crate::constants::*;
use core::arch::asm;

/// FPCR RMode field, bits 22-23.
const FPCR_RMODE_SHIFT: u64 = 22;
const FPCR_RMODE_MASK: u64 = 0b11 << FPCR_RMODE_SHIFT;

/// FPCR FZ bit (flush-to-zero), bit 24. On ARM this covers both FTZ and DAZ.
const FPCR_FZ_BIT: u64 = 1 << 24;

fn read_fpcr() -> u64 {
    let v: u64;
    unsafe { asm!("mrs {}, FPCR", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

fn write_fpcr(v: u64) {
    unsafe { asm!("msr FPCR, {}", in(reg) v, options(nomem, nostack, preserves_flags)) };
}

/// Read the current rounding mode as one of the `_MM_ROUND_*` values.
///
/// Matches `_MM_GET_ROUNDING_MODE`.
#[inline]
pub fn _MM_GET_ROUNDING_MODE() -> u32 {
    let rmode = (read_fpcr() & FPCR_RMODE_MASK) >> FPCR_RMODE_SHIFT;
    match rmode {
        0b00 => _MM_ROUND_NEAREST,
        0b01 => _MM_ROUND_UP,
        0b10 => _MM_ROUND_DOWN,
        _ => _MM_ROUND_TOWARD_ZERO,
    }
}

/// Set the rounding mode from a `_MM_ROUND_*` value.
///
/// Any value outside the four defined modes selects toward-zero, matching the
/// x86 fallback. Matches `_MM_SET_ROUNDING_MODE`.
#[inline]
pub fn _MM_SET_ROUNDING_MODE(mode: u32) {
    let rmode: u64 = match mode {
        _MM_ROUND_NEAREST => 0b00,
        _MM_ROUND_UP => 0b01,
        _MM_ROUND_DOWN => 0b10,
        _ => 0b11,
    };
    let fpcr = (read_fpcr() & !FPCR_RMODE_MASK) | (rmode << FPCR_RMODE_SHIFT);
    write_fpcr(fpcr);
}

/// Read the flush-to-zero mode as `_MM_FLUSH_ZERO_ON` or `_MM_FLUSH_ZERO_OFF`.
///
/// Matches `_MM_GET_FLUSH_ZERO_MODE`.
#[inline]
pub fn _MM_GET_FLUSH_ZERO_MODE() -> u32 {
    if read_fpcr() & FPCR_FZ_BIT != 0 {
        _MM_FLUSH_ZERO_ON
    } else {
        _MM_FLUSH_ZERO_OFF
    }
}

/// Set the flush-to-zero mode. Matches `_MM_SET_FLUSH_ZERO_MODE`.
#[inline]
pub fn _MM_SET_FLUSH_ZERO_MODE(mode: u32) {
    let fpcr = read_fpcr();
    let next = if mode == _MM_FLUSH_ZERO_ON {
        fpcr | FPCR_FZ_BIT
    } else {
        fpcr & !FPCR_FZ_BIT
    };
    write_fpcr(next);
}

/// Read the denormals-are-zero mode. ARM ties DAZ to the same bit as FTZ.
///
/// Matches `_MM_GET_DENORMALS_ZERO_MODE`.
#[inline]
pub fn _MM_GET_DENORMALS_ZERO_MODE() -> u32 {
    if read_fpcr() & FPCR_FZ_BIT != 0 {
        _MM_DENORMALS_ZERO_ON
    } else {
        _MM_DENORMALS_ZERO_OFF
    }
}

/// Set the denormals-are-zero mode. Matches `_MM_SET_DENORMALS_ZERO_MODE`.
#[inline]
pub fn _MM_SET_DENORMALS_ZERO_MODE(mode: u32) {
    let fpcr = read_fpcr();
    let next = if mode == _MM_DENORMALS_ZERO_ON {
        fpcr | FPCR_FZ_BIT
    } else {
        fpcr & !FPCR_FZ_BIT
    };
    write_fpcr(next);
}

/// Read the MXCSR register value. Reports rounding, FTZ, and DAZ modes.
///
/// FPCR bit 24 backs both FTZ and DAZ, so when it is set both bits are reported.
/// Exception flags are always zero. Matches `_mm_getcsr`.
#[inline]
pub fn _mm_getcsr() -> u32 {
    _MM_GET_ROUNDING_MODE() | _MM_GET_FLUSH_ZERO_MODE() | _MM_GET_DENORMALS_ZERO_MODE()
}

/// Write the MXCSR register. Applies the rounding, FTZ, and DAZ fields.
///
/// FPCR bit 24 backs both FTZ and DAZ, so either bit in the input word sets it.
/// Exception flags and masks are ignored. Matches `_mm_setcsr`.
#[inline]
pub fn _mm_setcsr(a: u32) {
    _MM_SET_ROUNDING_MODE(a & _MM_ROUND_MASK);
    let flush = (a & _MM_FLUSH_ZERO_MASK != 0) || (a & _MM_DENORMALS_ZERO_MASK != 0);
    _MM_SET_FLUSH_ZERO_MODE(if flush {
        _MM_FLUSH_ZERO_ON
    } else {
        _MM_FLUSH_ZERO_OFF
    });
}
