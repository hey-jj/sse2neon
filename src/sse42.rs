//! SSE4.2 intrinsics: 64-bit signed compare, CRC32C, and population count.

use crate::types::*;
use core::arch::aarch64::*;

/// Signed `a > b` for two `i64` lanes. Matches `_mm_cmpgt_epi64`.
#[inline]
pub fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i {
    __m128i::from_u64(unsafe { vcgtq_s64(a.s64(), b.s64()) })
}

/// Half-byte lookup table for CRC-32C (Castagnoli), reflected form.
///
/// Every sixteenth entry of the full 256-entry table. The two-step reduction in
/// [`_mm_crc32_u8`] recovers the full byte result.
const CRC32C_TABLE: [u32; 16] = [
    0x0000_0000,
    0x105e_c76f,
    0x20bd_8ede,
    0x30e3_49b1,
    0x417b_1dbc,
    0x5125_dad3,
    0x61c6_9362,
    0x7198_540d,
    0x82f6_3b78,
    0x92a8_fc17,
    0xa24b_b5a6,
    0xb215_72c9,
    0xc38d_26c4,
    0xd3d3_e1ab,
    0xe330_a81a,
    0xf36e_6f75,
];

/// Accumulate one byte into a CRC-32C checksum. Matches `_mm_crc32_u8`.
///
/// Uses the reflected Castagnoli polynomial `0x82F63B78`.
#[inline]
pub fn _mm_crc32_u8(mut crc: u32, v: u8) -> u32 {
    crc ^= v as u32;
    crc = (crc >> 4) ^ CRC32C_TABLE[(crc & 0x0f) as usize];
    crc = (crc >> 4) ^ CRC32C_TABLE[(crc & 0x0f) as usize];
    crc
}

/// Accumulate two bytes into a CRC-32C checksum. Matches `_mm_crc32_u16`.
#[inline]
pub fn _mm_crc32_u16(mut crc: u32, v: u16) -> u32 {
    crc = _mm_crc32_u8(crc, (v & 0xff) as u8);
    crc = _mm_crc32_u8(crc, (v >> 8) as u8);
    crc
}

/// Accumulate four bytes into a CRC-32C checksum. Matches `_mm_crc32_u32`.
#[inline]
pub fn _mm_crc32_u32(mut crc: u32, v: u32) -> u32 {
    crc = _mm_crc32_u16(crc, (v & 0xffff) as u16);
    crc = _mm_crc32_u16(crc, (v >> 16) as u16);
    crc
}

/// Accumulate eight bytes into a CRC-32C checksum. Matches `_mm_crc32_u64`.
///
/// x86 defines this as taking a 64-bit running value and returning 64 bits. The
/// checksum itself is 32 bits held in the low half.
#[inline]
pub fn _mm_crc32_u64(crc: u64, v: u64) -> u64 {
    let mut c = crc as u32;
    c = _mm_crc32_u32(c, (v & 0xffff_ffff) as u32);
    c = _mm_crc32_u32(c, (v >> 32) as u32);
    c as u64
}

/// Count set bits in a 32-bit value. Matches `_mm_popcnt_u32`.
#[inline]
pub fn _mm_popcnt_u32(a: u32) -> i32 {
    a.count_ones() as i32
}

/// Count set bits in a 64-bit value. Matches `_mm_popcnt_u64`.
#[inline]
pub fn _mm_popcnt_u64(a: u64) -> i64 {
    a.count_ones() as i64
}
