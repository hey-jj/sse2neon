# sse2neon

x86 SSE-family SIMD intrinsics reimplemented on ARM NEON.

The crate mirrors the Intel `_mm_*` intrinsic surface on top of
`core::arch::aarch64` NEON. Code written against x86 SSE, SSE2, SSE3, SSSE3,
SSE4.1, SSE4.2, AES, and CRC32C can call these functions and get matching lane
semantics on AArch64. Lane ordering, saturation, NaN handling, and the x86
"integer indefinite" conversion rule all follow the Intel definitions.

The SSE4.2 surface covers 64-bit compare (`_mm_cmpgt_epi64`), CRC32C
(`_mm_crc32_u8/u16/u32/u64`), and population count (`_mm_popcnt_u32/u64`). The
packed string-compare instructions (`_mm_cmpistr*`, `_mm_cmpestr*`) are not
provided.

## Platform

AArch64 only, little-endian. The intrinsics use NEON registers present on ARMv8.
On other targets the crate compiles to an empty surface. The crate is
`#![no_std]` and pulls in no allocator.

## Types

- `__m128`  four `f32` lanes
- `__m128d` two `f64` lanes
- `__m128i` a 128-bit integer vector
- `__m64`   a 64-bit integer vector

Lane 0 is the least significant. `_mm_set_ps(w, z, y, x)` places `x` in lane 0.
`_mm_setr_*` reverses the argument order.

## Example

```rust
use sse2neon::*;

let a = _mm_set_ps(4.0, 3.0, 2.0, 1.0);
let b = _mm_set_ps(8.0, 7.0, 6.0, 5.0);
let sum = _mm_add_ps(a, b);
let mut out = [0.0f32; 4];
unsafe { _mm_storeu_ps(out.as_mut_ptr(), sum) };
assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
```

## Installation

```toml
[dependencies]
sse2neon = "0.1"
```

## License

Licensed under the [MIT license](LICENSE).
