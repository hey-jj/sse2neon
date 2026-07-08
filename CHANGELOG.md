# Changelog

## [0.1.1] - 2026-07-07

### Fixed

- `_mm_loadl_epi64` and `_mm_storel_epi64` now work with unaligned 8 byte pointers. (#22, #23)

### Documentation

- The crate docs now state that float min and max functions propagate NaN operands and order `-0.0` below `+0.0`. (#24)
- The `_mm_crc32_u8` docs now name `0x82F63B78` as the reflected Castagnoli polynomial. (#25)

## [0.1.1] - 2026-07-07

### Fixed

- `_mm_loadl_epi64` and `_mm_storel_epi64` now work with unaligned 8 byte pointers. (#22, #23)

### Documentation

- The crate docs now state that float min and max functions propagate NaN operands and order `-0.0` below `+0.0`. (#24)
- The `_mm_crc32_u8` docs now name `0x82F63B78` as the reflected Castagnoli polynomial. (#25)
