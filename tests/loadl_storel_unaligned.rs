use sse2neon::*;

#[test]
fn loadl_epi64_accepts_unaligned_8_byte_pointer() {
    let bytes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let read_offset = (1..=8)
        .find(|offset| (bytes.as_ptr() as usize + offset) % core::mem::align_of::<i64>() != 0)
        .unwrap();
    let read_p = unsafe { bytes.as_ptr().add(read_offset) } as *const __m128i;

    assert_ne!((read_p as usize) % core::mem::align_of::<i64>(), 0);

    let loaded = unsafe { _mm_loadl_epi64(read_p) };
    let mut loaded_bytes = [0u8; 16];
    unsafe { _mm_storeu_si128(loaded_bytes.as_mut_ptr() as *mut __m128i, loaded) };

    assert_eq!(&loaded_bytes[..8], &bytes[read_offset..read_offset + 8]);
    assert_eq!(&loaded_bytes[8..], &[0; 8]);
}

#[test]
fn storel_epi64_accepts_unaligned_8_byte_pointer() {
    let value = _mm_set_epi8(
        25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
    );
    let mut out = [0u8; 16];
    let write_offset = (1..=8)
        .find(|offset| (out.as_ptr() as usize + offset) % core::mem::align_of::<i64>() != 0)
        .unwrap();
    let write_p = unsafe { out.as_mut_ptr().add(write_offset) } as *mut __m128i;

    assert_ne!((write_p as usize) % core::mem::align_of::<i64>(), 0);

    unsafe { _mm_storel_epi64(write_p, value) };

    assert_eq!(&out[..write_offset], &[0; 8][..write_offset]);
    assert_eq!(
        &out[write_offset..write_offset + 8],
        &[10, 11, 12, 13, 14, 15, 16, 17]
    );
    assert_eq!(&out[write_offset + 8..], &[0; 8][..8 - write_offset]);
}
