//! Shared vocabulary of the USearch bindings - the Rust twin of `include/usearch/types.h`.
//!
//! Scalar types, matrices, and their traits come from NumKong, so a caller already holding a
//! `MatrixView` hands it straight to an index without a reshape or a copy.

use core::ffi::{c_char, CStr};
use core::fmt;

use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};
use numkong::{MatrixSpan, MatrixView, StorageElement};

// region: Raw Declarations

/// One-to-one mirror of `include/usearch/types.h`.
pub(crate) mod ffi {
    #![allow(non_camel_case_types)]

    use core::ffi::{c_char, c_int, c_void};

    pub type usearch_distance_t = f64;

    /// The `usearch_status_t` and `usearch_residency_t` enums lower to the platform `int`.
    pub type usearch_enum_t = i32;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_error_t {
        pub status: usearch_enum_t,
        pub message: *const c_char,
    }

    impl Default for usearch_error_t {
        fn default() -> Self {
            usearch_error_t {
                status: 0,
                message: core::ptr::null(),
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_key_t {
        pub bytes: [u8; 16],
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_string_t {
        pub data: *const c_char,
        pub length: usize,
    }

    impl From<&str> for usearch_string_t {
        /// Borrows the text as the string view the C interface accepts - no terminator, no copy.
        ///
        /// The view carries no lifetime, so it must not outlive the `str` it borrows.
        fn from(text: &str) -> Self {
            usearch_string_t {
                data: text.as_ptr() as *const c_char,
                length: text.len(),
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_ram_options_t {
        pub threads: usize,
        pub huge_pages: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_vram_options_t {
        pub device: i32,
        pub unified_addressing: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_nvme_options_t {
        pub sector_bytes: u32,
        pub queue_depth: u32,
        pub direct_io: bool,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub union usearch_residency_options_t {
        pub ram: usearch_ram_options_t,
        pub vram: usearch_vram_options_t,
        pub nvme: usearch_nvme_options_t,
    }

    pub type usearch_predicate_t = Option<unsafe extern "C" fn(usearch_key_t, *mut c_void) -> c_int>;

    extern "C" {
        pub fn usearch_version() -> *const c_char;
        pub fn usearch_capabilities_compiled() -> *const c_char;
        pub fn usearch_capabilities_available() -> *const c_char;
    }
}

// endregion: Raw Declarations

// region: Errors

/// What went wrong, in the vocabulary the C interface defines.
///
/// The binding reports the same status the engine would for the same failure, so a caller
/// branches on the identity of the fault rather than on which side of the boundary saw it.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// An allocation or reservation could not be satisfied.
    OutOfMemory = 1,
    /// Dimensionality, row counts, or output capacity disagree.
    ShapeMismatch = 2,
    /// A stride or packing the engine cannot traverse.
    UnsupportedLayout = 3,
    /// A scalar, metric, or residency tag this build does not know.
    UnknownKind = 4,
    /// A mutating verb on a handle that opens its checkpoint in place.
    ReadOnly = 5,
    /// A file could not be opened, read, written, or parsed.
    Io = 6,
    /// A device could not be selected, or its memory could not be claimed.
    Device = 7,
    /// A caller-supplied representation could not be decoded.
    MalformedInput = 8,
    /// The library reported neither success nor a failure.
    Internal = 9,
    /// A status a newer library reported and this binding has no name for.
    Unrecognized = -1,
}

impl Status {
    fn from_raw(value: ffi::usearch_enum_t) -> Self {
        match value {
            1 => Status::OutOfMemory,
            2 => Status::ShapeMismatch,
            3 => Status::UnsupportedLayout,
            4 => Status::UnknownKind,
            5 => Status::ReadOnly,
            6 => Status::Io,
            7 => Status::Device,
            8 => Status::MalformedInput,
            9 => Status::Internal,
            _ => Status::Unrecognized,
        }
    }
}

/// A failure, whether the engine reported it or the binding caught it before the call.
///
/// Messages are static storage, so nothing allocates on the failure path.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Error {
    /// The identity of the failure, shared with the C interface.
    pub status: Status,
    /// The detail the status alone cannot carry.
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { formatter.write_str(self.message) }
}

impl fmt::Debug for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Error({:?}, {:?})", self.status, self.message)
    }
}

impl std::error::Error for Error {}

/// The result of every fallible call in this crate.
pub type Result<Value> = core::result::Result<Value, Error>;

/// Reported by every verb that takes a path, since the C interface exchanges bytes.
pub(crate) const PATH_NOT_UTF8: &str = "the path is not valid UTF-8";

/// Turns a populated error slot into a `Result`.
pub(crate) fn check(error: ffi::usearch_error_t) -> Result<()> {
    if error.status == 0 {
        return Ok(());
    }
    Err(Error {
        status: Status::from_raw(error.status),
        message: unsafe { static_string(error.message) },
    })
}

/// Reports a failure the binding caught before the call crossed the boundary, under the same
/// status the engine would have used had the ABI let it see the same thing.
pub(crate) fn fail(status: Status, message: &'static str) -> Error { Error { status, message } }

/// Borrows a static null-terminated string the library owns.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    if raw.is_null() {
        return "";
    }
    CStr::from_ptr(raw).to_str().unwrap_or("")
}

// endregion: Errors

// region: Keys & Distances

/// Distance or similarity score in the metric of the index that produced it.
pub type Distance = ffi::usearch_distance_t;

/// A 128-bit entry identity, rendered as a lowercase dashed UUID.
///
/// Keys need not be unique: several entries may share one, and `count` reports how many.
/// Integer identities park in the low eight bytes, so `Key::from(42u64)` prints as
/// `00000000-0000-0000-0000-00000000002a`.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(
    /// The sixteen identity bytes, most significant first.
    pub [u8; 16],
);

impl Key {
    /// The all-zero key.
    pub const NIL: Key = Key([0; 16]);

    /// The integer identity in the low eight bytes, when the high eight are zero.
    pub fn as_u64(self) -> Option<u64> {
        let (high, low) = self.0.split_at(8);
        match high.iter().any(|&byte| byte != 0) {
            true => None,
            false => Some(u64::from_be_bytes(low.try_into().unwrap())),
        }
    }

    #[inline]
    pub(crate) fn to_raw(self) -> ffi::usearch_key_t { ffi::usearch_key_t { bytes: self.0 } }

    #[inline]
    pub(crate) fn from_raw(raw: ffi::usearch_key_t) -> Self { Key(raw.bytes) }
}

impl From<u64> for Key {
    fn from(value: u64) -> Self { Key::from(value as u128) }
}

impl From<u128> for Key {
    fn from(value: u128) -> Self { Key(value.to_be_bytes()) }
}

impl From<Key> for u128 {
    fn from(key: Key) -> Self { u128::from_be_bytes(key.0) }
}

impl core::str::FromStr for Key {
    type Err = Error;

    /// Parses the dashed-UUID rendering `Display` produces; dashes may sit anywhere.
    fn from_str(text: &str) -> Result<Self> {
        const MALFORMED: &str = "the key is not thirty-two hexadecimal digits";
        let mut digits = [0u8; 32];
        let mut written = 0usize;
        for character in text.chars() {
            if character == '-' {
                continue;
            }
            if written == digits.len() {
                return Err(fail(Status::MalformedInput, MALFORMED));
            }
            digits[written] = character.to_digit(16).ok_or(fail(Status::MalformedInput, MALFORMED))? as u8;
            written += 1;
        }
        if written != digits.len() {
            return Err(fail(Status::MalformedInput, MALFORMED));
        }
        let mut bytes = [0u8; 16];
        for (position, slot) in bytes.iter_mut().enumerate() {
            *slot = digits[position * 2] << 4 | digits[position * 2 + 1];
        }
        Ok(Key(bytes))
    }
}

impl fmt::Display for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (position, byte) in self.0.iter().enumerate() {
            if matches!(position, 4 | 6 | 8 | 10) {
                formatter.write_str("-")?;
            }
            write!(formatter, "{:02x}", byte)?;
        }
        Ok(())
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, formatter) }
}

// Lets result matrices be NumKong containers like every other output. The identities are the
// nil key and the integer one.
impl StorageElement for Key {
    fn zero() -> Self { Key::NIL }
    fn one() -> Self { Key::from(1u64) }
}

// endregion: Keys & Distances

// region: Scalars

/// Scalar types the indexes store and exchange, named by width and encoding.
///
/// Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell. The C enum
/// reserves the numbering gaps for kinds that have not landed yet, so this one is open.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum ScalarKind {
    /// The tag for a type the library could not name.
    #[default]
    Unknown = 0,

    F64 = 1,
    F32 = 2,
    BF16 = 3,
    F16 = 4,
    E5M2 = 5,
    E4M3 = 6,
    E3M2 = 7,
    E2M3 = 8,

    I8 = 10,
    I4x2 = 11,

    U8 = 20,
    U4x2 = 21,
    U1x8 = 22,
}

impl ScalarKind {
    pub(crate) fn from_raw(value: i32) -> Option<Self> {
        let kind = match value {
            1 => ScalarKind::F64,
            2 => ScalarKind::F32,
            3 => ScalarKind::BF16,
            4 => ScalarKind::F16,
            5 => ScalarKind::E5M2,
            6 => ScalarKind::E4M3,
            7 => ScalarKind::E3M2,
            8 => ScalarKind::E2M3,
            10 => ScalarKind::I8,
            11 => ScalarKind::I4x2,
            20 => ScalarKind::U8,
            21 => ScalarKind::U4x2,
            22 => ScalarKind::U1x8,
            _ => return None,
        };
        Some(kind)
    }
}

/// Ties a NumKong storage type to the scalar kind the C interface tags buffers with.
///
/// The stored quantization is chosen separately in the index options, so an `f32` query may
/// search an `i8` index - the library converts on the way in.
///
/// # Safety
/// The associated kind must describe the type's in-memory encoding exactly; the library reads
/// raw buffers under that tag.
pub unsafe trait Scalar: StorageElement {
    /// The tag the C interface knows this type by.
    const KIND: ScalarKind;
}

unsafe impl Scalar for f64 {
    const KIND: ScalarKind = ScalarKind::F64;
}

unsafe impl Scalar for f32 {
    const KIND: ScalarKind = ScalarKind::F32;
}

unsafe impl Scalar for bf16 {
    const KIND: ScalarKind = ScalarKind::BF16;
}

unsafe impl Scalar for f16 {
    const KIND: ScalarKind = ScalarKind::F16;
}

unsafe impl Scalar for e5m2 {
    const KIND: ScalarKind = ScalarKind::E5M2;
}

unsafe impl Scalar for e4m3 {
    const KIND: ScalarKind = ScalarKind::E4M3;
}

unsafe impl Scalar for e3m2 {
    const KIND: ScalarKind = ScalarKind::E3M2;
}

unsafe impl Scalar for e2m3 {
    const KIND: ScalarKind = ScalarKind::E2M3;
}

unsafe impl Scalar for i8 {
    const KIND: ScalarKind = ScalarKind::I8;
}

unsafe impl Scalar for i4x2 {
    const KIND: ScalarKind = ScalarKind::I4x2;
}

unsafe impl Scalar for u8 {
    const KIND: ScalarKind = ScalarKind::U8;
}

unsafe impl Scalar for u4x2 {
    const KIND: ScalarKind = ScalarKind::U4x2;
}

unsafe impl Scalar for u1x8 {
    const KIND: ScalarKind = ScalarKind::U1x8;
}

// endregion: Scalars

// region: Residency

/// Whether the RAM arenas demand a provisioned huge-page pool.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum HugePages {
    /// Ordinary pages, whatever the kernel provides.
    #[default]
    BasePages,
    /// Fail loudly when the pool is missing, rather than degrading silently.
    RequireHugePages,
}

/// Which accelerator hosts a VRAM index.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Device {
    /// The first device with free memory.
    #[default]
    FirstFree,
    /// A specific CUDA device ordinal.
    Ordinal(i32),
}

/// Whether a VRAM index may spill past the device's memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Addressing {
    /// Everything stays in device memory.
    #[default]
    DeviceOnly,
    /// Spill cold shards into host memory through the pager, trading traversal latency.
    Unified,
}

/// How an NVMe index reaches the block device.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum IoMode {
    /// O_DIRECT-class page-cache bypass - the default and the point.
    #[default]
    Direct,
    /// Ordinary buffered reads through the page cache.
    PageCached,
}

/// Where an index's payload lives and which engine traverses it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Residency {
    /// Volatile-memory index traversed by CPU threads - the concurrent baseline.
    Ram {
        /// Worker threads; zero takes 7/8 of the logical cores, since full occupancy starves
        /// the ordered publication path.
        threads: usize,
        /// Whether the arenas demand a provisioned huge-page pool.
        huge_pages: HugePages,
    },
    /// Accelerator-memory index traversed by GPU warps on a chosen device.
    Vram {
        /// Which accelerator hosts the index.
        device: Device,
        /// Whether the index may spill past the device's memory.
        addressing: Addressing,
    },
    /// Persisted index traversed through pluggable I/O engines, sized past RAM.
    ///
    /// External-memory graphs are never modified in place, so the handle is read-only and
    /// every mutating verb fails on it. Bigger indexes come from merging.
    Nvme {
        /// I/O granularity in bytes; zero probes the device. 512 fits one 256-d `i8` row.
        sector_bytes: u32,
        /// Submission ring depth; zero derives from the device's queue capabilities.
        queue_depth: u32,
        /// How the handle reaches the block device.
        io: IoMode,
    },
}

impl Default for Residency {
    fn default() -> Self {
        Residency::Ram {
            threads: 0,
            huge_pages: HugePages::BasePages,
        }
    }
}

impl Residency {
    /// The residency tag and the union member it selects.
    pub(crate) fn to_raw(self) -> (ffi::usearch_enum_t, ffi::usearch_residency_options_t) {
        match self {
            Residency::Ram { threads, huge_pages } => (
                0,
                ffi::usearch_residency_options_t {
                    ram: ffi::usearch_ram_options_t {
                        threads,
                        huge_pages: huge_pages == HugePages::RequireHugePages,
                    },
                },
            ),
            Residency::Vram { device, addressing } => (
                1,
                ffi::usearch_residency_options_t {
                    vram: ffi::usearch_vram_options_t {
                        device: match device {
                            Device::FirstFree => -1,
                            Device::Ordinal(ordinal) => ordinal,
                        },
                        unified_addressing: addressing == Addressing::Unified,
                    },
                },
            ),
            Residency::Nvme {
                sector_bytes,
                queue_depth,
                io,
            } => (
                2,
                ffi::usearch_residency_options_t {
                    nvme: ffi::usearch_nvme_options_t {
                        sector_bytes,
                        queue_depth,
                        direct_io: io == IoMode::Direct,
                    },
                },
            ),
        }
    }

    /// Reads the union member the tag selects.
    ///
    /// # Safety
    /// `tag` must be the one that populated `options`, as it is inside every options struct
    /// the C interface fills.
    pub(crate) unsafe fn from_raw(
        tag: ffi::usearch_enum_t,
        options: &ffi::usearch_residency_options_t,
    ) -> Result<Self> {
        let residency = match tag {
            0 => Residency::Ram {
                threads: options.ram.threads,
                huge_pages: match options.ram.huge_pages {
                    true => HugePages::RequireHugePages,
                    false => HugePages::BasePages,
                },
            },
            1 => Residency::Vram {
                device: match options.vram.device {
                    ordinal if ordinal < 0 => Device::FirstFree,
                    ordinal => Device::Ordinal(ordinal),
                },
                addressing: match options.vram.unified_addressing {
                    true => Addressing::Unified,
                    false => Addressing::DeviceOnly,
                },
            },
            2 => Residency::Nvme {
                sector_bytes: options.nvme.sector_bytes,
                queue_depth: options.nvme.queue_depth,
                io: match options.nvme.direct_io {
                    true => IoMode::Direct,
                    false => IoMode::PageCached,
                },
            },
            _ => {
                return Err(fail(
                    Status::UnknownKind,
                    "the checkpoint names a residency this binding does not know",
                ))
            }
        };
        Ok(residency)
    }
}

// endregion: Residency

// region: Matrix Views

/// Shape and byte strides of a packed slab of `dimensions`-wide rows.
fn packed_layout<Scalar: StorageElement>(values: usize, dimensions: usize) -> ([usize; 2], [isize; 2]) {
    let values_per_row = Scalar::dims_to_values(dimensions);
    assert!(values_per_row != 0, "a row needs at least one dimension");
    assert_eq!(values % values_per_row, 0, "the slab is not a whole number of rows");
    let element_bytes = core::mem::size_of::<Scalar>() as isize;
    (
        [values / values_per_row, dimensions],
        [values_per_row as isize * element_bytes, element_bytes],
    )
}

/// Borrows a packed slab as a row-major matrix of `dimensions`-wide rows.
///
/// Slabs whose rows are padded to a wider pitch - memory-mapped datasets, cache-line-aligned
/// arenas - go through `MatrixView::from_raw_parts` instead, which takes the byte stride.
pub fn rows<Scalar: StorageElement>(data: &[Scalar], dimensions: usize) -> MatrixView<'_, Scalar> {
    let (shape, strides) = packed_layout::<Scalar>(data.len(), dimensions);
    unsafe { MatrixView::from_raw_parts(data.as_ptr(), &shape, &strides) }
}

/// The mutable sibling of [`rows`], for output slabs the index writes into.
pub fn rows_mut<Scalar: StorageElement>(data: &mut [Scalar], dimensions: usize) -> MatrixSpan<'_, Scalar> {
    let (shape, strides) = packed_layout::<Scalar>(data.len(), dimensions);
    unsafe { MatrixSpan::from_raw_parts(data.as_mut_ptr(), &shape, &strides) }
}

// endregion: Matrix Views

// region: Introspection

/// The version of the linked library.
pub fn version() -> &'static str { unsafe { static_string(ffi::usearch_version()) } }

/// Comma-separated hardware capabilities the library was compiled with.
pub fn capabilities_compiled() -> &'static str { unsafe { static_string(ffi::usearch_capabilities_compiled()) } }

/// Comma-separated hardware capabilities this machine offers at runtime.
pub fn capabilities_available() -> &'static str { unsafe { static_string(ffi::usearch_capabilities_available()) } }

// endregion: Introspection

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn keys_render_as_dashed_uuids() {
        assert_eq!(Key::from(42u64).to_string(), "00000000-0000-0000-0000-00000000002a");
        assert_eq!(Key::NIL.to_string(), "00000000-0000-0000-0000-000000000000");
        assert_eq!(Key::from(u128::MAX).to_string(), "ffffffff-ffff-ffff-ffff-ffffffffffff");
    }

    #[test]
    fn keys_round_trip_through_their_rendering() {
        for value in [0u128, 42, u64::MAX as u128, u128::MAX] {
            let key = Key::from(value);
            assert_eq!(Key::from_str(&key.to_string()).expect("the rendering must parse"), key);
            assert_eq!(u128::from(key), value);
        }
        // Dashes are cosmetic, so a bare hex run parses too.
        assert_eq!(Key::from_str("0000000000000000000000000000002a"), Ok(Key::from(42u64)));
    }

    #[test]
    fn keys_reject_malformed_renderings() {
        assert!(Key::from_str("").is_err());
        assert!(
            Key::from_str("00000000-0000-0000-0000-00000000002").is_err(),
            "one digit short"
        );
        assert!(
            Key::from_str("00000000-0000-0000-0000-00000000002ab").is_err(),
            "one digit long"
        );
        assert!(
            Key::from_str("0000000g-0000-0000-0000-00000000002a").is_err(),
            "not hexadecimal"
        );
    }

    #[test]
    fn integer_identities_park_in_the_low_bytes() {
        assert_eq!(Key::from(42u64).as_u64(), Some(42));
        assert_eq!(Key::NIL.as_u64(), Some(0));
        assert_eq!(Key::from(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(
            Key::from(u128::MAX).as_u64(),
            None,
            "a non-zero high half has no integer identity"
        );
        assert_eq!(Key::from(1u128 << 64).as_u64(), None);
    }

    #[test]
    fn scalar_kinds_round_trip_through_their_tags() {
        let kinds = [
            ScalarKind::F64,
            ScalarKind::F32,
            ScalarKind::BF16,
            ScalarKind::F16,
            ScalarKind::E5M2,
            ScalarKind::E4M3,
            ScalarKind::E3M2,
            ScalarKind::E2M3,
            ScalarKind::I8,
            ScalarKind::I4x2,
            ScalarKind::U8,
            ScalarKind::U4x2,
            ScalarKind::U1x8,
        ];
        for kind in kinds {
            assert_eq!(ScalarKind::from_raw(kind as i32), Some(kind));
        }
        // The C enum's numbering gaps are reserved, not assigned.
        assert_eq!(ScalarKind::from_raw(0), None);
        assert_eq!(ScalarKind::from_raw(9), None);
        assert_eq!(ScalarKind::from_raw(12), None);
    }

    #[test]
    fn scalar_tags_match_their_storage_types() {
        assert_eq!(<f32 as Scalar>::KIND, ScalarKind::F32);
        assert_eq!(<bf16 as Scalar>::KIND, ScalarKind::BF16);
        assert_eq!(<i8 as Scalar>::KIND, ScalarKind::I8);
        assert_eq!(<u1x8 as Scalar>::KIND, ScalarKind::U1x8);
    }

    #[test]
    fn packed_slabs_borrow_as_matrices() {
        use numkong::TensorRef;

        let slab = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = rows(&slab, 3);
        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(matrix.stride_bytes(0), 3 * core::mem::size_of::<f32>() as isize);
        assert!(matrix.has_contiguous_rows());
    }

    #[test]
    fn packed_slabs_count_sub_byte_rows_by_dimension() {
        // Sixteen logical dimensions occupy two bytes per row.
        let slab = [u1x8(0b1010_1010), u1x8(0b0101_0101), u1x8(0), u1x8(0)];
        let matrix = rows(&slab, 16);
        assert_eq!(matrix.shape(), &[2, 16]);
        assert_eq!(matrix.stride_bytes(0), 2);
    }

    #[test]
    #[should_panic(expected = "whole number of rows")]
    fn packed_slabs_reject_a_partial_row() { let _ = rows(&[1.0f32, 2.0, 3.0], 2); }

    #[test]
    fn errors_carry_a_status_and_a_message() {
        let failure = fail(Status::ShapeMismatch, "something specific went wrong");
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "something specific went wrong");
        assert!(
            check(ffi::usearch_error_t::default()).is_ok(),
            "a zero status is success"
        );
    }

    #[test]
    fn statuses_survive_the_round_trip_and_tolerate_newer_libraries() {
        let statuses = [
            Status::OutOfMemory,
            Status::ShapeMismatch,
            Status::UnsupportedLayout,
            Status::UnknownKind,
            Status::ReadOnly,
            Status::Io,
            Status::Device,
            Status::MalformedInput,
            Status::Internal,
        ];
        for status in statuses {
            let raw = ffi::usearch_error_t {
                status: status as i32,
                message: core::ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, status);
        }
        // A status this build has no name for still reports as a failure rather than success.
        let newer = ffi::usearch_error_t {
            status: 4242,
            message: core::ptr::null(),
        };
        assert_eq!(check(newer).unwrap_err().status, Status::Unrecognized);
    }

    #[test]
    fn residencies_survive_the_raw_round_trip() {
        let residencies = [
            Residency::Ram {
                threads: 7,
                huge_pages: HugePages::RequireHugePages,
            },
            Residency::Vram {
                device: Device::Ordinal(0),
                addressing: Addressing::Unified,
            },
            Residency::Nvme {
                sector_bytes: 512,
                queue_depth: 128,
                io: IoMode::PageCached,
            },
        ];
        for residency in residencies {
            let (tag, options) = residency.to_raw();
            assert_eq!(unsafe { Residency::from_raw(tag, &options) }, Ok(residency));
        }
        let (_, options) = Residency::default().to_raw();
        assert!(
            unsafe { Residency::from_raw(9, &options) }.is_err(),
            "an unknown tag is not guessed"
        );
    }
}

// endregion: Tests
