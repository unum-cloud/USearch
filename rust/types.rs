//! Shared vocabulary of the USearch bindings - the Rust twin of `include/usearch/types.h`.
//!
//! Scalar types, matrices, and their traits come from NumKong, so a caller already holding a
//! `MatrixView` hands it straight to an index without a reshape or a copy.

use core::ffi::{c_char, c_int, c_void, CStr};
use core::fmt;
use std::path::{Path, PathBuf};

use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};
use numkong::{MatrixSpan, MatrixView, StorageElement};

// region: Raw Declarations

/// One-to-one mirror of `include/usearch/types.h`.
pub(crate) mod ffi {
    #![allow(non_camel_case_types)]
    // A one-to-one mirror declares every field, including the ones no method sets yet.
    #![allow(dead_code)]

    use core::ffi::{c_char, c_int, c_void};

    pub type usearch_distance_t = f64;

    /// The C enums lower to the platform `int`.
    pub type usearch_status_t = c_int;
    pub type usearch_scalar_t = c_int;
    pub type usearch_completion_t = c_int;
    pub type usearch_mood_t = c_int;

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_outcome_t {
        pub status: usearch_status_t,
        pub message: *const c_char,
    }

    impl Default for usearch_outcome_t {
        fn default() -> Self {
            usearch_outcome_t {
                status: 0,
                message: core::ptr::null(),
            }
        }
    }

    pub type usearch_key_t = super::Key;

    /// A borrowed run of bytes, never null-terminated by contract.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_bytes_view_t {
        pub data: *const u8,
        pub length: usize,
    }

    impl From<&str> for usearch_bytes_view_t {
        /// Borrows the text as the view the C interface accepts - no terminator, no copy.
        ///
        /// The view carries no lifetime, so it must not outlive the `str` it borrows.
        fn from(text: &str) -> Self {
            usearch_bytes_view_t {
                data: text.as_ptr(),
                length: text.len(),
            }
        }
    }

    impl From<&[u8]> for usearch_bytes_view_t {
        /// Borrows the bytes as the view the C interface accepts; it must not outlive them.
        fn from(bytes: &[u8]) -> Self {
            usearch_bytes_view_t {
                data: bytes.as_ptr(),
                length: bytes.len(),
            }
        }
    }

    /// A borrowed, strided run of keys - how a batch names the identities it carries.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_keys_view_t {
        pub data: *const usearch_key_t,
        pub stride: usize,
    }

    impl usearch_keys_view_t {
        /// Borrows a packed slice as the view the C interface accepts.
        pub fn of(keys: &[super::Key]) -> Self {
            usearch_keys_view_t {
                data: keys.as_ptr(),
                stride: core::mem::size_of::<usearch_key_t>(),
            }
        }
    }

    /// A caller-owned, writable run of keys - the mutable twin of `usearch_keys_view_t`, and the
    /// only bound a streaming verb has on how much it may write.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_keys_span_t {
        pub data: *mut usearch_key_t,
        pub stride: usize,
        pub capacity: usize,
    }

    impl usearch_keys_span_t {
        /// Lends a packed mutable slice as the span the C interface accepts.
        pub fn of(keys: &mut [super::Key]) -> Self {
            usearch_keys_span_t {
                data: keys.as_mut_ptr(),
                stride: core::mem::size_of::<usearch_key_t>(),
                capacity: keys.len(),
            }
        }
    }

    /// A borrowed run of rows along a call's batch axis, carrying the width the caller laid out.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_rows_view_t {
        pub data: *const c_void,
        pub stride: usize,
        pub columns: usize,
        pub scalar: usearch_scalar_t,
    }

    /// The writable twin of `usearch_rows_view_t`, for rows landing in memory the caller allocated.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_rows_span_t {
        pub data: *mut c_void,
        pub stride: usize,
        pub columns: usize,
        pub capacity: usize,
        pub scalar: usearch_scalar_t,
    }

    /// A borrowed strided matrix off the batch axis, carrying its own rows and columns.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_matrix_view_t {
        pub data: *const c_void,
        pub rows: usize,
        pub columns: usize,
        pub stride: usize,
        pub scalar: usearch_scalar_t,
    }

    /// Where a ranked run lands - three caller-owned outputs of the call's count of rows.
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_matches_span_t {
        pub keys: *mut usearch_key_t,
        pub keys_stride: usize,
        pub scores: *mut usearch_distance_t,
        pub scores_stride: usize,
        pub counts: *mut usize,
        pub capacity: usize,
        pub wanted: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_ram_options_t {
        pub memory_domains: *const i32,
        pub memory_domains_count: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_vram_options_t {
        pub devices: *const i32,
        pub devices_count: usize,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct usearch_nvme_options_t {
        pub paths: *const usearch_bytes_view_t,
        pub paths_count: usize,
    }

    pub type usearch_predicate_t = Option<unsafe extern "C" fn(usearch_key_t, *mut c_void) -> c_int>;

    /// Periodic advance report and the caller's chance to stop the call.
    pub type usearch_progress_callback_t = Option<unsafe extern "C" fn(usize, usize, *mut c_void) -> c_int>;

    pub type usearch_arena_t = *mut c_void;
    pub type usearch_executor_t = *mut c_void;

    #[repr(C)]
    pub struct usearch_executor_for_ram_t {
        pub executor: *mut usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub ram: usearch_ram_options_t,
        pub threads: usize,
        pub in_flight_limit: usize,
    }

    #[repr(C)]
    pub struct usearch_executor_on_pool_t {
        pub executor: *mut usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub ram: usearch_ram_options_t,
        pub pool: *mut c_void,
        pub in_flight_limit: usize,
    }

    #[repr(C)]
    pub struct usearch_executor_for_vram_t {
        pub executor: *mut usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub vram: usearch_vram_options_t,
        pub streams: *const *mut c_void,
        pub scratch_limit: usize,
        pub in_flight_limit: usize,
    }

    #[repr(C)]
    pub struct usearch_executor_for_nvme_t {
        pub executor: *mut usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub nvme: usearch_nvme_options_t,
        pub threads: usize,
        pub in_flight_limit: usize,
    }

    #[repr(C)]
    pub struct usearch_executor_drain_t {
        pub executor: usearch_executor_t,
        pub outcome: *mut usearch_outcome_t,
        pub mood: usearch_mood_t,
    }

    extern "C" {
        pub fn usearch_arena_free(arena: usearch_arena_t);

        pub fn usearch_executor_for_ram(arguments: *mut usearch_executor_for_ram_t);
        pub fn usearch_executor_on_pool(arguments: *mut usearch_executor_on_pool_t);
        pub fn usearch_executor_for_vram(arguments: *mut usearch_executor_for_vram_t);
        pub fn usearch_executor_for_nvme(arguments: *mut usearch_executor_for_nvme_t);
        pub fn usearch_executor_drain(arguments: *mut usearch_executor_drain_t);
        pub fn usearch_executor_free(executor: usearch_executor_t);

        pub fn usearch_version_name() -> *const c_char;
        pub fn usearch_capabilities_compiled() -> *const c_char;
        pub fn usearch_capabilities_available() -> *const c_char;
    }
}

// endregion: Raw Declarations

// region: Errors

/// What went wrong, in the vocabulary the C interface defines - one variant per
/// `usearch_status_*_k`, numbered alike.
///
/// The binding reports the same status the engine would for the same failure, so a caller
/// branches on the identity of the fault rather than on which side of the boundary saw it. An
/// [`Error`] never carries `Success` or `Pending`, which only a call's outcome passes through.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The call completed.
    Success = 0,
    /// An allocation or reservation could not be satisfied.
    OutOfMemory = 1,
    /// Dimensionality, row counts, or output capacity disagree.
    ShapeMismatch = 2,
    /// A stride or packing the engine cannot traverse.
    UnsupportedLayout = 3,
    /// A scalar, metric, or residency tag this build does not know.
    UnknownKind = 4,
    /// A mutating verb on a handle that serves its checkpoint in place.
    ReadOnly = 5,
    /// A file could not be opened, read, written, or parsed.
    Io = 6,
    /// A device could not be selected, or its memory could not be claimed.
    Device = 7,
    /// A caller-supplied representation could not be decoded.
    Malformed = 8,
    /// The library reported neither success nor a failure.
    Internal = 9,
    /// A hard design ceiling was reached, like the 32-bit ID space.
    Overflow = 10,
    /// The verb needs a prior successful reservation that never happened.
    Uninitialized = 11,
    /// The request contradicts an already-frozen configuration.
    ConfigMismatch = 12,
    /// A synchronization contract was violated - a full in-flight table, or an arena still serving
    /// another call.
    Contended = 13,
    /// A progress handler asked the operation to stop; what the finished increments leave behind
    /// is each verb's own contract.
    Cancelled = 14,
    /// The verb is declared for every residency but this one does not implement it.
    Unsupported = 15,
    /// The call was submitted and has not finished; never a final status.
    Pending = 16,
    /// An argument breaks the verb's contract.
    InvalidArgument = 17,
}

impl Status {
    /// Lifts a raw status, reading one this binding has no name for as the contract violation it is.
    fn from_raw(value: ffi::usearch_status_t) -> Self {
        match value {
            0 => Status::Success,
            1 => Status::OutOfMemory,
            2 => Status::ShapeMismatch,
            3 => Status::UnsupportedLayout,
            4 => Status::UnknownKind,
            5 => Status::ReadOnly,
            6 => Status::Io,
            7 => Status::Device,
            8 => Status::Malformed,
            9 => Status::Internal,
            10 => Status::Overflow,
            11 => Status::Uninitialized,
            12 => Status::ConfigMismatch,
            13 => Status::Contended,
            14 => Status::Cancelled,
            15 => Status::Unsupported,
            16 => Status::Pending,
            17 => Status::InvalidArgument,
            _ => Status::Internal,
        }
    }
}

/// A failure, whether the engine reported it or the binding caught it before the call.
///
/// Messages are static storage, so nothing allocates on the failure path.
#[derive(Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// Turns the outcome of a blocking call into a `Result`.
pub(crate) fn check(outcome: ffi::usearch_outcome_t) -> Result<()> {
    match Status::from_raw(outcome.status) {
        Status::Success => Ok(()),
        Status::Pending => Err(fail(
            Status::Internal,
            "a blocking call returned before its outcome was final",
        )),
        status => Err(Error {
            status,
            message: unsafe { static_string(outcome.message) },
        }),
    }
}

/// Reports a failure the binding caught before the call crossed the boundary, under the same
/// status the engine would have used had the ABI let it see the same thing.
pub(crate) fn fail(status: Status, message: &'static str) -> Error { Error { status, message } }

/// Borrows a static null-terminated string the library owns.
///
/// Every such string is ASCII by construction - error messages are C literals, the version is
/// rendered from digits, capability lists join names from a fixed ASCII table - so its bytes are
/// read as UTF-8 without a check that could only ever pass.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    if raw.is_null() {
        return "";
    }
    core::str::from_utf8_unchecked(CStr::from_ptr(raw).to_bytes())
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
        match self.0 {
            [0, 0, 0, 0, 0, 0, 0, 0, low @ ..] => Some(u64::from_be_bytes(low)),
            _ => None,
        }
    }
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
                return Err(fail(Status::Malformed, MALFORMED));
            }
            digits[written] = character.to_digit(16).ok_or(fail(Status::Malformed, MALFORMED))? as u8;
            written += 1;
        }
        if written != digits.len() {
            return Err(fail(Status::Malformed, MALFORMED));
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

/// Where an index's payload lives, which engine traverses it, and the hardware its shards are
/// placed across - one shard per named domain, device, or path.
///
/// Names where the index lives, never who walks it: threads and streams belong to the
/// [`Executor`] the bulk verbs take. An empty placement derives one from the machine.
#[derive(Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Residency {
    /// Volatile-memory index traversed by CPU threads - the concurrent baseline.
    Ram {
        /// NUMA node IDs to place shards on; empty derives every node the process may use.
        memory_domains: Vec<i32>,
    },
    /// Accelerator-memory index traversed by GPU warps, sharded across the named devices and
    /// never replicated.
    Vram {
        /// CUDA device ordinals to place shards on; empty derives one free device.
        devices: Vec<i32>,
    },
    /// Persisted index traversed through pluggable I/O engines, sized past RAM.
    ///
    /// A path is whatever identifies a device to the engine that can reach it - a filesystem
    /// path, a transport identifier, or a FUSE mount over object storage.
    Nvme {
        /// Devices or directories to place shards on; empty leaves every shard where its
        /// manifest points.
        paths: Vec<PathBuf>,
    },
}

impl Default for Residency {
    fn default() -> Self {
        Residency::Ram {
            memory_domains: Vec::new(),
        }
    }
}

/// The memory domains a RAM placement or executor names.
pub(crate) fn ram_options(memory_domains: &[i32]) -> ffi::usearch_ram_options_t {
    ffi::usearch_ram_options_t {
        memory_domains: memory_domains.as_ptr(),
        memory_domains_count: memory_domains.len(),
    }
}

/// The devices a VRAM placement or executor names.
pub(crate) fn vram_options(devices: &[i32]) -> ffi::usearch_vram_options_t {
    ffi::usearch_vram_options_t {
        devices: devices.as_ptr(),
        devices_count: devices.len(),
    }
}

/// The paths an NVMe placement or executor names, borrowing `views`, which must outlive the call.
pub(crate) fn nvme_options(views: &[ffi::usearch_bytes_view_t]) -> ffi::usearch_nvme_options_t {
    ffi::usearch_nvme_options_t {
        paths: views.as_ptr(),
        paths_count: views.len(),
    }
}

/// Borrows a path as the bytes the C interface exchanges, which must be UTF-8.
pub(crate) fn path_view(path: &Path) -> Result<ffi::usearch_bytes_view_t> {
    let path = path.to_str().ok_or(fail(Status::Io, "the path is not valid UTF-8"))?;
    Ok(path.into())
}

/// Stages paths as the one contiguous array of byte views the C interface reads.
pub(crate) fn path_views(paths: &[impl AsRef<Path>]) -> Result<Vec<ffi::usearch_bytes_view_t>> {
    paths.iter().map(|path| path_view(path.as_ref())).collect()
}

// endregion: Residency

// region: Progress

/// Whether the operation a handler is watching carries on or unwinds.
///
/// Progress handlers and predicates run on whichever thread advanced the work. A progress handler
/// is never called from two threads at once, so it owes `Send` and may be `FnMut`; a predicate runs
/// on many threads together, so it owes `Sync`. A panic inside a progress handler is caught and
/// read as [`Continuation::Cancel`], since unwinding out of an `extern "C"` frame aborts.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Continuation {
    /// Carry on to the next unit of work.
    Continue = 0,
    /// Stop; what the finished increments leave behind is each verb's own contract.
    Cancel = 1,
}

/// A periodic report of completed and total units, answering whether to carry on.
pub type Progress<'a> = dyn FnMut(usize, usize) -> Continuation + Send + 'a;

/// # Safety
/// `state` must address a live `&mut Progress` that outlives the call the callback was passed to.
unsafe extern "C" fn progress_trampoline(completed: usize, total: usize, state: *mut c_void) -> c_int {
    let handler = &mut **(state as *mut &mut Progress);
    let reported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(completed, total)));
    reported.unwrap_or(Continuation::Cancel) as c_int
}

/// Lends an optional handler as the callback and state pair every reporting verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_progress(handler: &mut Option<&mut Progress>) -> (ffi::usearch_progress_callback_t, *mut c_void) {
    match handler {
        Some(handler) => (Some(progress_trampoline), (handler as *mut &mut Progress).cast()),
        None => (None, core::ptr::null_mut()),
    }
}

// endregion: Progress

// region: Executors

/// What an executor does with its threads or streams once it goes idle.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Mood {
    /// Park them; the next call pays a wake-up.
    #[default]
    Chill = 0,
    /// Keep them spinning, trading power for the next call's latency.
    Grind = 1,
}

/// The host threads or device streams a verb runs on, and the traversal scratch each of them walks
/// with.
///
/// One executor serves any number of index handles, so separate categories of indexes share one
/// machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
/// constructor that produced it and must match the residency of every handle it drives.
///
/// [`Executor::default`] is the C interface's null executor: it owns nothing, and every call
/// handed it runs on the calling thread - single-threaded on RAM, on the handle's device and
/// default stream on VRAM.
///
/// Calls on one executor run in submission order, which is why it is `Send` but not `Sync`: a
/// thread that wants to drive its own calls spawns its own executor. Index handles stay freely
/// shareable. Every constructor's zero `in_flight_limit` takes the library's default size for the
/// table of deferred calls.
#[derive(Debug)]
pub struct Executor {
    handle: ffi::usearch_executor_t,
}

// Safety: an executor may be handed to another thread outright; what it forbids is two threads
// driving it at once, which the missing `Sync` already prevents.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            handle: core::ptr::null_mut(),
        }
    }
}

impl Executor {
    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local
    /// to `memory_domains`, which is the same list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, since full occupancy starves
    /// the ordered publication path.
    pub fn for_ram(memory_domains: &[i32], threads: usize, in_flight_limit: usize) -> Result<Self> {
        let mut executor = Executor::default();
        let mut outcome = ffi::usearch_outcome_t::default();
        unsafe {
            ffi::usearch_executor_for_ram(&mut ffi::usearch_executor_for_ram_t {
                executor: &mut executor.handle,
                outcome: &mut outcome,
                ram: ram_options(memory_domains),
                threads,
                in_flight_limit,
            })
        };
        check(outcome).map(|()| executor)
    }

    /// Creates a RAM executor over a borrowed ForkUnion pool, so several libraries in one process
    /// share one set of threads while each keeps its own scratch.
    ///
    /// # Safety
    /// `pool` must address a live ForkUnion `fu_pool_t` that outlives the executor.
    pub unsafe fn on_pool(memory_domains: &[i32], pool: *mut c_void, in_flight_limit: usize) -> Result<Self> {
        let mut executor = Executor::default();
        let mut outcome = ffi::usearch_outcome_t::default();
        ffi::usearch_executor_on_pool(&mut ffi::usearch_executor_on_pool_t {
            executor: &mut executor.handle,
            outcome: &mut outcome,
            ram: ram_options(memory_domains),
            pool,
            in_flight_limit,
        });
        check(outcome).map(|()| executor)
    }

    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the
    /// compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities.
    pub fn for_nvme(paths: &[impl AsRef<Path>], threads: usize, in_flight_limit: usize) -> Result<Self> {
        let views = path_views(paths)?;
        let mut executor = Executor::default();
        let mut outcome = ffi::usearch_outcome_t::default();
        unsafe {
            ffi::usearch_executor_for_nvme(&mut ffi::usearch_executor_for_nvme_t {
                executor: &mut executor.handle,
                outcome: &mut outcome,
                nvme: nvme_options(&views),
                threads,
                in_flight_limit,
            })
        };
        check(outcome).map(|()| executor)
    }

    /// Opens one submission stream on every named device, or borrows the caller's, all of them
    /// driven by the one host thread that calls the verbs - which is why this takes no thread count.
    ///
    /// `streams` is empty, or holds one CUDA stream per device in `devices` order. Zero
    /// `scratch_limit` lets scratch grow as geometry asks.
    ///
    /// # Safety
    /// Every stream must be a live CUDA stream on its device that outlives the executor.
    pub unsafe fn for_vram(
        devices: &[i32],
        streams: &[*mut c_void],
        scratch_limit: usize,
        in_flight_limit: usize,
    ) -> Result<Self> {
        if !streams.is_empty() && streams.len() != devices.len() {
            return Err(fail(Status::ShapeMismatch, "pass no streams or one per device"));
        }
        let mut executor = Executor::default();
        let mut outcome = ffi::usearch_outcome_t::default();
        ffi::usearch_executor_for_vram(&mut ffi::usearch_executor_for_vram_t {
            executor: &mut executor.handle,
            outcome: &mut outcome,
            vram: vram_options(devices),
            // The C side reads one stream per device from any non-null pointer.
            streams: match streams.is_empty() {
                true => core::ptr::null(),
                false => streams.as_ptr(),
            },
            scratch_limit,
            in_flight_limit,
        });
        check(outcome).map(|()| executor)
    }

    /// Waits for every call submitted to this executor, then adopts `mood` whenever it next goes
    /// idle; a failure is the executor's own, such as a lost device, never one of a drained call.
    pub fn drain(&self, mood: Mood) -> Result<()> {
        let mut outcome = ffi::usearch_outcome_t::default();
        unsafe {
            ffi::usearch_executor_drain(&mut ffi::usearch_executor_drain_t {
                executor: self.handle,
                outcome: &mut outcome,
                mood: mood as ffi::usearch_mood_t,
            })
        };
        check(outcome)
    }

    /// The raw handle the modality verbs pass across the boundary.
    pub(crate) fn handle(&self) -> ffi::usearch_executor_t { self.handle }
}

impl Drop for Executor {
    fn drop(&mut self) { unsafe { ffi::usearch_executor_free(self.handle) } }
}

/// Memory the library allocates for the outputs only a call can size, reused across calls.
///
/// [`Arena::default`] holds nothing; the first call handed it creates the memory where that call's
/// executor computes. Every later call rewinds and refills it, so a steady loop allocates once.
/// The outputs a call returns borrow the arena, so the next call that needs it waits until they
/// are dropped; to keep one call's outputs alive past the next, pass the next one another arena.
#[derive(Debug)]
pub struct Arena {
    handle: ffi::usearch_arena_t,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces.
unsafe impl Send for Arena {}

impl Default for Arena {
    fn default() -> Self {
        Arena {
            handle: core::ptr::null_mut(),
        }
    }
}

impl Arena {
    /// The slot a verb creates or rewinds the arena through.
    pub(crate) fn handle(&mut self) -> *mut ffi::usearch_arena_t { &mut self.handle }
}

impl Drop for Arena {
    fn drop(&mut self) { unsafe { ffi::usearch_arena_free(self.handle) } }
}

/// Borrows `length` elements the library wrote into an arena; a failed or empty call leaves null.
///
/// # Safety
/// `data` must address `length` initialized elements that live as long as the arena borrow.
pub(crate) unsafe fn arena_slice<'arena, Element>(data: *const Element, length: usize) -> &'arena [Element] {
    match length {
        0 => &[],
        length => core::slice::from_raw_parts(data, length),
    }
}

// endregion: Executors

// region: Matrix Views

/// Storage values one `dimensions`-wide row occupies, sub-byte packing included.
fn values_of<Scalar: StorageElement>(dimensions: usize) -> usize { dimensions.div_ceil(Scalar::dimensions_per_value()) }

/// Shape and byte strides of a packed slab of `dimensions`-wide rows.
fn packed_layout<Scalar: StorageElement>(values: usize, dimensions: usize) -> Result<([usize; 2], [isize; 2])> {
    let values_per_row = values_of::<Scalar>(dimensions);
    if values_per_row == 0 {
        return Err(fail(Status::ShapeMismatch, "a row needs at least one dimension"));
    }
    if !values.is_multiple_of(values_per_row) {
        return Err(fail(Status::ShapeMismatch, "the slab is not a whole number of rows"));
    }
    let element_bytes = core::mem::size_of::<Scalar>() as isize;
    Ok((
        [values / values_per_row, dimensions],
        [values_per_row as isize * element_bytes, element_bytes],
    ))
}

/// Borrows a packed slab as a row-major matrix of `dimensions`-wide rows.
///
/// Slabs whose rows are padded to a wider pitch - memory-mapped datasets, cache-line-aligned
/// arenas - go through `MatrixView::from_raw_parts` instead, which takes the byte stride.
pub fn rows<Scalar: StorageElement>(data: &[Scalar], dimensions: usize) -> Result<MatrixView<'_, Scalar>> {
    let (shape, strides) = packed_layout::<Scalar>(data.len(), dimensions)?;
    unsafe { MatrixView::from_raw_parts(data.as_ptr(), &shape, &strides) }
        .map_err(|_| fail(Status::ShapeMismatch, "a matrix holds two axes"))
}

/// The mutable sibling of [`rows`], for output slabs the index writes into.
pub fn rows_mut<Scalar: StorageElement>(data: &mut [Scalar], dimensions: usize) -> Result<MatrixSpan<'_, Scalar>> {
    let (shape, strides) = packed_layout::<Scalar>(data.len(), dimensions)?;
    unsafe { MatrixSpan::from_raw_parts(data.as_mut_ptr(), &shape, &strides) }
        .map_err(|_| fail(Status::ShapeMismatch, "a matrix holds two axes"))
}

// endregion: Matrix Views

// region: Introspection

/// The version of the linked library.
pub fn version() -> &'static str { unsafe { static_string(ffi::usearch_version_name()) } }

/// Comma-separated hardware capabilities the library was compiled with.
pub fn capabilities_compiled() -> &'static str { unsafe { static_string(ffi::usearch_capabilities_compiled()) } }

/// Comma-separated hardware capabilities this machine offers at runtime.
pub fn capabilities_available() -> &'static str { unsafe { static_string(ffi::usearch_capabilities_available()) } }

// endregion: Introspection

// region: Tests

#[cfg(test)]
mod tests {
    use core::str::FromStr;

    use numkong::TensorRef;

    use super::*;

    #[test]
    fn keys_round_trip_through_their_rendering() {
        let cases: [(Key, &str); 4] = [
            (Key::NIL, "00000000-0000-0000-0000-000000000000"),
            (Key::from(42u64), "00000000-0000-0000-0000-00000000002a"),
            (Key::from(u64::MAX), "00000000-0000-0000-ffff-ffffffffffff"),
            (Key::from(u128::MAX), "ffffffff-ffff-ffff-ffff-ffffffffffff"),
        ];
        for (key, rendered) in cases {
            assert_eq!(key.to_string(), rendered);
            assert_eq!(Key::from_str(rendered), Ok(key));
        }
        // Dashes are cosmetic, so a bare hex run parses too.
        assert_eq!(Key::from_str("0000000000000000000000000000002a"), Ok(Key::from(42u64)));
    }

    #[test]
    fn keys_reject_malformed_renderings() {
        for text in [
            "",
            "00000000-0000-0000-0000-00000000002",   // one digit short
            "00000000-0000-0000-0000-00000000002ab", // one digit long
            "0000000g-0000-0000-0000-00000000002a",  // not hexadecimal
        ] {
            assert!(Key::from_str(text).is_err(), "{text:?} must not parse");
        }
    }

    #[test]
    fn integer_identities_park_in_the_low_bytes() {
        assert_eq!(Key::from(42u64).as_u64(), Some(42));
        assert_eq!(Key::from(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(u128::from(Key::from(42u64)), 42);
        // A non-zero high half has no integer identity, and says so rather than truncating.
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
        let slab = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let matrix = rows(&slab, 3).unwrap();
        assert_eq!(matrix.shape(), &[2, 3]);
        assert_eq!(matrix.stride_bytes(0), 3 * core::mem::size_of::<f32>() as isize);
        assert!(matrix.has_contiguous_rows());
    }

    #[test]
    fn packed_slabs_count_sub_byte_rows_by_dimension() {
        // Sixteen logical dimensions occupy two bytes per row.
        let slab = [u1x8(0b1010_1010), u1x8(0b0101_0101), u1x8(0), u1x8(0)];
        let matrix = rows(&slab, 16).unwrap();
        assert_eq!(matrix.shape(), &[2, 16]);
        assert_eq!(matrix.stride_bytes(0), 2);
    }

    #[test]
    fn packed_slabs_reject_a_partial_row() {
        let failure = rows(&[1.0f32, 2.0, 3.0], 2).unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "the slab is not a whole number of rows");
    }

    #[test]
    fn packed_slabs_reject_a_zero_width() {
        // The width divides the slab, so a zero would be a division rather than an empty run -
        // the same guard an output span of zero wanted matches trips.
        let failure = rows_mut(&mut [1.0f32, 2.0], 0).unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "a row needs at least one dimension");
    }

    #[test]
    fn errors_carry_a_status_and_a_message() {
        let failure = fail(Status::ShapeMismatch, "something specific went wrong");
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "something specific went wrong");
        assert!(
            check(ffi::usearch_outcome_t::default()).is_ok(),
            "a zero status is success"
        );
    }

    #[test]
    fn statuses_survive_the_round_trip_and_tolerate_newer_libraries() {
        let failures = [
            Status::OutOfMemory,
            Status::ShapeMismatch,
            Status::UnsupportedLayout,
            Status::UnknownKind,
            Status::ReadOnly,
            Status::Io,
            Status::Device,
            Status::Malformed,
            Status::Internal,
            Status::Overflow,
            Status::Uninitialized,
            Status::ConfigMismatch,
            Status::Contended,
            Status::Cancelled,
            Status::Unsupported,
            Status::InvalidArgument,
        ];
        for status in failures {
            let raw = ffi::usearch_outcome_t {
                status: status as i32,
                message: core::ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, status);
        }
        // A blocking call never ends pending, and a status this build has no name for is not
        // success either, so both read as the contract violation they are.
        for status in [Status::Pending as i32, 4242] {
            let raw = ffi::usearch_outcome_t {
                status,
                message: core::ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, Status::Internal);
        }
    }

    #[test]
    fn placements_marshal_as_borrowed_lists() {
        let derived = ram_options(&[]);
        assert_eq!(derived.memory_domains_count, 0, "an empty list derives");

        let devices = [0, 1, 2, 3];
        let pinned = vram_options(&devices);
        assert_eq!(pinned.devices_count, 4);
        assert_eq!(unsafe { *pinned.devices.add(3) }, 3);

        let views = path_views(&["/mnt/nvme0", "/mnt/nvme1"]).unwrap();
        let drives = nvme_options(&views);
        assert_eq!(drives.paths_count, 2);
        assert_eq!(unsafe { (*drives.paths).length }, "/mnt/nvme0".len());
    }
}

// endregion: Tests
