//! Shared vocabulary of the USearch bindings - the Rust twin of `include/usearch/types.h`.
//!
//! Scalar types, matrices, and their traits come from NumKong, so a caller already holding a
//! `MatrixView` hands it straight to an index without a reshape or a copy.
//!
//! An [`On`] is the receiver every modality's verbs run through: it borrows an index and an
//! executor, and its completion decides whether a verb answers what the call produced or a
//! [`Deferred`] once the call is submitted.

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types, non_upper_case_globals)]

use core::ffi::{c_char, c_int, c_void, CStr};
use core::fmt;
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::ControlFlow;
use core::ptr::{self, NonNull};
use core::str::FromStr;
use core::sync::atomic::{AtomicI32, Ordering};
use std::path::{Path, PathBuf};

use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};
use numkong::{Allocator, Matrix, MatrixSpan, MatrixView, StorageElement, TensorMut, TensorRef};

// region: Raw Declarations

pub type usearch_key_t = Key;
pub type usearch_distance_t = f64;

/// A borrowed run of bytes, never null-terminated by contract.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_bytes_view_t {
    pub data: *const u8,
    pub length: usize,
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
    pub fn of(keys: &[Key]) -> Self {
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
    pub fn of(keys: &mut [Key]) -> Self {
        usearch_keys_span_t {
            data: keys.as_mut_ptr(),
            stride: core::mem::size_of::<usearch_key_t>(),
            capacity: keys.len(),
        }
    }
}

/// The C enums lower to the platform `int`.
pub type usearch_status_t = c_int;

pub const usearch_status_success_k: usearch_status_t = 0;
pub const usearch_status_deferred_k: usearch_status_t = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_outcome_t {
    pub status: usearch_status_t,
    pub message: *const c_char,
}

impl Default for usearch_outcome_t {
    fn default() -> Self {
        usearch_outcome_t {
            status: usearch_status_success_k,
            message: ptr::null(),
        }
    }
}

pub type usearch_scalar_t = c_int;

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

pub type usearch_completion_t = c_int;

pub const usearch_completion_blocking_k: usearch_completion_t = 0;
pub const usearch_completion_deferred_k: usearch_completion_t = 1;

pub type usearch_arena_t = *mut c_void;
pub type usearch_executor_t = *mut c_void;

#[repr(C)]
pub struct usearch_executor_for_ram_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub ram: usearch_ram_options_t,
    pub threads: usize,
    pub deferred_limit: usize,
}

#[repr(C)]
pub struct usearch_executor_on_pool_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub ram: usearch_ram_options_t,
    pub pool: *mut c_void,
    pub deferred_limit: usize,
}

#[repr(C)]
pub struct usearch_executor_for_vram_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub vram: usearch_vram_options_t,
    pub streams: *const *mut c_void,
    pub scratch_limit: usize,
    pub deferred_limit: usize,
}

#[repr(C)]
pub struct usearch_executor_for_nvme_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub nvme: usearch_nvme_options_t,
    pub threads: usize,
    pub deferred_limit: usize,
}

pub type usearch_mood_t = c_int;

#[repr(C)]
pub struct usearch_executor_drain_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub mood: usearch_mood_t,
}

extern "C" {
    pub fn usearch_outcome_wait(outcome: *const usearch_outcome_t) -> usearch_status_t;

    pub fn usearch_arena_free(arena: usearch_arena_t);

    pub fn usearch_executor_for_ram(arguments: *mut usearch_executor_for_ram_t);
    pub fn usearch_executor_on_pool(arguments: *mut usearch_executor_on_pool_t);
    pub fn usearch_executor_for_vram(arguments: *mut usearch_executor_for_vram_t);
    pub fn usearch_executor_for_nvme(arguments: *mut usearch_executor_for_nvme_t);
    pub fn usearch_executor_drain(arguments: *mut usearch_executor_drain_t);
    pub fn usearch_executor_free(executor: usearch_executor_t);

    pub fn usearch_status_name(status: usearch_status_t) -> *const c_char;
    pub fn usearch_status_from_name(name: usearch_bytes_view_t) -> usearch_status_t;
    pub fn usearch_scalar_name(scalar: usearch_scalar_t) -> *const c_char;
    pub fn usearch_scalar_from_name(name: usearch_bytes_view_t) -> usearch_scalar_t;
    pub fn usearch_mood_name(mood: usearch_mood_t) -> *const c_char;
    pub fn usearch_mood_from_name(name: usearch_bytes_view_t) -> usearch_mood_t;

    pub fn usearch_version_name() -> *const c_char;
    pub fn usearch_capabilities_compiled() -> *const c_char;
    pub fn usearch_capabilities_available() -> *const c_char;
}

// endregion: Raw Declarations

// region: Sealed Traits

/// The handles every call names: the index it reaches and the executor it runs on.
#[doc(hidden)]
#[derive(Debug)]
pub struct Handles {
    pub(crate) index: *mut c_void,
    pub(crate) executor: usearch_executor_t,
}

// Safety: an index is shared across threads, and an executor goes with the calls submitted to it.
unsafe impl Send for Handles {}

/// One verb over one `#[repr(C)]` argument struct: where the arguments point, which C entry
/// point takes them, and what a successful call hands back.
#[doc(hidden)]
pub trait Calling: Sized {
    /// The argument struct of the verb's C entry point.
    type Arguments;
    /// What a successful call hands back.
    type Output;

    /// The arguments over the inputs and outputs this verb holds, which stays in place until the
    /// call they feed is final.
    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
    ) -> Result<Self::Arguments>;

    /// Passes `arguments` to the verb's C entry point.
    ///
    /// # Safety
    /// Everything `arguments` points to stays alive and in place until the outcome is final.
    unsafe fn call(arguments: *mut Self::Arguments);

    /// What the verb hands back once its outcome is a success.
    fn finish(self) -> Result<Self::Output>;
}

/// How the verbs of a receiver complete, which decides what each of them answers.
#[doc(hidden)]
pub trait Completion: Sized {
    /// `Result<Output>` when blocking, `Result<Deferred<'index, Verb>>` when deferring.
    type Answer<'index, Verb: Calling>;

    /// Runs `verb`, or submits it.
    fn answer<'index, Verb: Calling>(verb: Verb) -> Self::Answer<'index, Verb>;
}

/// The inputs a completion takes: a blocking receiver takes borrows and owned values alike, while
/// a deferring one takes only what it owns, since the library reads its inputs after the verb returns.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "a deferred receiver needs owned inputs and an owned arena",
    label = "the library would still read this borrow after the verb returns",
    note = "pass keys in a `Vec` or an array, rows in a NumKong `Matrix` or a `'static` `MatrixView`, an owned \
            `Arena`, a `PathBuf`, `String` or `'static` path, and `'static` configs, handlers and value lookups"
)]
pub trait Accepts<Input> {}

/// A matrix a verb reads its rows from, borrowed or owned.
#[doc(hidden)]
pub trait RowsView {
    /// The rows and their count, refused when the layout cannot cross the boundary.
    fn batch(&self) -> Result<Batch>;
}

/// A matrix a verb writes its rows into, borrowed or owned, and what the call hands back of it.
#[doc(hidden)]
pub trait RowsSpan {
    /// The scalar type the rows land in.
    type Value: Element;
    /// The matrix itself when owned, nothing when borrowed.
    type Output;
    /// The span the call fills, refused when the layout cannot cross the boundary.
    fn span(&mut self) -> Result<usearch_rows_span_t>;
    /// Hands the matrix back once the call is final.
    fn finish(self) -> Self::Output;
}

/// Where a ranked call lands its matches, and what it hands back of them.
#[doc(hidden)]
pub trait Ranked {
    /// The matches themselves when the verb allocated them, nothing when the caller lent a span.
    type Output;
    /// The span the call fills for `queries` rows of up to `wanted` matches.
    fn span(&mut self, queries: usize, wanted: usize) -> Result<usearch_matches_span_t>;
    /// Hands the matches back once the call is final.
    fn finish(self) -> Self::Output;
}

// endregion: Sealed Traits

// region: Errors and Introspection

/// Every value the C status word takes besides success, numbered alike.
///
/// An [`Error`] never carries [`Status::Deferred`], which only [`Deferred::status`] reports.
///
/// Mirrors `usearch_status_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Status {
    /// The call was submitted deferred and is not final yet.
    Deferred = 1,
    /// An allocation or reservation could not be satisfied.
    OutOfMemory = 2,
    /// Dimensionality, row counts, or output capacity disagree.
    ShapeMismatch = 3,
    /// A stride or packing the engine cannot traverse.
    UnsupportedLayout = 4,
    /// A scalar, metric, or residency tag this build does not know.
    UnknownKind = 5,
    /// A mutating verb on a handle that serves its checkpoint in place.
    ReadOnly = 6,
    /// A file could not be opened, read, written, or parsed.
    Io = 7,
    /// A device could not be selected, or its memory could not be claimed.
    Device = 8,
    /// A caller-supplied representation could not be decoded.
    Malformed = 9,
    /// The library reported neither success nor a failure.
    Internal = 10,
    /// A hard design ceiling was reached, like the 32-bit ID space.
    Overflow = 11,
    /// The verb needs a prior successful reservation that never happened.
    Uninitialized = 12,
    /// The request contradicts an already-frozen configuration.
    ConfigMismatch = 13,
    /// The call was refused rather than waited for, like a submission to a full deferred table.
    Contended = 14,
    /// A progress handler stopped the call; what the finished increments leave behind is each
    /// verb's own contract.
    Cancelled = 15,
    /// The verb is declared for every residency but this one does not implement it.
    Unsupported = 16,
    /// An argument breaks the verb's contract.
    InvalidArgument = 17,
}

impl Status {
    /// Lifts the C status word, reading a value this binding has no name for as `Internal`.
    fn from_raw(value: usearch_status_t) -> core::result::Result<(), Self> {
        let status = match value {
            usearch_status_success_k => return Ok(()),
            usearch_status_deferred_k => Status::Deferred,
            2 => Status::OutOfMemory,
            3 => Status::ShapeMismatch,
            4 => Status::UnsupportedLayout,
            5 => Status::UnknownKind,
            6 => Status::ReadOnly,
            7 => Status::Io,
            8 => Status::Device,
            9 => Status::Malformed,
            11 => Status::Overflow,
            12 => Status::Uninitialized,
            13 => Status::ConfigMismatch,
            14 => Status::Contended,
            15 => Status::Cancelled,
            16 => Status::Unsupported,
            17 => Status::InvalidArgument,
            _ => Status::Internal,
        };
        Err(status)
    }
}

impl fmt::Display for Status {
    /// Calls `usearch_status_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_status_name, *self as c_int))
    }
}

impl FromStr for Status {
    type Err = Error;

    /// Calls `usearch_status_from_name`, refusing "success", which names no failure.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, usearch_status_from_name, usearch_status_name, |value| {
            Status::from_raw(value).err()
        })
    }
}

/// A failure, whether the engine reported it or the binding caught it before the call.
///
/// Messages are static storage, so nothing allocates on the failure path.
///
/// Mirrors a failed `usearch_outcome_t`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

impl std::error::Error for Error {}

/// The result of every fallible call in this crate.
pub type Result<Value> = core::result::Result<Value, Error>;

/// Turns a final outcome into a `Result`.
pub(crate) fn check(outcome: usearch_outcome_t) -> Result<()> {
    match Status::from_raw(outcome.status) {
        Ok(()) => Ok(()),
        Err(Status::Deferred) => Err(fail(Status::Internal, "the outcome was read before it was final")),
        Err(status) => Err(Error {
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
/// Every such string is ASCII by construction, so its bytes are read as UTF-8 unchecked.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    if raw.is_null() {
        return "";
    }
    core::str::from_utf8_unchecked(CStr::from_ptr(raw).to_bytes())
}

/// Spells an enumerator through its C `usearch_*_name`.
pub(crate) fn spelling(name: unsafe extern "C" fn(c_int) -> *const c_char, value: c_int) -> &'static str {
    unsafe { static_string(name(value)) }
}

/// Reads a spelling back through its C `usearch_*_from_name`, refusing one the C table spells
/// otherwise, since every such inverse answers a default for a name it does not know.
pub(crate) fn read_spelling<Value>(
    text: &str,
    from_name: unsafe extern "C" fn(usearch_bytes_view_t) -> c_int,
    name: unsafe extern "C" fn(c_int) -> *const c_char,
    lift: fn(c_int) -> Option<Value>,
) -> Result<Value> {
    let value = unsafe { from_name(text.as_bytes().into()) };
    match spelling(name, value) == text {
        true => lift(value),
        false => None,
    }
    .ok_or(fail(Status::InvalidArgument, "no enumerator is spelled that way"))
}

/// The version of the linked library.
///
/// Calls `usearch_version_name`.
pub fn version() -> &'static str { unsafe { static_string(usearch_version_name()) } }

/// Comma-separated hardware capabilities the library was compiled with.
///
/// Calls `usearch_capabilities_compiled`.
pub fn capabilities_compiled() -> &'static str { unsafe { static_string(usearch_capabilities_compiled()) } }

/// Comma-separated hardware capabilities this machine offers at runtime.
///
/// Calls `usearch_capabilities_available`.
pub fn capabilities_available() -> &'static str { unsafe { static_string(usearch_capabilities_available()) } }

// endregion: Errors and Introspection

// region: Keys and Distances

/// Distance or similarity score in the metric of the index that produced it.
///
/// Mirrors `usearch_distance_t`.
pub type Distance = usearch_distance_t;

/// A 128-bit entry identity, rendered as a lowercase dashed UUID.
///
/// Keys need not be unique: several entries may share one, and `count` reports how many.
/// Integer identities park in the low eight bytes, so `Key::from(42u64)` prints as
/// `00000000-0000-0000-0000-00000000002a`.
///
/// Mirrors `usearch_key_t`.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Key(
    /// The sixteen identity bytes, most significant first.
    pub [u8; 16],
);

impl Key {
    /// The all-zero key.
    pub const NIL: Key = Key([0; 16]);
}

impl From<u64> for Key {
    fn from(value: u64) -> Self { Key::from(u128::from(value)) }
}

impl From<u128> for Key {
    fn from(value: u128) -> Self { Key(value.to_be_bytes()) }
}

impl From<Key> for u128 {
    fn from(key: Key) -> Self { u128::from_be_bytes(key.0) }
}

impl TryFrom<Key> for u64 {
    type Error = core::num::TryFromIntError;

    /// The integer identity in the low eight bytes, refused when the high eight are not zero.
    fn try_from(key: Key) -> core::result::Result<Self, Self::Error> { u64::try_from(u128::from(key)) }
}

impl FromStr for Key {
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
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl fmt::Debug for Key {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Display::fmt(self, formatter) }
}

// endregion: Keys and Distances

// region: Scalars

/// Scalar types the indexes store and exchange, named by width and encoding.
///
/// Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell. The C enum
/// reserves the numbering gaps for kinds that have not landed yet, so this one is open.
///
/// Mirrors `usearch_scalar_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Scalar {
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

impl Scalar {
    pub(crate) fn from_raw(value: usearch_scalar_t) -> Option<Self> {
        let scalar = match value {
            1 => Scalar::F64,
            2 => Scalar::F32,
            3 => Scalar::BF16,
            4 => Scalar::F16,
            5 => Scalar::E5M2,
            6 => Scalar::E4M3,
            7 => Scalar::E3M2,
            8 => Scalar::E2M3,
            10 => Scalar::I8,
            11 => Scalar::I4x2,
            20 => Scalar::U8,
            21 => Scalar::U4x2,
            22 => Scalar::U1x8,
            _ => return None,
        };
        Some(scalar)
    }
}

impl fmt::Display for Scalar {
    /// Calls `usearch_scalar_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_scalar_name, *self as c_int))
    }
}

impl FromStr for Scalar {
    type Err = Error;

    /// Calls `usearch_scalar_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, usearch_scalar_from_name, usearch_scalar_name, Scalar::from_raw)
    }
}

/// Ties a NumKong storage type to the [`Scalar`] the C interface tags buffers with.
///
/// The stored quantization is chosen separately in the index options, so an `f32` query may
/// search an `i8` index - the library converts on the way in.
///
/// Rust-only.
///
/// # Safety
/// The associated scalar must describe the type's in-memory encoding exactly; the library reads
/// raw buffers under that tag.
pub unsafe trait Element: StorageElement + 'static {
    /// The tag the C interface knows this type by.
    const SCALAR: Scalar;
}

unsafe impl Element for f64 {
    const SCALAR: Scalar = Scalar::F64;
}

unsafe impl Element for f32 {
    const SCALAR: Scalar = Scalar::F32;
}

unsafe impl Element for bf16 {
    const SCALAR: Scalar = Scalar::BF16;
}

unsafe impl Element for f16 {
    const SCALAR: Scalar = Scalar::F16;
}

unsafe impl Element for e5m2 {
    const SCALAR: Scalar = Scalar::E5M2;
}

unsafe impl Element for e4m3 {
    const SCALAR: Scalar = Scalar::E4M3;
}

unsafe impl Element for e3m2 {
    const SCALAR: Scalar = Scalar::E3M2;
}

unsafe impl Element for e2m3 {
    const SCALAR: Scalar = Scalar::E2M3;
}

unsafe impl Element for i8 {
    const SCALAR: Scalar = Scalar::I8;
}

unsafe impl Element for i4x2 {
    const SCALAR: Scalar = Scalar::I4x2;
}

unsafe impl Element for u8 {
    const SCALAR: Scalar = Scalar::U8;
}

unsafe impl Element for u4x2 {
    const SCALAR: Scalar = Scalar::U4x2;
}

unsafe impl Element for u1x8 {
    const SCALAR: Scalar = Scalar::U1x8;
}

// endregion: Scalars

// region: Matrix Views

/// Storage values one `dimensions`-wide row occupies, sub-byte packing included.
pub(crate) fn values_of<Value: StorageElement>(dimensions: usize) -> usize {
    dimensions.div_ceil(Value::dimensions_per_value())
}

/// Shape and byte strides of a packed slab of `dimensions`-wide rows.
fn packed_layout<Value: StorageElement>(values: usize, dimensions: usize) -> Result<([usize; 2], [isize; 2])> {
    let values_per_row = values_of::<Value>(dimensions);
    if values_per_row == 0 {
        return Err(fail(Status::ShapeMismatch, "a row needs at least one dimension"));
    }
    if !values.is_multiple_of(values_per_row) {
        return Err(fail(Status::ShapeMismatch, "the slab is not a whole number of rows"));
    }
    let value_bytes = core::mem::size_of::<Value>() as isize;
    Ok((
        [values / values_per_row, dimensions],
        [values_per_row as isize * value_bytes, value_bytes],
    ))
}

/// Borrows a packed slab as a row-major matrix of `dimensions`-wide rows.
///
/// Slabs whose rows are padded to a wider pitch - memory-mapped datasets, cache-line-aligned
/// arenas - go through `MatrixView::from_raw_parts` instead, which takes the byte stride.
///
/// Rust-only.
pub fn rows<Value: StorageElement>(data: &[Value], dimensions: usize) -> Result<MatrixView<'_, Value>> {
    let (shape, strides) = packed_layout::<Value>(data.len(), dimensions)?;
    unsafe { MatrixView::from_raw_parts(data.as_ptr(), &shape, &strides) }
        .map_err(|_| fail(Status::ShapeMismatch, "a matrix holds two axes"))
}

/// The mutable sibling of [`rows`], for output slabs the index writes into.
///
/// Rust-only.
pub fn rows_mut<Value: StorageElement>(data: &mut [Value], dimensions: usize) -> Result<MatrixSpan<'_, Value>> {
    let (shape, strides) = packed_layout::<Value>(data.len(), dimensions)?;
    unsafe { MatrixSpan::from_raw_parts(data.as_mut_ptr(), &shape, &strides) }
        .map_err(|_| fail(Status::ShapeMismatch, "a matrix holds two axes"))
}

/// What a matrix supplies to a descriptor, once the layout it must satisfy is checked.
struct MatrixLayout {
    /// Rows the matrix spans, which a span reports as its capacity.
    rows: usize,
    /// Logical scalars per row, which the call checks against the frozen geometry.
    columns: usize,
    /// Bytes between the starts of consecutive rows.
    stride_bytes: usize,
}

impl MatrixLayout {
    /// Reads a matrix's shape, refusing the layouts the C interface cannot traverse.
    ///
    /// Only this side can answer whether rows are contiguous and ascend in memory, which NumKong
    /// knows and a bare pointer does not.
    fn of<Value: StorageElement>(matrix: &impl TensorRef<Value, 2>) -> Result<Self> {
        let &[rows, columns] = matrix.shape() else {
            return Err(fail(Status::ShapeMismatch, "a matrix holds two axes"));
        };
        if !matrix.has_contiguous_rows() {
            return Err(fail(
                Status::UnsupportedLayout,
                "expected a matrix whose rows are contiguous; only the pitch may stride",
            ));
        }
        let pitch = matrix.stride_bytes(0);
        if pitch < 0 {
            return Err(fail(Status::UnsupportedLayout, "matrix rows must ascend in memory"));
        }
        Ok(MatrixLayout {
            rows,
            columns,
            stride_bytes: pitch as usize,
        })
    }
}

/// A borrowed run of rows and the batch axis it spans.
pub struct Batch {
    pub(crate) rows: usearch_rows_view_t,
    pub(crate) count: usize,
}

/// Lowers a matrix as the batch of rows a verb reads, its width passed through for the call to check.
fn batch_of<Value: Element>(matrix: &impl TensorRef<Value, 2>) -> Result<Batch> {
    let layout = MatrixLayout::of(matrix)?;
    Ok(Batch {
        rows: usearch_rows_view_t {
            data: matrix.as_ptr().cast(),
            stride: layout.stride_bytes,
            columns: layout.columns,
            scalar: Value::SCALAR as usearch_scalar_t,
        },
        count: layout.rows,
    })
}

/// Lowers a writable matrix as the span of rows a verb fills.
pub(crate) fn span_of<Value: Element>(matrix: &mut impl TensorMut<Value, 2>) -> Result<usearch_rows_span_t> {
    let layout = MatrixLayout::of(matrix)?;
    Ok(usearch_rows_span_t {
        data: matrix.as_mut_ptr().cast(),
        stride: layout.stride_bytes,
        columns: layout.columns,
        capacity: layout.rows,
        scalar: Value::SCALAR as usearch_scalar_t,
    })
}

impl<Value: Element, Alloc: Allocator> RowsView for MatrixView<'_, Value, Alloc> {
    fn batch(&self) -> Result<Batch> { batch_of(self) }
}

impl<Value: Element, Alloc: Allocator> RowsView for Matrix<Value, Alloc> {
    fn batch(&self) -> Result<Batch> { batch_of(self) }
}

impl<Rows: RowsView + ?Sized> RowsView for &Rows {
    fn batch(&self) -> Result<Batch> { (**self).batch() }
}

impl<Value: Element, Alloc: Allocator> RowsSpan for &mut MatrixSpan<'_, Value, Alloc> {
    type Value = Value;
    type Output = ();
    fn span(&mut self) -> Result<usearch_rows_span_t> { span_of(*self) }
    fn finish(self) {}
}

impl<Value: Element, Alloc: Allocator> RowsSpan for &mut Matrix<Value, Alloc> {
    type Value = Value;
    type Output = ();
    fn span(&mut self) -> Result<usearch_rows_span_t> { span_of(*self) }
    fn finish(self) {}
}

impl<Value: Element, Alloc: Allocator> RowsSpan for Matrix<Value, Alloc> {
    type Value = Value;
    type Output = Self;
    fn span(&mut self) -> Result<usearch_rows_span_t> { span_of(self) }
    fn finish(self) -> Self { self }
}

// endregion: Matrix Views

// region: Matches

/// One retrieved entry.
///
/// Rust-only.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Match {
    /// The matched entry's key.
    pub key: Key,
    /// Its distance to the query, in the index's metric.
    pub distance: Distance,
}

/// One query's matches, or one left row's pairs, trimmed to the cells the call wrote.
///
/// Rust-only.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MatchesRow<'matches> {
    /// The keys written for this row: nearest first for a search, largest score first for a
    /// `maxsim`, and in traversal order for a `multiply`.
    pub keys: &'matches [Key],
    /// Positionally paired with the keys: distances for a search, summed similarities for a
    /// `maxsim`, and similarities for a `multiply`.
    pub distances: &'matches [Distance],
}

/// Reads one row of a `wanted`-wide ranked slab.
fn row_of<'slab>(
    keys: &'slab [Key],
    distances: &'slab [Distance],
    counts: &[usize],
    wanted: usize,
    query: usize,
) -> Option<MatchesRow<'slab>> {
    let start = query * wanted;
    let end = start + (*counts.get(query)?).min(wanted);
    Some(MatchesRow {
        keys: keys.get(start..end)?,
        distances: distances.get(start..end)?,
    })
}

/// Lends three packed `columns`-wide slabs as the span a ranked call writes, keeping up to
/// `wanted` matches per row.
pub(crate) fn matches_span(
    keys: &mut [Key],
    distances: &mut [Distance],
    counts: &mut [usize],
    columns: usize,
    wanted: usize,
) -> usearch_matches_span_t {
    usearch_matches_span_t {
        keys: keys.as_mut_ptr(),
        keys_stride: columns * core::mem::size_of::<Key>(),
        scores: distances.as_mut_ptr(),
        scores_stride: columns * core::mem::size_of::<Distance>(),
        counts: counts.as_mut_ptr(),
        capacity: counts.len(),
        wanted,
    }
}

/// Where a ranked call writes its results - three caller-owned buffers, sized together.
///
/// The buffer lengths decide the row width: `keys.len() / counts.len()` columns per query, which
/// may exceed the call's wanted count to land results inside a wider matrix.
///
/// Mirrors `usearch_matches_span_t`.
pub struct MatchesSpan<'buffers> {
    /// The `queries × wanted` key slab, rows contiguous.
    keys: &'buffers mut [Key],
    /// The `queries × wanted` distance slab, rows contiguous.
    distances: &'buffers mut [Distance],
    /// One match count per query.
    counts: &'buffers mut [usize],
    /// Columns per row.
    pub(crate) wanted: usize,
}

impl<'buffers> MatchesSpan<'buffers> {
    /// Lends three packed slabs: `counts.len()` rows of keys and distances, and one count per query.
    pub fn new(
        keys: &'buffers mut [Key],
        distances: &'buffers mut [Distance],
        counts: &'buffers mut [usize],
    ) -> Result<Self> {
        if counts.is_empty() {
            return Err(fail(Status::ShapeMismatch, "a run must hold at least one query"));
        }
        if keys.len() != distances.len() {
            return Err(fail(
                Status::ShapeMismatch,
                "the key and distance outputs are not the same shape",
            ));
        }
        if keys.is_empty() || !keys.len().is_multiple_of(counts.len()) {
            return Err(fail(
                Status::ShapeMismatch,
                "the outputs do not divide into one row per query",
            ));
        }
        let wanted = keys.len() / counts.len();
        Ok(MatchesSpan {
            keys,
            distances,
            counts,
            wanted,
        })
    }

    /// Query rows the buffers were sized for.
    pub fn queries(&self) -> usize { self.counts.len() }

    /// Matches written for one query, keys and distances read together, or `None` when `query`
    /// is not below [`MatchesSpan::queries`].
    pub fn row(&self, query: usize) -> Option<MatchesRow<'_>> {
        row_of(self.keys, self.distances, self.counts, self.wanted, query)
    }

    /// Every query's row, in query order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> {
        (0..self.queries()).filter_map(|query| self.row(query))
    }
}

impl Ranked for &mut MatchesSpan<'_> {
    type Output = ();

    fn span(&mut self, _queries: usize, wanted: usize) -> Result<usearch_matches_span_t> {
        Ok(matches_span(
            self.keys,
            self.distances,
            self.counts,
            self.wanted,
            wanted,
        ))
    }

    fn finish(self) {}
}

/// The matches a ranked call allocates when no span was lent to it; see
/// [`DenseVerbs::search`](crate::DenseVerbs::search).
///
/// Rust-only.
#[derive(Default)]
pub struct Matches {
    keys: Vec<Key>,
    distances: Vec<Distance>,
    counts: Vec<usize>,
    wanted: usize,
}

impl Matches {
    /// Query rows the call resolved.
    pub fn queries(&self) -> usize { self.counts.len() }

    /// Matches written for one query, or `None` when `query` is not below [`Matches::queries`].
    pub fn row(&self, query: usize) -> Option<MatchesRow<'_>> {
        row_of(&self.keys, &self.distances, &self.counts, self.wanted, query)
    }

    /// Every query's row, in query order.
    pub fn rows(&self) -> impl Iterator<Item = MatchesRow<'_>> {
        (0..self.queries()).filter_map(|query| self.row(query))
    }
}

impl Ranked for Matches {
    type Output = Matches;

    fn span(&mut self, queries: usize, wanted: usize) -> Result<usearch_matches_span_t> {
        let cells = queries.checked_mul(wanted).ok_or(fail(
            Status::Overflow,
            "the outputs would not fit an addressable buffer",
        ))?;
        *self = Matches {
            keys: vec![Key::NIL; cells],
            distances: vec![0.0; cells],
            counts: vec![0; queries],
            wanted,
        };
        Ok(matches_span(
            &mut self.keys,
            &mut self.distances,
            &mut self.counts,
            wanted,
            wanted,
        ))
    }

    fn finish(self) -> Matches { self }
}

impl Accepts<Matches> for Deferring {}

// endregion: Matches

// region: Placements

/// The memory domains a RAM placement or executor names.
pub(crate) fn ram_options(memory_domains: &[i32]) -> usearch_ram_options_t {
    usearch_ram_options_t {
        memory_domains: memory_domains.as_ptr(),
        memory_domains_count: memory_domains.len(),
    }
}

/// The devices a VRAM placement or executor names.
pub(crate) fn vram_options(devices: &[i32]) -> usearch_vram_options_t {
    usearch_vram_options_t {
        devices: devices.as_ptr(),
        devices_count: devices.len(),
    }
}

/// The paths an NVMe placement or executor names, borrowing `views`, which must outlive the call.
pub(crate) fn nvme_options(views: &[usearch_bytes_view_t]) -> usearch_nvme_options_t {
    usearch_nvme_options_t {
        paths: views.as_ptr(),
        paths_count: views.len(),
    }
}

/// Borrows a path as the bytes the C interface exchanges, which must be UTF-8.
pub(crate) fn path_view(path: &Path) -> Result<usearch_bytes_view_t> {
    let path = path.to_str().ok_or(fail(Status::Io, "the path is not valid UTF-8"))?;
    Ok(path.as_bytes().into())
}

/// Stages paths as the one contiguous array of byte views the C interface reads.
pub(crate) fn path_views(paths: &[impl AsRef<Path>]) -> Result<Vec<usearch_bytes_view_t>> {
    paths.iter().map(|path| path_view(path.as_ref())).collect()
}

// endregion: Placements

// region: Callbacks

/// A periodic report of finished and total units; `Break` stops the call at the next increment
/// wherever the verb can stop, which an insertion cannot.
///
/// It runs on whichever thread advanced the work, never on two at once, so it owes `Send` and may
/// be `FnMut`. A panic inside it is caught and read as `Break`, since unwinding out of an
/// `extern "C"` frame aborts.
///
/// Mirrors `usearch_progress_callback_t`.
pub type ProgressCallback<'a> = dyn FnMut(usize, usize) -> ControlFlow<()> + Send + 'a;

/// # Safety
/// `state` must address a live `&mut ProgressCallback` that outlives the call the callback was passed to.
unsafe extern "C" fn progress_trampoline(finished: usize, total: usize, state: *mut c_void) -> c_int {
    let handler = &mut **state.cast::<&mut ProgressCallback>();
    let reported = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(finished, total)));
    reported.map_or(1, |flow| c_int::from(flow.is_break()))
}

/// Lends an optional handler as the callback and state pair every reporting verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_progress(
    handler: &mut Option<&mut ProgressCallback>,
) -> (usearch_progress_callback_t, *mut c_void) {
    match handler {
        Some(handler) => (Some(progress_trampoline), ptr::from_mut(handler).cast()),
        None => (None, ptr::null_mut()),
    }
}

/// A search filter, run on the executor's threads together; a panic rejects the key.
///
/// Mirrors `usearch_predicate_t`.
pub type Predicate<'a> = dyn Fn(Key) -> bool + Sync + 'a;

/// # Safety
/// `state` must address a live `&Predicate` that outlives the call the callback was passed to.
unsafe extern "C" fn admission_trampoline(key: Key, state: *mut c_void) -> c_int {
    let predicate = *state.cast::<&Predicate>();
    // Unwinding out of an `extern "C"` frame aborts, so a panic rejects the key.
    c_int::from(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| predicate(key))).unwrap_or(false))
}

/// Lends an optional predicate as the callback and state pair every filtering verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_predicate(predicate: &Option<&Predicate>) -> (usearch_predicate_t, *mut c_void) {
    match predicate {
        Some(predicate) => (Some(admission_trampoline), ptr::from_ref(predicate).cast_mut().cast()),
        None => (None, ptr::null_mut()),
    }
}

// endregion: Callbacks

// region: Executors

/// What an executor does with its threads or streams once it goes idle.
///
/// Mirrors `usearch_mood_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Mood {
    /// Idle threads sleep; the next call pays a wake-up.
    #[default]
    Parked = 0,
    /// Idle threads stay hot, trading power for the next call's latency.
    Spinning = 1,
}

impl Mood {
    fn from_raw(value: usearch_mood_t) -> Option<Self> {
        match value {
            0 => Some(Mood::Parked),
            1 => Some(Mood::Spinning),
            _ => None,
        }
    }
}

impl fmt::Display for Mood {
    /// Calls `usearch_mood_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_mood_name, *self as c_int))
    }
}

impl FromStr for Mood {
    type Err = Error;

    /// Calls `usearch_mood_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(text, usearch_mood_from_name, usearch_mood_name, Mood::from_raw)
    }
}

/// The host threads or device streams an [`On`] runs its calls on, and the traversal scratch each
/// of them walks with.
///
/// One executor serves any number of index handles, so separate categories of indexes share one
/// machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
/// constructor that produced it and must match the residency of every handle it drives.
///
/// [`Executor::default`] is the C interface's null executor: it owns nothing, and every call
/// handed it runs on the calling thread.
///
/// Calls on one executor start in submission order, so threads wanting to overlap take one apiece;
/// it is `Send` but not `Sync`.
/// Every constructor takes a `deferred_limit` sizing its table of deferred calls, where zero takes
/// the default; a deferred submission finding the table full fails with [`Status::Contended`].
/// Dropping an executor drains it.
///
/// Mirrors `usearch_executor_t`; dropping it calls `usearch_executor_free`.
#[derive(Debug)]
pub struct Executor {
    pub(crate) handle: usearch_executor_t,
}

// Safety: nothing in an executor is tied to the thread that created it, so it may move to another;
// staying out of `Sync` keeps submissions to it on one thread at a time.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self {
        Executor {
            handle: ptr::null_mut(),
        }
    }
}

impl Executor {
    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local
    /// to `memory_domains`, which is the same list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, since full occupancy starves
    /// the ordered publication path.
    ///
    /// Calls `usearch_executor_for_ram`.
    pub fn for_ram(memory_domains: &[i32], threads: usize, deferred_limit: usize) -> Result<Self> {
        let mut executor = Executor::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_executor_for_ram(&mut usearch_executor_for_ram_t {
                executor: &mut executor.handle,
                outcome: &mut outcome,
                ram: ram_options(memory_domains),
                threads,
                deferred_limit,
            })
        };
        check(outcome).map(|()| executor)
    }

    /// Creates a RAM executor over a borrowed ForkUnion pool, so several libraries in one process
    /// share one set of threads while each keeps its own scratch.
    ///
    /// Calls `usearch_executor_on_pool`.
    ///
    /// # Safety
    /// `pool` must address a live ForkUnion `fu_pool_t` that outlives the executor.
    pub unsafe fn on_pool(memory_domains: &[i32], pool: *mut c_void, deferred_limit: usize) -> Result<Self> {
        let mut executor = Executor::default();
        let mut outcome = usearch_outcome_t::default();
        usearch_executor_on_pool(&mut usearch_executor_on_pool_t {
            executor: &mut executor.handle,
            outcome: &mut outcome,
            ram: ram_options(memory_domains),
            pool,
            deferred_limit,
        });
        check(outcome).map(|()| executor)
    }

    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the
    /// compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities.
    ///
    /// Calls `usearch_executor_for_nvme`.
    pub fn for_nvme(paths: &[impl AsRef<Path>], threads: usize, deferred_limit: usize) -> Result<Self> {
        let views = path_views(paths)?;
        let mut executor = Executor::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_executor_for_nvme(&mut usearch_executor_for_nvme_t {
                executor: &mut executor.handle,
                outcome: &mut outcome,
                nvme: nvme_options(&views),
                threads,
                deferred_limit,
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
    /// Calls `usearch_executor_for_vram`.
    ///
    /// # Safety
    /// Every stream must be a live CUDA stream on its device that outlives the executor.
    pub unsafe fn for_vram(
        devices: &[i32],
        streams: &[*mut c_void],
        scratch_limit: usize,
        deferred_limit: usize,
    ) -> Result<Self> {
        if !streams.is_empty() && streams.len() != devices.len() {
            return Err(fail(Status::ShapeMismatch, "pass no streams or one per device"));
        }
        let mut executor = Executor::default();
        let mut outcome = usearch_outcome_t::default();
        usearch_executor_for_vram(&mut usearch_executor_for_vram_t {
            executor: &mut executor.handle,
            outcome: &mut outcome,
            vram: vram_options(devices),
            // The C side reads one stream per device from any non-null pointer.
            streams: match streams.is_empty() {
                true => ptr::null(),
                false => streams.as_ptr(),
            },
            scratch_limit,
            deferred_limit,
        });
        check(outcome).map(|()| executor)
    }

    /// Waits for every call submitted to this executor, then adopts `mood` whenever it next goes
    /// idle; a failure is the executor's own, such as a lost device, never one of a drained call.
    ///
    /// Calls `usearch_executor_drain`.
    pub fn drain(&self, mood: Mood) -> Result<()> {
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_executor_drain(&mut usearch_executor_drain_t {
                executor: self.handle,
                outcome: &mut outcome,
                mood: mood as usearch_mood_t,
            })
        };
        check(outcome)
    }
}

impl Drop for Executor {
    fn drop(&mut self) { unsafe { usearch_executor_free(self.handle) } }
}

// endregion: Executors

// region: Arenas

/// Memory the library allocates for the outputs only a call can size, reused across calls.
///
/// [`Arena::default`] holds nothing; the first call handed it creates the memory where that call's
/// executor computes. Every later call rewinds and refills it, so a steady loop allocates once.
/// A verb given `&mut arena` answers outputs that borrow it, and one given an owned `Arena`
/// answers outputs that own it and hand it back through `into_arena`.
///
/// Mirrors `usearch_arena_t`; dropping it calls `usearch_arena_free`.
#[derive(Debug)]
pub struct Arena {
    pub(crate) handle: usearch_arena_t,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces.
unsafe impl Send for Arena {}

impl Default for Arena {
    fn default() -> Self {
        Arena {
            handle: ptr::null_mut(),
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) { unsafe { usearch_arena_free(self.handle) } }
}

/// An output the library exported into an arena the verb holds, which moves with that arena.
pub(crate) struct Exported<Value>(pub(crate) *mut Value);

impl<Value> Default for Exported<Value> {
    fn default() -> Self { Exported(ptr::null_mut()) }
}

// Safety: the pointee lives in the arena held beside it, which is `Send`.
unsafe impl<Value: Send> Send for Exported<Value> {}

/// Borrows `length` values the library wrote into an arena; a failed or empty call leaves null.
///
/// # Safety
/// `data` must address `length` initialized values that live as long as the arena borrow.
pub(crate) unsafe fn arena_slice<'arena, Value>(data: *const Value, length: usize) -> &'arena [Value] {
    match length {
        0 => &[],
        length => core::slice::from_raw_parts(data, length),
    }
}

// endregion: Arenas

// region: Receivers

/// An index borrowed with the executor its calls run on and how they complete, shared by every
/// modality.
///
/// Neither the receiver nor a [`Deferred`] it submits outlives the index or the executor.
///
/// Rust-only.
#[derive(Debug)]
pub struct On<'index, Index, Completion = Blocking> {
    pub(crate) index: &'index Index,
    pub(crate) executor: &'index Executor,
    pub(crate) completion: PhantomData<Completion>,
}

impl<Index, Completion> Clone for On<'_, Index, Completion> {
    fn clone(&self) -> Self { *self }
}

impl<Index, Completion> Copy for On<'_, Index, Completion> {}

/// The completion of a blocking receiver, whose verbs return once their call is final and answer
/// what it produced.
///
/// Rust-only.
#[derive(Debug)]
pub struct Blocking;

impl Completion for Blocking {
    type Answer<'index, Verb: Calling> = Result<Verb::Output>;

    fn answer<'index, Verb: Calling>(mut verb: Verb) -> Self::Answer<'index, Verb> {
        let mut outcome = usearch_outcome_t::default();
        let mut arguments = verb.arguments(&mut outcome, usearch_completion_blocking_k)?;
        // Safety: the arguments point into `verb` and `outcome`, which stay in place until the call
        // returns, and a blocking call is final on return.
        unsafe { Verb::call(&mut arguments) };
        check(outcome)?;
        verb.finish()
    }
}

impl<Input> Accepts<Input> for Blocking {}

/// The completion of a deferring receiver, whose verbs return once their call is submitted and
/// answer a [`Deferred`].
///
/// Rust-only.
#[derive(Debug)]
pub struct Deferring;

impl Completion for Deferring {
    type Answer<'index, Verb: Calling> = Result<Deferred<'index, Verb>>;

    fn answer<'index, Verb: Calling>(verb: Verb) -> Self::Answer<'index, Verb> {
        let lent = Box::into_raw(Box::new(Lent {
            outcome: usearch_outcome_t::default(),
            arguments: None,
            verb,
        }));
        // Safety: the box is fresh and reached through `lent` alone, and the arguments point into its
        // fields, where they stay until the box is freed.
        unsafe {
            let outcome = ptr::addr_of_mut!((*lent).outcome);
            match (*lent).verb.arguments(outcome, usearch_completion_deferred_k) {
                Ok(arguments) => Verb::call((*lent).arguments.insert(arguments)),
                Err(refusal) => {
                    drop(Box::from_raw(lent));
                    return Err(refusal);
                }
            }
            Ok(Deferred {
                lent: NonNull::new_unchecked(lent),
                receiver: PhantomData,
            })
        }
    }
}

impl Accepts<Arena> for Deferring {}
impl Accepts<Vec<Key>> for Deferring {}
impl<const COUNT: usize> Accepts<[Key; COUNT]> for Deferring {}
impl<Value, Alloc: Allocator> Accepts<Matrix<Value, Alloc>> for Deferring where Value: StorageElement {}
impl<Value, Alloc> Accepts<MatrixView<'static, Value, Alloc>> for Deferring {}
impl Accepts<PathBuf> for Deferring {}
impl Accepts<String> for Deferring {}
impl Accepts<&'static str> for Deferring {}
impl Accepts<&'static Path> for Deferring {}

/// One deferred call and everything it lends the library, at one heap address until it is final.
struct Lent<Verb: Calling> {
    /// The outcome the library finishes.
    outcome: usearch_outcome_t,
    /// The argument struct, pointing into `outcome` and `verb`, filled once the box is in place.
    arguments: Option<Verb::Arguments>,
    /// The inputs, the outputs and the arena.
    verb: Verb,
}

/// A submitted call, which owns its outcome, argument struct, inputs and arena in one box the
/// library reads until the call is final.
///
/// [`Deferred::status`] polls without blocking, and [`Deferred::wait`] blocks until the call is
/// final and hands back what it produced. Dropping an unfinished one waits for it, then frees the box.
/// `'index` is the borrow of the [`On`] that submitted it.
///
/// Rust-only; waiting and dropping call `usearch_outcome_wait`.
#[must_use = "a deferred call's outcome is seen only through `status` or `wait`"]
pub struct Deferred<'index, Verb: Calling> {
    lent: NonNull<Lent<Verb>>,
    receiver: PhantomData<&'index ()>,
}

impl<Verb: Calling> Deferred<'_, Verb> {
    /// `Err(Status::Deferred)` until the call is final, then its final status, read without blocking.
    pub fn status(&self) -> core::result::Result<(), Status> {
        // Safety: the status is an aligned `c_int` in the box this `Deferred` owns, which the library
        // stores atomically.
        let word = unsafe { AtomicI32::from_ptr(ptr::addr_of_mut!((*self.lent.as_ptr()).outcome.status)) };
        Status::from_raw(word.load(Ordering::Acquire))
    }

    /// Blocks until the call is final, then hands back what it produced.
    ///
    /// Calls `usearch_outcome_wait`.
    pub fn wait(self) -> Result<Verb::Output> {
        let lent = ManuallyDrop::new(self).lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, so it is freed here.
        let Lent { outcome, verb, .. } = unsafe {
            usearch_outcome_wait(ptr::addr_of!((*lent).outcome));
            *Box::from_raw(lent)
        };
        check(outcome)?;
        verb.finish()
    }
}

impl<Verb: Calling> Drop for Deferred<'_, Verb> {
    fn drop(&mut self) {
        let lent = self.lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, and this owned it.
        unsafe {
            usearch_outcome_wait(ptr::addr_of!((*lent).outcome));
            drop(Box::from_raw(lent));
        }
    }
}

impl<Verb: Calling> fmt::Debug for Deferred<'_, Verb> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "Deferred({:?})", self.status())
    }
}

// Safety: the box moves with the `Deferred`, and until its call is final only the library reaches it.
unsafe impl<Verb: Calling + Send> Send for Deferred<'_, Verb> {}

// endregion: Receivers

// region: Tests

#[cfg(test)]
mod tests {
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
        assert_eq!(u64::try_from(Key::from(42u64)), Ok(42));
        assert_eq!(u64::try_from(Key::from(u64::MAX)), Ok(u64::MAX));
        assert_eq!(u128::from(Key::from(42u64)), 42);
        // A non-zero high half has no integer identity, and says so rather than truncating.
        assert!(u64::try_from(Key::from(1u128 << 64)).is_err());
    }

    #[test]
    fn scalar_kinds_round_trip_through_their_tags() {
        let scalars = [
            Scalar::F64,
            Scalar::F32,
            Scalar::BF16,
            Scalar::F16,
            Scalar::E5M2,
            Scalar::E4M3,
            Scalar::E3M2,
            Scalar::E2M3,
            Scalar::I8,
            Scalar::I4x2,
            Scalar::U8,
            Scalar::U4x2,
            Scalar::U1x8,
        ];
        for scalar in scalars {
            assert_eq!(Scalar::from_raw(scalar as i32), Some(scalar));
            assert_eq!(scalar.to_string().parse(), Ok(scalar), "{scalar:?}");
        }
        // The C enum's numbering gaps are reserved, not assigned.
        assert_eq!(Scalar::from_raw(0), None);
        assert_eq!(Scalar::from_raw(9), None);
        assert_eq!(Scalar::from_raw(12), None);
    }

    /// Every C inverse answers a default for a name it does not know, which must not read back as
    /// that default.
    #[test]
    fn names_round_trip_through_the_library_and_refuse_strangers() {
        assert_eq!(Status::ShapeMismatch.to_string(), "shape-mismatch");
        assert_eq!("cancelled".parse(), Ok(Status::Cancelled));
        assert_eq!("invalid-argument".parse(), Ok(Status::InvalidArgument));
        assert_eq!(Mood::Spinning.to_string(), "spinning");
        assert_eq!("parked".parse(), Ok(Mood::Parked));

        assert!("success".parse::<Status>().is_err(), "success names no failure");
        assert!("shape mismatch".parse::<Status>().is_err());
        assert!("unknown".parse::<Scalar>().is_err());
        assert!("parking".parse::<Mood>().is_err());
    }

    #[test]
    fn scalar_tags_match_their_storage_types() {
        assert_eq!(<f32 as Element>::SCALAR, Scalar::F32);
        assert_eq!(<bf16 as Element>::SCALAR, Scalar::BF16);
        assert_eq!(<i8 as Element>::SCALAR, Scalar::I8);
        assert_eq!(<u1x8 as Element>::SCALAR, Scalar::U1x8);
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

    /// The row width follows from the buffer lengths, so lengths that cannot be read as one
    /// shape are refused before any verb sees them.
    #[test]
    fn rejects_outputs_that_are_not_one_shape() {
        let mut found = vec![Key::NIL; 8];
        let mut scores = vec![0.0 as Distance; 8];
        let mut counts = vec![0usize; 3];
        let mut queryless: [usize; 0] = [];
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores, &mut queryless),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "no queries to write for"
        );
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores[..6], &mut counts),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "the keys and distances disagree"
        );
        assert!(
            matches!(
                MatchesSpan::new(&mut found, &mut scores, &mut counts),
                Err(Error {
                    status: Status::ShapeMismatch,
                    ..
                })
            ),
            "eight cells do not divide into three rows"
        );
    }

    #[test]
    fn errors_carry_a_status_and_a_message() {
        let failure = fail(Status::ShapeMismatch, "something specific went wrong");
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "something specific went wrong");
        assert!(check(usearch_outcome_t::default()).is_ok(), "a zero status is success");
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
            let raw = usearch_outcome_t {
                status: status as i32,
                message: ptr::null(),
            };
            assert_eq!(check(raw).unwrap_err().status, status);
        }
        // A final outcome is never deferred, and a status this build has no name for is not
        // success either, so both read as the contract violation they are.
        for status in [Status::Deferred as i32, 4242] {
            let raw = usearch_outcome_t {
                status,
                message: ptr::null(),
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
