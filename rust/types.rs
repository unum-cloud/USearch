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

use core::any::Any;
use core::cell::Cell;
use core::ffi::{c_char, c_int, c_void, CStr};
use core::fmt;
use core::future::{Future, IntoFuture};
use core::marker::PhantomData;
use core::mem::ManuallyDrop;
use core::ops::ControlFlow;
use core::pin::Pin;
use core::ptr::{self, NonNull};
use core::str::FromStr;
use core::task::{Context, Poll};
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};
use numkong::{Allocator, Matrix, MatrixView, StorageElement, TensorRef};

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

impl Default for usearch_bytes_view_t {
    fn default() -> Self {
        usearch_bytes_view_t {
            data: ptr::null(),
            length: 0,
        }
    }
}

/// A strided run of keys - how a batch names the identities it carries, or a cursor pull the
/// ones it wrote into its arena.
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

impl Default for usearch_keys_view_t {
    fn default() -> Self {
        usearch_keys_view_t {
            data: ptr::null(),
            stride: 0,
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

/// A run of rows along a call's batch axis, borrowed from the caller as an input and written
/// packed into an arena as an output.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_rows_view_t {
    pub data: *const c_void,
    pub stride: usize,
    pub columns: usize,
    pub scalar: usearch_scalar_t,
}

impl Default for usearch_rows_view_t {
    fn default() -> Self {
        usearch_rows_view_t {
            data: ptr::null(),
            stride: 0,
            columns: 0,
            scalar: 0,
        }
    }
}

/// A ranked run a call wrote into its arena - the call's count of rows, `columns` packed columns each.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_matches_view_t {
    pub keys: *const usearch_key_t,
    pub scores: *const usearch_distance_t,
    pub counts: *const usize,
    pub columns: usize,
}

impl Default for usearch_matches_view_t {
    fn default() -> Self {
        usearch_matches_view_t {
            keys: ptr::null(),
            scores: ptr::null(),
            counts: ptr::null(),
            columns: 0,
        }
    }
}

impl usearch_matches_view_t {
    /// Row `query`, trimmed to the matches the call wrote; a call with nothing to report leaves
    /// null counts, which read as empty rows.
    ///
    /// # Safety
    /// The view must be a finished call's output in host memory, and `query` below its count.
    pub(crate) unsafe fn row<'arena>(&self, query: usize) -> MatchesRow<'arena> {
        let written = match self.counts.is_null() {
            true => 0,
            false => (*self.counts.add(query)).min(self.columns),
        };
        let start = query * self.columns;
        MatchesRow {
            keys: arena_slice(self.keys.wrapping_add(start), written),
            distances: arena_slice(self.scores.wrapping_add(start), written),
        }
    }
}

/// A ragged run of matches a call wrote into its arena - keys and scores end to end, and one
/// more offset than rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_matches_tape_view_t {
    pub keys: *const usearch_key_t,
    pub scores: *const usearch_distance_t,
    pub offsets: *const usize,
}

impl Default for usearch_matches_tape_view_t {
    fn default() -> Self {
        usearch_matches_tape_view_t {
            keys: ptr::null(),
            scores: ptr::null(),
            offsets: ptr::null(),
        }
    }
}

impl usearch_matches_tape_view_t {
    /// Row `row`, spanning its two offsets; a call with nothing to report leaves null offsets,
    /// which read as empty rows.
    ///
    /// # Safety
    /// The view must be a finished call's output in host memory, and `row` below its count.
    pub(crate) unsafe fn row<'arena>(&self, row: usize) -> MatchesRow<'arena> {
        let (start, end) = match self.offsets.is_null() {
            true => (0, 0),
            false => (*self.offsets.add(row), *self.offsets.add(row + 1)),
        };
        MatchesRow {
            keys: arena_slice(self.keys.wrapping_add(start), end - start),
            distances: arena_slice(self.scores.wrapping_add(start), end - start),
        }
    }
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

pub type usearch_residency_t = c_int;
pub type usearch_traversal_t = c_int;

pub type usearch_arena_t = *mut c_void;
pub type usearch_executor_t = *mut c_void;

#[repr(C)]
pub struct usearch_arena_init_t {
    pub arena: *mut usearch_arena_t,
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
}

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
    pub fn usearch_outcome_wait(outcome: *mut usearch_outcome_t) -> usearch_status_t;
    pub fn usearch_outcome_status(outcome: *mut usearch_outcome_t) -> usearch_status_t;

    pub fn usearch_executor_for_ram(arguments: *mut usearch_executor_for_ram_t);
    pub fn usearch_executor_on_pool(arguments: *mut usearch_executor_on_pool_t);
    pub fn usearch_executor_for_vram(arguments: *mut usearch_executor_for_vram_t);
    pub fn usearch_executor_for_nvme(arguments: *mut usearch_executor_for_nvme_t);
    pub fn usearch_executor_drain(arguments: *mut usearch_executor_drain_t);
    pub fn usearch_executor_free(executor: usearch_executor_t);

    pub fn usearch_arena_init(arguments: *mut usearch_arena_init_t);
    pub fn usearch_arena_free(arena: usearch_arena_t);

    pub fn usearch_status_name(status: usearch_status_t) -> *const c_char;
    pub fn usearch_status_from_name(name: usearch_bytes_view_t) -> usearch_status_t;
    pub fn usearch_scalar_name(scalar: usearch_scalar_t) -> *const c_char;
    pub fn usearch_scalar_from_name(name: usearch_bytes_view_t) -> usearch_scalar_t;
    pub fn usearch_residency_name(residency: usearch_residency_t) -> *const c_char;
    pub fn usearch_residency_from_name(name: usearch_bytes_view_t) -> usearch_residency_t;
    pub fn usearch_traversal_name(traversal: usearch_traversal_t) -> *const c_char;
    pub fn usearch_traversal_from_name(name: usearch_bytes_view_t) -> usearch_traversal_t;
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
    /// call they feed is final, lending its handlers through `callbacks`.
    fn arguments(
        &mut self,
        outcome: *mut usearch_outcome_t,
        completion: usearch_completion_t,
        callbacks: &Callbacks,
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
/// Both borrow the arena, which no call reads as an input.
#[doc(hidden)]
#[diagnostic::on_unimplemented(
    message = "a deferred receiver needs owned inputs",
    label = "the library would still read this borrow after the verb returns",
    note = "pass keys in a `Vec` or an array, rows in a NumKong `Matrix` or a `'static` `MatrixView`, a `PathBuf`, \
            `String` or `'static` path, and `'static` configs, handlers and value lookups"
)]
pub trait Accepts<Input> {}

/// A matrix a verb reads its rows from, borrowed or owned.
#[doc(hidden)]
pub trait RowsView {
    /// The scalar type the rows hold.
    type Value: Element;
    /// The rows and their count, refused when the layout cannot cross the boundary.
    fn batch(&self) -> Result<Batch>;
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
    /// Dimensionality or row counts disagree.
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
    I32 = 12,

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
            12 => Scalar::I32,
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

/// What a matrix supplies to a descriptor, once the layout it must satisfy is checked.
struct MatrixLayout {
    /// Rows the matrix spans, which the call reads as its count.
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

impl<Value: Element, Alloc: Allocator> RowsView for MatrixView<'_, Value, Alloc> {
    type Value = Value;
    fn batch(&self) -> Result<Batch> { batch_of(self) }
}

impl<Value: Element, Alloc: Allocator> RowsView for Matrix<Value, Alloc> {
    type Value = Value;
    fn batch(&self) -> Result<Batch> { batch_of(self) }
}

impl<Rows: RowsView + ?Sized> RowsView for &Rows {
    type Value = Rows::Value;
    fn batch(&self) -> Result<Batch> { (**self).batch() }
}

// endregion: Matrix Views

// region: Matches

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

/// The ranked rows a search, a `maxsim` or a stream pull wrote into the arena it borrows; see
/// [`DenseVerbs::search`](crate::DenseVerbs::search).
///
/// Mirrors `usearch_matches_view_t`.
pub struct Matches<'arena> {
    pub(crate) arena: &'arena Arena,
    pub(crate) view: Exported<usearch_matches_view_t>,
    pub(crate) queries: usize,
}

impl<'arena> Matches<'arena> {
    /// Query rows the call resolved.
    pub fn queries(&self) -> usize { self.queries }

    /// Matches written for one query, refused with [`Status::InvalidArgument`] when `query` is not
    /// below [`Matches::queries`] and with [`Status::Device`] over device memory.
    pub fn row(&self, query: usize) -> Result<MatchesRow<'arena>> {
        self.arena.host()?;
        match query < self.queries {
            // Safety: the arena holds a finished call's output in host memory.
            true => Ok(unsafe { self.view.0.row(query) }),
            false => Err(fail(Status::InvalidArgument, "the row is not below the call's count")),
        }
    }

    /// Every query's row, in query order, refused with [`Status::Device`] over device memory.
    pub fn rows(&self) -> Result<impl Iterator<Item = MatchesRow<'arena>>> {
        self.arena.host()?;
        let view = self.view.0;
        // Safety: the arena holds a finished call's output in host memory.
        Ok((0..self.queries).map(move |query| unsafe { view.row(query) }))
    }
}

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

/// The handlers one call lends the library, reached by every trampoline through one state pointer,
/// and the first panic any of them raised, re-raised once the call is final since unwinding out of an
/// `extern "C"` frame aborts.
#[doc(hidden)]
pub struct Callbacks {
    pub(crate) predicate: Cell<*const c_void>,
    pub(crate) progress: Cell<*mut c_void>,
    pub(crate) lookup: Cell<*const c_void>,
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Default for Callbacks {
    fn default() -> Self {
        Callbacks {
            predicate: Cell::new(ptr::null()),
            progress: Cell::new(ptr::null_mut()),
            lookup: Cell::new(ptr::null()),
            panic: Mutex::new(None),
        }
    }
}

impl Callbacks {
    /// The state every trampoline of this call receives.
    pub(crate) fn state(&self) -> *mut c_void { ptr::from_ref(self).cast_mut().cast() }

    /// Runs a handler, answering `fallback` in place of a panic and latching the first one.
    pub(crate) fn guard<Answer>(&self, fallback: Answer, handler: impl FnOnce() -> Answer) -> Answer {
        std::panic::catch_unwind(AssertUnwindSafe(handler)).unwrap_or_else(|panic| {
            self.panic
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .get_or_insert(panic);
            fallback
        })
    }

    /// Re-raises the first panic a handler raised; called once the call is final.
    pub(crate) fn raise(self) {
        if let Some(panic) = self.panic.into_inner().unwrap_or_else(PoisonError::into_inner) {
            std::panic::resume_unwind(panic)
        }
    }
}

/// A periodic report of finished and total units; `Break` stops the call at the next increment
/// wherever the verb can stop, which an insertion cannot.
///
/// It runs on whichever thread advanced the work, never on two at once, so it owes `Send` and may
/// be `FnMut`. A panic inside it stops the call like a `Break` and is re-raised once the call is final.
///
/// Mirrors `usearch_progress_callback_t`.
pub type ProgressCallback<'a> = dyn FnMut(usize, usize) -> ControlFlow<()> + Send + 'a;

/// # Safety
/// `state` must address the live [`Callbacks`] of the call the callback was passed to, lending a
/// `&mut ProgressCallback`.
unsafe extern "C" fn progress_trampoline(finished: usize, total: usize, state: *mut c_void) -> c_int {
    let callbacks = &*state.cast::<Callbacks>();
    let handler = &mut **callbacks.progress.get().cast::<&mut ProgressCallback>();
    callbacks.guard(1, || c_int::from(handler(finished, total).is_break()))
}

/// Lends an optional handler as the callback and state pair every reporting verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_progress(
    handler: &mut Option<&mut ProgressCallback>,
    callbacks: &Callbacks,
) -> (usearch_progress_callback_t, *mut c_void) {
    match handler {
        Some(handler) => {
            callbacks.progress.set(ptr::from_mut(handler).cast());
            (Some(progress_trampoline), callbacks.state())
        }
        None => (None, ptr::null_mut()),
    }
}

/// A search filter, run on the executor's threads together; a panic rejects the key and is
/// re-raised once the call is final.
///
/// Mirrors `usearch_predicate_t`.
pub type Predicate<'a> = dyn Fn(Key) -> bool + Sync + 'a;

/// # Safety
/// `state` must address the live [`Callbacks`] of the call the callback was passed to, lending a `&Predicate`.
unsafe extern "C" fn admission_trampoline(key: Key, state: *mut c_void) -> c_int {
    let callbacks = &*state.cast::<Callbacks>();
    let predicate = *callbacks.predicate.get().cast::<&Predicate>();
    callbacks.guard(0, || c_int::from(predicate(key)))
}

/// Lends an optional predicate as the callback and state pair every filtering verb takes, so the
/// trampoline is never paired with the wrong state; `None` lends nulls.
pub(crate) fn lower_predicate(
    predicate: &Option<&Predicate>,
    callbacks: &Callbacks,
) -> (usearch_predicate_t, *mut c_void) {
    match predicate {
        Some(predicate) => {
            callbacks.predicate.set(ptr::from_ref(predicate).cast());
            (Some(admission_trampoline), callbacks.state())
        }
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

/// Where an executor computes, and with it where the arenas it makes keep their memory.
///
/// Mirrors `usearch_residency_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Residency {
    /// Volatile memory, traversed by CPU threads.
    Ram = 0,
    /// Accelerator memory, traversed by GPU warps on a chosen device.
    Vram = 1,
    /// Persisted storage, traversed through pluggable I/O engines.
    Nvme = 2,
}

impl Residency {
    fn from_raw(value: usearch_residency_t) -> Option<Self> {
        match value {
            0 => Some(Residency::Ram),
            1 => Some(Residency::Vram),
            2 => Some(Residency::Nvme),
            _ => None,
        }
    }
}

impl fmt::Display for Residency {
    /// Calls `usearch_residency_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_residency_name, *self as c_int))
    }
}

impl FromStr for Residency {
    type Err = Error;

    /// Calls `usearch_residency_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            usearch_residency_from_name,
            usearch_residency_name,
            Residency::from_raw,
        )
    }
}

/// How a search reaches the indexed rows it scores.
///
/// Mirrors `usearch_traversal_t`.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum Traversal {
    /// Walks the graph, scoring the rows the expansion reaches - approximate.
    #[default]
    Walk = 0,
    /// Scores every row published when the query's scan starts - exact, on RAM and NVMe handles.
    Scan = 1,
}

impl Traversal {
    fn from_raw(value: usearch_traversal_t) -> Option<Self> {
        match value {
            0 => Some(Traversal::Walk),
            1 => Some(Traversal::Scan),
            _ => None,
        }
    }
}

impl fmt::Display for Traversal {
    /// Calls `usearch_traversal_name`.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_traversal_name, *self as c_int))
    }
}

impl FromStr for Traversal {
    type Err = Error;

    /// Calls `usearch_traversal_from_name`.
    fn from_str(text: &str) -> Result<Self> {
        read_spelling(
            text,
            usearch_traversal_from_name,
            usearch_traversal_name,
            Traversal::from_raw,
        )
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
/// [`Arena::new_in`] places an arena where it computes, and the arena may outlive it.
/// Dropping an executor drains it.
///
/// Mirrors `usearch_executor_t`; dropping it calls `usearch_executor_free`.
#[derive(Debug)]
pub struct Executor {
    pub(crate) handle: usearch_executor_t,
    /// The residency of the constructor that produced it, which places the arenas it makes.
    residency: Residency,
}

// Safety: nothing in an executor is tied to the thread that created it, so it may move to another;
// staying out of `Sync` keeps submissions to it on one thread at a time.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self { Executor::null(Residency::Ram) }
}

impl Executor {
    /// An executor whose handle a constructor is about to fill.
    fn null(residency: Residency) -> Self {
        Executor {
            handle: ptr::null_mut(),
            residency,
        }
    }

    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local
    /// to `memory_domains`, which is the same list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains carry, since full occupancy starves
    /// the ordered publication path.
    ///
    /// Calls `usearch_executor_for_ram`.
    pub fn for_ram(memory_domains: &[i32], threads: usize, deferred_limit: usize) -> Result<Self> {
        let mut executor = Executor::null(Residency::Ram);
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
        let mut executor = Executor::null(Residency::Ram);
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
        let mut executor = Executor::null(Residency::Nvme);
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
        let mut executor = Executor::null(Residency::Vram);
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

/// Memory the library writes every output whose length scales with a call into, reused across calls.
///
/// An executor places it where it computes, host pages or its GPU's memory, decided once, and
/// nothing is allocated until a call writes into it. Every call rewinds and refills it,
/// invalidating the previous call's outputs, so a steady loop allocates once.
/// Every verb, blocking or deferred, borrows it as `&mut Arena` and answers outputs that borrow it in turn.
/// A [`Residency::Vram`] arena holds device memory, so the host accessors of its outputs refuse
/// with [`Status::Device`].
///
/// Mirrors `usearch_arena_t`; dropping it calls `usearch_arena_free`, which waits for the calls
/// still writing into it.
#[derive(Debug)]
pub struct Arena {
    pub(crate) handle: usearch_arena_t,
    /// The residency of the executor that made it.
    residency: Residency,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces,
// and a shared borrow reads only its residency.
unsafe impl Send for Arena {}
unsafe impl Sync for Arena {}

impl Arena {
    /// Makes an arena of host pages for calls on the calling thread.
    ///
    /// Calls `usearch_arena_init`.
    pub fn new() -> Result<Self> { Arena::new_in(&Executor::default()) }

    /// Makes an arena where `executor` computes, serving the calls of its residency.
    ///
    /// Calls `usearch_arena_init`.
    pub fn new_in(executor: &Executor) -> Result<Self> {
        let mut arena = Arena {
            handle: ptr::null_mut(),
            residency: executor.residency,
        };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_arena_init(&mut usearch_arena_init_t {
                arena: &mut arena.handle,
                executor: executor.handle,
                outcome: &mut outcome,
            })
        };
        check(outcome).map(|()| arena)
    }

    /// The residency of the executor that made it, whose memory it keeps.
    pub fn residency(&self) -> Residency { self.residency }

    /// Refuses device memory, which no host accessor may read.
    pub(crate) fn host(&self) -> Result<()> {
        match self.residency {
            Residency::Vram => Err(fail(Status::Device, "the outputs are device memory")),
            _ => Ok(()),
        }
    }
}

impl Drop for Arena {
    fn drop(&mut self) { unsafe { usearch_arena_free(self.handle) } }
}

/// A view the library wrote into an arena the verb or result borrows.
#[derive(Default)]
pub(crate) struct Exported<View>(pub(crate) View);

// Safety: the pointees live in the arena borrowed beside the view, and its borrows are `Send`.
unsafe impl<View> Send for Exported<View> {}

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
        let callbacks = Callbacks::default();
        let mut arguments = verb.arguments(&mut outcome, usearch_completion_blocking_k, &callbacks)?;
        // Safety: the arguments point into `verb`, `outcome` and `callbacks`, which stay in place until
        // the call returns, and a blocking call is final on return.
        unsafe { Verb::call(&mut arguments) };
        callbacks.raise();
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
            callbacks: Callbacks::default(),
            arguments: None,
            verb,
        }));
        // Safety: the box is fresh and reached through `lent` alone, and the arguments point into its
        // fields, where they stay until the box is freed.
        unsafe {
            let outcome = ptr::addr_of_mut!((*lent).outcome);
            let (callbacks, verb) = (&(*lent).callbacks, &mut (*lent).verb);
            match verb.arguments(outcome, usearch_completion_deferred_k, callbacks) {
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
    /// The handlers the call lends, and the first panic they raised.
    callbacks: Callbacks,
    /// The argument struct, pointing into `outcome` and `verb`, filled once the box is in place.
    arguments: Option<Verb::Arguments>,
    /// The inputs, the outputs and the arena borrow.
    verb: Verb,
}

/// A submitted call, which owns its outcome, argument struct and inputs in one box the library
/// reads until the call is final, and borrows its arena as a blocking call does.
///
/// [`Deferred::status`] polls without blocking, and [`Deferred::wait`] blocks until the call is
/// final and hands back what it produced, while `.await` polls it from async code. Dropping an
/// unfinished one waits for it, then frees the box.
/// A panic a handler raised resurfaces from `wait`, `.await`, or a drop outside unwinding.
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
    ///
    /// Calls `usearch_outcome_status`.
    pub fn status(&self) -> core::result::Result<(), Status> {
        // Safety: the outcome lives in the box this `Deferred` owns until it is final.
        Status::from_raw(unsafe { usearch_outcome_status(ptr::addr_of_mut!((*self.lent.as_ptr()).outcome)) })
    }

    /// Blocks until the call is final, then hands back what it produced.
    ///
    /// Calls `usearch_outcome_wait`.
    pub fn wait(self) -> Result<Verb::Output> {
        let lent = ManuallyDrop::new(self).lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, so it is freed here.
        let Lent {
            outcome,
            callbacks,
            verb,
            ..
        } = unsafe {
            usearch_outcome_wait(ptr::addr_of_mut!((*lent).outcome));
            *Box::from_raw(lent)
        };
        callbacks.raise();
        check(outcome)?;
        verb.finish()
    }
}

impl<Verb: Calling> Drop for Deferred<'_, Verb> {
    fn drop(&mut self) {
        let lent = self.lent.as_ptr();
        // Safety: once the outcome is final the library touches the box no more, and this owned it.
        let Lent { callbacks, .. } = unsafe {
            usearch_outcome_wait(ptr::addr_of_mut!((*lent).outcome));
            *Box::from_raw(lent)
        };
        // A second panic while unwinding would abort, so only a drop outside one re-raises.
        if !std::thread::panicking() {
            callbacks.raise();
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

impl<'index, Verb: Calling> IntoFuture for Deferred<'index, Verb> {
    type Output = Result<Verb::Output>;
    type IntoFuture = Wait<'index, Verb>;

    fn into_future(self) -> Wait<'index, Verb> { Wait(Some(self)) }
}

/// A [`Deferred`] being awaited: every poll reads its status, and one not yet final asks to be polled again.
///
/// The C library has no wake-up, so an executor re-polls it while other tasks run, spinning only
/// when nothing else is ready; [`Deferred::wait`] blocks without spinning.
///
/// Rust-only; the final poll calls `usearch_outcome_wait`, which returns at once.
#[must_use = "an awaited call does nothing unless polled"]
pub struct Wait<'index, Verb: Calling>(Option<Deferred<'index, Verb>>);

impl<Verb: Calling> Future for Wait<'_, Verb> {
    type Output = Result<Verb::Output>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let awaited = &mut self.get_mut().0;
        match awaited.take() {
            None => Poll::Ready(Err(fail(Status::InvalidArgument, "the call was already awaited"))),
            Some(call) if call.status() == Err(Status::Deferred) => {
                *awaited = Some(call);
                context.waker().wake_by_ref();
                Poll::Pending
            }
            Some(call) => Poll::Ready(call.wait()),
        }
    }
}

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
            Scalar::I32,
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
        assert_eq!(Scalar::from_raw(13), None);
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
        assert_eq!(Residency::Vram.to_string(), "vram");
        assert_eq!("nvme".parse(), Ok(Residency::Nvme));
        assert_eq!(Traversal::Scan.to_string(), "scan");
        assert_eq!("walk".parse(), Ok(Traversal::Walk));

        assert!("success".parse::<Status>().is_err(), "success names no failure");
        assert!("shape mismatch".parse::<Status>().is_err());
        assert!("unknown".parse::<Scalar>().is_err());
        assert!("parking".parse::<Mood>().is_err());
        assert!("gpu".parse::<Residency>().is_err());
        assert!("exact".parse::<Traversal>().is_err());
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
        // The width divides the slab, so a zero would be a division rather than an empty run.
        let failure = rows(&[1.0f32, 2.0], 0).unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
        assert_eq!(failure.to_string(), "a row needs at least one dimension");
    }

    /// An arena keeps the residency of the executor that made it, and the host reads only host pages.
    #[test]
    fn arenas_live_where_their_executor_computes() -> Result<()> {
        let local = Arena::new()?;
        assert_eq!(local.residency(), Residency::Ram);
        assert!(local.host().is_ok());

        let executor = Executor::for_ram(&[], 1, 0)?;
        let placed = Arena::new_in(&executor)?;
        drop(executor);
        assert_eq!(placed.residency(), Residency::Ram, "an arena outlives its executor");

        let device = Arena {
            handle: ptr::null_mut(),
            residency: Residency::Vram,
        };
        assert_eq!(device.host().unwrap_err().status, Status::Device);
        Ok(())
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
