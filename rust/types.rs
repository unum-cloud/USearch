//! Shared vocabulary of every USearch modality.
//!
//! Scalar types, matrices, and their traits come from NumKong, so a caller already holding a
//! `MatrixView` hands it straight to an index without a reshape or a copy.
//!
//! An [`On`] is the receiver every modality's verbs run through: it refers to an index and an
//! executor, and its completion decides whether a verb answers what the call produced or a
//! [`Deferred`] that submits the call once polled or waited on.
//!
//! File: rust/types.rs
//! Author: Ash Vardanian
//! Date: August 15, 2026

// The C mirror keeps the C spelling of every name.
#![allow(non_camel_case_types, non_upper_case_globals)]

use core::cell::UnsafeCell;
use core::ffi::{CStr, c_char, c_int, c_void};
use core::fmt;
use core::future::Future;
use core::marker::{PhantomData, PhantomPinned};
use core::mem::{ManuallyDrop, MaybeUninit};
use core::ops::ControlFlow;
use core::pin::{Pin, pin};
use core::ptr;
use core::str::FromStr;
use core::task::{Context, Poll};
use std::borrow::Cow;
use std::path::{Path, PathBuf};

use numkong::{Allocator, Matrix, MatrixView, StorageElement, TensorRef};
use numkong::{bf16, e2m3, e3m2, e4m3, e5m2, f16, i4x2, u1x8, u4x2};

// region: Raw Declarations

pub type usearch_key_t = Key;
pub type usearch_distance_t = f64;

/// A run of bytes the caller owns, never null-terminated by contract.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_bytes_view_t {
    pub data: *const u8,
    pub length: usize,
}

impl From<&[u8]> for usearch_bytes_view_t {
    /// Views the bytes as the C interface accepts them; the view must not outlive them.
    fn from(bytes: &[u8]) -> Self {
        usearch_bytes_view_t { data: bytes.as_ptr(), length: bytes.len() }
    }
}

impl Default for usearch_bytes_view_t {
    fn default() -> Self {
        usearch_bytes_view_t { data: ptr::null(), length: 0 }
    }
}

// Safety: it views bytes as `&[u8]` does, which crosses threads freely.
unsafe impl Send for usearch_bytes_view_t {}

/// A strided run of keys - how a batch names the identities it holds, or a cursor pull the
/// ones it wrote into its arena.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_keys_view_t {
    pub data: *const usearch_key_t,
    pub stride: usize,
}

impl usearch_keys_view_t {
    /// Views a packed slice as the C interface accepts it.
    pub fn of(keys: &[Key]) -> Self {
        usearch_keys_view_t { data: keys.as_ptr(), stride: core::mem::size_of::<usearch_key_t>() }
    }
}

impl Default for usearch_keys_view_t {
    fn default() -> Self {
        usearch_keys_view_t { data: ptr::null(), stride: 0 }
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
        usearch_outcome_t { status: usearch_status_success_k, message: ptr::null() }
    }
}

pub type usearch_scalar_t = c_int;

/// A run of rows along a call's batch axis, read from the caller's memory as an input and written
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
        usearch_rows_view_t { data: ptr::null(), stride: 0, columns: 0, scalar: 0 }
    }
}

/// A ranked run a call wrote into its arena - one list per query, `columns` packed columns each.
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
        usearch_matches_view_t { keys: ptr::null(), scores: ptr::null(), counts: ptr::null(), columns: 0 }
    }
}

impl usearch_matches_view_t {
    /// Query `query`'s matches, trimmed to the ones the call wrote; a call with nothing to report
    /// leaves null counts, which read as no matches.
    ///
    /// # Safety
    /// The view must be a finished call's output, and `query` below its count.
    pub(crate) unsafe fn query<'arena>(&self, query: usize) -> QueryMatches<'arena> {
        unsafe {
            let written = match self.counts.is_null() {
                true => 0,
                false => (*self.counts.add(query)).min(self.columns),
            };
            let start = query * self.columns;
            QueryMatches {
                keys: arena_slice(self.keys.wrapping_add(start), written),
                distances: arena_slice(self.scores.wrapping_add(start), written),
            }
        }
    }
}

/// A ragged run of matches a call wrote into its arena - keys and scores end to end, and one
/// more offset than queries.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_matches_tape_view_t {
    pub keys: *const usearch_key_t,
    pub scores: *const usearch_distance_t,
    pub offsets: *const usize,
}

impl Default for usearch_matches_tape_view_t {
    fn default() -> Self {
        usearch_matches_tape_view_t { keys: ptr::null(), scores: ptr::null(), offsets: ptr::null() }
    }
}

impl usearch_matches_tape_view_t {
    /// Query `query`, spanning its two offsets; a call with nothing to report leaves null offsets,
    /// which read as no matches.
    ///
    /// # Safety
    /// The view must be a finished call's output, and `query` below its count.
    pub(crate) unsafe fn query<'arena>(&self, query: usize) -> QueryMatches<'arena> {
        unsafe {
            let (start, end) = match self.offsets.is_null() {
                true => (0, 0),
                false => (*self.offsets.add(query), *self.offsets.add(query + 1)),
            };
            QueryMatches {
                keys: arena_slice(self.keys.wrapping_add(start), end - start),
                distances: arena_slice(self.scores.wrapping_add(start), end - start),
            }
        }
    }
}

/// A ragged run of keys a call wrote into its arena - every entry's keys end to end, and one
/// more offset than entries.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_keys_tape_view_t {
    pub keys: *const usearch_key_t,
    pub offsets: *const usize,
}

impl Default for usearch_keys_tape_view_t {
    fn default() -> Self {
        usearch_keys_tape_view_t { keys: ptr::null(), offsets: ptr::null() }
    }
}

impl usearch_keys_tape_view_t {
    /// Entry `entry`'s keys, spanning its two offsets; a call with nothing to report leaves null
    /// offsets, which read as no keys.
    ///
    /// # Safety
    /// The view must be a finished call's output, and `entry` below its count.
    pub(crate) unsafe fn entry<'arena>(&self, entry: usize) -> &'arena [Key] {
        unsafe {
            let (start, end) = match self.offsets.is_null() {
                true => (0, 0),
                false => (*self.offsets.add(entry), *self.offsets.add(entry + 1)),
            };
            arena_slice(self.keys.wrapping_add(start), end - start)
        }
    }
}

pub type usearch_tape_layout_t = c_int;

pub const usearch_tape_u32_k: usearch_tape_layout_t = 0;
pub const usearch_tape_u64_k: usearch_tape_layout_t = 1;

/// A ragged batch as Apache Arrow lays one out - elements end to end, and one more offset than
/// entries - the caller's as an input or written into an arena as an output; `offsets` is the C
/// union, read as wide as `layout` says.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_tape_view_t {
    pub data: *const c_void,
    pub layout: usearch_tape_layout_t,
    pub offsets: *const c_void,
    pub scalar: usearch_scalar_t,
}

impl Default for usearch_tape_view_t {
    fn default() -> Self {
        usearch_tape_view_t { data: ptr::null(), layout: usearch_tape_u64_k, offsets: ptr::null(), scalar: 0 }
    }
}

impl usearch_tape_view_t {
    /// Entry `entry`'s elements, spanning its two offsets; a call with nothing to report leaves
    /// null offsets, which read as no elements.
    ///
    /// # Safety
    /// The view must be a finished call's output of `Element`s, and `entry` below its count.
    pub(crate) unsafe fn entry<'arena, Element>(&self, entry: usize) -> &'arena [Element] {
        unsafe {
            let (start, end) = match (self.offsets.is_null(), self.layout) {
                (true, _) => (0, 0),
                (false, usearch_tape_u32_k) => {
                    let offsets = self.offsets.cast::<u32>();
                    (*offsets.add(entry) as usize, *offsets.add(entry + 1) as usize)
                }
                (false, _) => {
                    let offsets = self.offsets.cast::<u64>();
                    (*offsets.add(entry) as usize, *offsets.add(entry + 1) as usize)
                }
            };
            arena_slice(self.data.cast::<Element>().wrapping_add(start), end - start)
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_ram_options_t {
    pub memory_domains: *const i32,
    pub memory_domains_count: usize,
    pub shard_capacity: usize,
    pub cpu_class_ceiling: usearch_cpu_class_t,
    pub memory_limit: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_vram_options_t {
    pub devices: *const i32,
    pub devices_count: usize,
    pub shard_capacity: usize,
    pub gpu_class_ceiling: usearch_gpu_class_t,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_nvme_options_t {
    pub paths: *const usearch_bytes_view_t,
    pub paths_count: usize,
    pub cpu_class_ceiling: usearch_cpu_class_t,
}

/// The CPU class a placement may dispatch as, or its ceiling; zero derives the CPU's own.
pub type usearch_cpu_class_t = c_int;

/// The GPU class a placement may dispatch as, or its ceiling; zero derives the devices' own.
pub type usearch_gpu_class_t = c_int;

/// The NVMe class an executor submits through; zero derives the fastest the host admits.
pub type usearch_nvme_class_t = c_int;

/// Admits or rejects a batch of candidate keys, one admission per key.
pub type usearch_predicate_callback_t =
    Option<unsafe extern "C" fn(*const usearch_key_t, *mut c_int, usize, *mut c_void)>;

pub type usearch_predicate_cost_t = c_int;

pub const usearch_predicate_cost_cheap_k: usearch_predicate_cost_t = 0;

/// Which candidate keys a search may report: a handler and the state it is called with.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_predicate_t {
    pub callback: usearch_predicate_callback_t,
    pub state: *mut c_void,
    pub cost: usearch_predicate_cost_t,
}

/// Periodic advance report and the caller's chance to stop the call.
pub type usearch_progress_callback_t = Option<unsafe extern "C" fn(usize, usize, *mut c_void) -> c_int>;

/// A periodic advance report and the caller's chance to stop the call.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct usearch_progress_t {
    pub callback: usearch_progress_callback_t,
    pub state: *mut c_void,
    pub steps: usize,
}

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
#[derive(Clone, Copy, Default)]
pub struct usearch_arena_description_t {
    pub residency: usearch_residency_t,
    pub device: usize,
}

#[repr(C)]
pub struct usearch_arena_describe_t {
    pub arena: usearch_arena_t,
    pub outcome: *mut usearch_outcome_t,
    pub description: *mut usearch_arena_description_t,
}

#[repr(C)]
pub struct usearch_executor_for_ram_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub ram: usearch_ram_options_t,
    pub threads: usize,
    pub scratch_limit: usize,
}

#[repr(C)]
pub struct usearch_executor_for_vram_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub vram: usearch_vram_options_t,
    pub streams: *const *mut c_void,
    pub scratch_limit: usize,
}

#[repr(C)]
pub struct usearch_executor_for_nvme_t {
    pub executor: *mut usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub nvme: usearch_nvme_options_t,
    pub ram: usearch_ram_options_t,
    pub threads: usize,
    pub nvme_class: usearch_nvme_class_t,
    pub scratch_limit: usize,
}

pub type usearch_mood_t = c_int;
pub type usearch_load_path_t = c_int;

#[repr(C)]
pub struct usearch_executor_drain_t {
    pub executor: usearch_executor_t,
    pub outcome: *mut usearch_outcome_t,
    pub mood: usearch_mood_t,
}

unsafe extern "C" {
    pub fn usearch_outcome_wait(outcome: *mut usearch_outcome_t) -> usearch_status_t;
    pub fn usearch_outcome_status(outcome: *mut usearch_outcome_t) -> usearch_status_t;

    pub fn usearch_executor_for_ram(arguments: *mut usearch_executor_for_ram_t);
    pub fn usearch_executor_for_vram(arguments: *mut usearch_executor_for_vram_t);
    pub fn usearch_executor_for_nvme(arguments: *mut usearch_executor_for_nvme_t);
    pub fn usearch_executor_drain(arguments: *mut usearch_executor_drain_t);
    pub fn usearch_executor_free(executor: usearch_executor_t);

    pub fn usearch_arena_init(arguments: *mut usearch_arena_init_t);
    pub fn usearch_arena_free(arena: usearch_arena_t);
    pub fn usearch_arena_describe(arguments: *mut usearch_arena_describe_t);

    pub fn usearch_status_name(status: usearch_status_t) -> *const c_char;
    pub fn usearch_status_from_name(name: usearch_bytes_view_t, status: *mut usearch_status_t) -> usearch_status_t;
    pub fn usearch_scalar_name(scalar: usearch_scalar_t) -> *const c_char;
    pub fn usearch_scalar_from_name(name: usearch_bytes_view_t, scalar: *mut usearch_scalar_t) -> usearch_status_t;
    pub fn usearch_residency_name(residency: usearch_residency_t) -> *const c_char;
    pub fn usearch_residency_from_name(
        name: usearch_bytes_view_t,
        residency: *mut usearch_residency_t,
    ) -> usearch_status_t;
    pub fn usearch_traversal_name(traversal: usearch_traversal_t) -> *const c_char;
    pub fn usearch_traversal_from_name(
        name: usearch_bytes_view_t,
        traversal: *mut usearch_traversal_t,
    ) -> usearch_status_t;
    pub fn usearch_mood_name(mood: usearch_mood_t) -> *const c_char;
    pub fn usearch_load_path_name(load_path: usearch_load_path_t) -> *const c_char;
    pub fn usearch_mood_from_name(name: usearch_bytes_view_t, mood: *mut usearch_mood_t) -> usearch_status_t;
    pub fn usearch_phase_name(phase: c_int) -> *const c_char;
    pub fn usearch_phase_from_name(name: usearch_bytes_view_t, phase: *mut c_int) -> usearch_status_t;
    pub fn usearch_refine_verdict_name(verdict: c_int) -> *const c_char;
    pub fn usearch_refine_verdict_from_name(name: usearch_bytes_view_t, verdict: *mut c_int) -> usearch_status_t;

    pub fn usearch_version_name() -> *const c_char;
    pub fn usearch_capabilities_compiled() -> *const c_char;
    pub fn usearch_capabilities_available() -> *const c_char;
    pub fn usearch_cpu_class_name(cpu_class: usearch_cpu_class_t) -> *const c_char;
    // Only the class round-trip test reads these until the classes gain `FromStr`
    #[cfg(test)]
    pub fn usearch_cpu_class_from_name(
        name: usearch_bytes_view_t,
        cpu_class: *mut usearch_cpu_class_t,
    ) -> usearch_status_t;
    pub fn usearch_gpu_class_name(gpu_class: usearch_gpu_class_t) -> *const c_char;
    #[cfg(test)]
    pub fn usearch_gpu_class_from_name(
        name: usearch_bytes_view_t,
        gpu_class: *mut usearch_gpu_class_t,
    ) -> usearch_status_t;
    #[cfg(test)]
    pub fn usearch_nvme_class_name(nvme_class: usearch_nvme_class_t) -> *const c_char;
    #[cfg(test)]
    pub fn usearch_nvme_class_from_name(
        name: usearch_bytes_view_t,
        nvme_class: *mut usearch_nvme_class_t,
    ) -> usearch_status_t;

    pub fn usearch_keys_view_from_arrow(
        schema: *const ArrowSchema,
        array: *const ArrowArray,
        keys: *mut usearch_keys_view_t,
        count: *mut usize,
    ) -> usearch_status_t;
    pub fn usearch_rows_view_from_arrow(
        schema: *const ArrowSchema,
        array: *const ArrowArray,
        rows: *mut usearch_rows_view_t,
        count: *mut usize,
    ) -> usearch_status_t;
    pub fn usearch_keys_view_to_arrow(
        keys: usearch_keys_view_t,
        count: usize,
        arena: usearch_arena_t,
        schema: *mut ArrowSchema,
        array: *mut ArrowDeviceArray,
    ) -> usearch_status_t;
    pub fn usearch_rows_view_to_arrow(
        rows: usearch_rows_view_t,
        count: usize,
        arena: usearch_arena_t,
        schema: *mut ArrowSchema,
        array: *mut ArrowDeviceArray,
    ) -> usearch_status_t;
    pub fn usearch_matches_tape_view_to_arrow(
        matches: usearch_matches_tape_view_t,
        count: usize,
        arena: usearch_arena_t,
        schema: *mut ArrowSchema,
        array: *mut ArrowDeviceArray,
    ) -> usearch_status_t;
}

// endregion: Raw Declarations

pub(crate) use crate::sealed::{Accepts, Calling, Completion, RowsView, Submitting};

// region: Errors and Introspection

/// Every way a call can end besides success, numbered as the library numbers them.
///
/// An [`Error`] never holds [`Status::Deferred`], the status of a call not yet final.
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
    /// The call was refused rather than waited for, like a submission to an executor another call
    /// holds.
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
    /// Writes the status's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_status_name, *self as c_int))
    }
}

impl FromStr for Status {
    type Err = Error;

    /// Reads a status back from the name `Display` writes, refusing "success", which names no
    /// failure.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_status_from_name, |value| Status::from_raw(value).err())
    }
}

/// A failure, whether the engine reported it or the binding caught it before the call.
///
/// Messages are static storage, so nothing allocates on the failure path.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct Error {
    /// The identity of the failure, shared by every binding of the library.
    pub status: Status,
    /// The detail the status alone cannot hold.
    pub message: &'static str,
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.message)
    }
}

impl std::error::Error for Error {}

/// The result of every fallible call in this crate.
pub type Result<Value> = core::result::Result<Value, Error>;

/// Turns a final outcome into a `Result`.
pub(crate) fn check(outcome: usearch_outcome_t) -> Result<()> {
    match Status::from_raw(outcome.status) {
        Ok(()) => Ok(()),
        Err(Status::Deferred) => Err(fail(Status::Internal, "the outcome was read before it was final")),
        Err(status) => Err(Error { status, message: unsafe { static_string(outcome.message) } }),
    }
}

/// Reports a failure the binding caught before the call crossed the boundary, under the same
/// status the engine would have used had the ABI let it see the same thing.
pub(crate) fn fail(status: Status, message: &'static str) -> Error {
    Error { status, message }
}

/// Views a static null-terminated string the library owns.
///
/// Every such string is ASCII by construction, so its bytes are read as UTF-8 unchecked.
pub(crate) unsafe fn static_string(raw: *const c_char) -> &'static str {
    unsafe {
        if raw.is_null() {
            return "";
        }
        core::str::from_utf8_unchecked(CStr::from_ptr(raw).to_bytes())
    }
}

/// Spells an enumerator through its C `usearch_*_name`.
pub(crate) fn spelling(name: unsafe extern "C" fn(c_int) -> *const c_char, value: c_int) -> &'static str {
    unsafe { static_string(name(value)) }
}

/// Reads `text` through a parser that answers a status, then lifts the enumerator it wrote.
pub(crate) fn read_named<Value>(
    text: &str,
    from_name: unsafe extern "C" fn(usearch_bytes_view_t, *mut c_int) -> usearch_status_t,
    lift: fn(c_int) -> Option<Value>,
) -> Result<Value> {
    let mut value: c_int = 0;
    let status = unsafe { from_name(text.as_bytes().into(), &mut value) };
    (status == usearch_status_success_k)
        .then(|| lift(value))
        .flatten()
        .ok_or(fail(Status::InvalidArgument, "no enumerator is spelled that way"))
}

/// The version of the linked library.
pub fn version() -> &'static str {
    unsafe { static_string(usearch_version_name()) }
}

/// Comma-separated hardware capabilities the library was compiled with.
pub fn capabilities_compiled() -> &'static str {
    unsafe { static_string(usearch_capabilities_compiled()) }
}

/// Comma-separated hardware capabilities this machine offers at runtime.
pub fn capabilities_available() -> &'static str {
    unsafe { static_string(usearch_capabilities_available()) }
}

// endregion: Errors and Introspection

// region: Keys and Distances

/// Distance or similarity score in the metric of the index that produced it.
pub type Distance = usearch_distance_t;

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
}

impl From<u64> for Key {
    fn from(value: u64) -> Self {
        Key::from(u128::from(value))
    }
}

impl From<u128> for Key {
    fn from(value: u128) -> Self {
        Key(value.to_be_bytes())
    }
}

impl From<Key> for u128 {
    fn from(key: Key) -> Self {
        u128::from_be_bytes(key.0)
    }
}

impl TryFrom<Key> for u64 {
    type Error = core::num::TryFromIntError;

    /// The integer identity in the low eight bytes, refused when the high eight are not zero.
    fn try_from(key: Key) -> core::result::Result<Self, Self::Error> {
        u64::try_from(u128::from(key))
    }
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
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, formatter)
    }
}

// endregion: Keys and Distances

// region: Scalars

/// Scalar types the indexes store and exchange, named by width and encoding.
///
/// Sub-byte kinds pack several dimensions per byte, as their `xN` suffixes spell. The library
/// reserves the numbering gaps for kinds not yet implemented, so this enum is open.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Scalar {
    /// IEEE-754 binary64: ±1.80e308, epsilon 2.22e-16, subnormals to 4.94e-324, with infinities and NaNs.
    F64 = 1,
    /// IEEE-754 binary32: ±3.40e38, epsilon 1.19e-7, subnormals to 1.40e-45, with infinities and NaNs.
    F32 = 2,
    /// Truncated f32 - 8-bit exponent, 7-bit mantissa: f32's range at ±3.39e38 and epsilon 7.81e-3, with
    /// infinities and NaNs.
    BF16 = 3,
    /// IEEE-754 binary16: ±65504, epsilon 9.77e-4, subnormals to 5.96e-8, with infinities and NaNs.
    F16 = 4,
    /// FP8 with a 5-bit exponent: ±57344, epsilon 0.25, with infinities and NaNs - f16's dynamic
    /// range held on two mantissa bits.
    E5M2 = 5,
    /// FP8 with a 4-bit exponent: ±448, epsilon 0.125, no infinities, and two NaN codes at 0x7F and 0xFF.
    E4M3 = 6,
    /// FP6 with a 3-bit exponent, byte-aligned with two padding bits: ±28, no infinities or NaNs, so every one
    /// of its 64 codes is a number.
    E3M2 = 7,
    /// FP6 with a 2-bit exponent, byte-aligned with two padding bits: ±7.5, no infinities or NaNs, and 72% of
    /// its codes outside [-1, 1].
    E2M3 = 8,

    /// 8-bit signed integer spanning -128 to 127, every code a number.
    I8 = 10,
    /// Two signed nibbles per byte spanning -8 to 7, even dimensions in the low nibble and odd ones in the high.
    I4x2 = 11,
    /// 32-bit signed integer spanning ±2.15e9, every code a number.
    I32 = 12,

    /// 8-bit unsigned integer spanning 0 to 255, every code a number.
    U8 = 20,
    /// Two unsigned nibbles per byte spanning 0 to 15, even dimensions in the low nibble and odd ones in the high.
    U4x2 = 21,
    /// Eight single-bit dimensions per byte - the form the Hamming & Jaccard metrics score, where only the
    /// set-bit count of a XOR or AND holds meaning.
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
    /// Writes the scalar's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_scalar_name, *self as c_int))
    }
}

impl FromStr for Scalar {
    type Err = Error;

    /// Reads a scalar back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_scalar_from_name, Scalar::from_raw)
    }
}

/// Ties a NumKong storage type to the [`Scalar`] the library tags its buffers with.
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
    /// The tag the library knows this type by.
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
    Ok(([values / values_per_row, dimensions], [values_per_row as isize * value_bytes, value_bytes]))
}

/// Views a packed slab as a row-major matrix of `dimensions`-wide rows.
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
        Ok(MatrixLayout { rows, columns, stride_bytes: pitch as usize })
    }
}

/// A view of rows the caller owns and the batch axis it spans.
#[derive(Clone, Copy)]
pub struct Batch {
    pub(crate) rows: usearch_rows_view_t,
    pub(crate) count: usize,
}

// Safety: it views plain scalars that the verb holding it owns or references beside it.
unsafe impl Send for Batch {}

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
    fn batch(&self) -> Result<Batch> {
        batch_of(self)
    }
}

impl<Value: Element, Alloc: Allocator> RowsView for Matrix<Value, Alloc> {
    type Value = Value;
    fn batch(&self) -> Result<Batch> {
        batch_of(self)
    }
}

impl<Rows: RowsView + ?Sized> RowsView for &Rows {
    type Value = Rows::Value;
    fn batch(&self) -> Result<Batch> {
        (**self).batch()
    }
}

// endregion: Matrix Views

// region: Tapes

/// The width of a tape's offsets, which the library reads as the tape's `layout`.
///
/// Rust-only.
///
/// # Safety
/// The layout must name exactly this type's width.
pub unsafe trait TapeOffset: Copy + 'static {
    /// The layout the library reads these offsets under.
    const LAYOUT: usearch_tape_layout_t;
}

unsafe impl TapeOffset for u32 {
    const LAYOUT: usearch_tape_layout_t = usearch_tape_u32_k;
}

unsafe impl TapeOffset for u64 {
    const LAYOUT: usearch_tape_layout_t = usearch_tape_u64_k;
}

/// A ragged batch laid out as Apache Arrow lays one out: elements end to end, and `count + 1`
/// non-decreasing offsets around them, so entry `i` spans `offsets[i]` up to `offsets[i + 1]`.
///
/// An element is a byte of a sequence or a point of a shape. Both runs are referenced or owned, and
/// a deferred call takes only an owned, `'static` tape.
///
/// Rust-only.
#[derive(Clone, Debug)]
pub struct Tape<'data, Element: Clone, Offset: Clone = u64> {
    /// Every entry's elements, end to end.
    pub elements: Cow<'data, [Element]>,
    /// One more offset than entries, counted in elements.
    pub offsets: Cow<'data, [Offset]>,
}

impl<'data, Element: Clone, Offset: TapeOffset> Tape<'data, Element, Offset> {
    /// Views an Arrow-shaped pair of runs, as a `Binary` or `List` column holds them.
    pub fn new(elements: &'data [Element], offsets: &'data [Offset]) -> Self {
        Tape { elements: Cow::Borrowed(elements), offsets: Cow::Borrowed(offsets) }
    }

    /// Entries the tape spans, one fewer than its offsets.
    pub fn count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Passes the tape as the view the C interface reads, its elements tagged `scalar`.
    pub(crate) fn view(&self, scalar: Scalar) -> usearch_tape_view_t {
        usearch_tape_view_t {
            data: self.elements.as_ptr().cast(),
            layout: Offset::LAYOUT,
            offsets: self.offsets.as_ptr().cast(),
            scalar: scalar as usearch_scalar_t,
        }
    }
}

impl<Element: Clone> Tape<'static, Element, u64> {
    /// Packs separate entries into one owned tape, copying them end to end.
    pub fn of<Entry: AsRef<[Element]>>(entries: impl IntoIterator<Item = Entry>) -> Self {
        let (mut elements, mut offsets) = (Vec::new(), vec![0u64]);
        for entry in entries {
            elements.extend_from_slice(entry.as_ref());
            offsets.push(elements.len() as u64);
        }
        Tape { elements: Cow::Owned(elements), offsets: Cow::Owned(offsets) }
    }
}

impl<Element: Clone + 'static, Offset: Clone + 'static> Accepts<Tape<'static, Element, Offset>> for Deferring {}

/// The tapes a `get` wrote into the arena it points into, one entry per value stored under its key,
/// in insertion order.
///
/// Rust-only.
pub struct TapeEntries<'arena, Element> {
    pub(crate) arena: PhantomData<&'arena Arena<'arena>>,
    pub(crate) view: usearch_tape_view_t,
    pub(crate) written: usize,
    pub(crate) element: PhantomData<Element>,
}

// Safety: the view points into the arena referenced beside it, which is `Sync`.
unsafe impl<Element> Send for TapeEntries<'_, Element> {}

impl<'arena, Element: 'arena> TapeEntries<'arena, Element> {
    /// An empty slot a call fills in an arena.
    pub(crate) fn new() -> Self {
        TapeEntries { arena: PhantomData, view: usearch_tape_view_t::default(), written: 0, element: PhantomData }
    }

    /// Entries the call wrote; zero for an absent key.
    pub fn written(&self) -> usize {
        self.written
    }

    /// One entry's elements, refused with [`Status::InvalidArgument`] when `entry` is not below
    /// [`TapeEntries::written`].
    pub fn entry(&self, entry: usize) -> Result<&'arena [Element]> {
        match entry < self.written {
            // Safety: the arena holds a finished call's output.
            true => Ok(unsafe { self.view.entry(entry) }),
            false => Err(fail(Status::InvalidArgument, "the entry is not below the call's count")),
        }
    }

    /// Every entry, in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = &'arena [Element]> + '_ {
        // Safety: the arena holds a finished call's output.
        (0..self.written).map(move |entry| unsafe { self.view.entry(entry) })
    }
}

/// The source and target handles a reshard passes, laid out as the library reads them.
pub(crate) struct ReshardHandles {
    pub(crate) sources: Vec<*mut c_void>,
    pub(crate) targets: Vec<*mut c_void>,
}

// Safety: the handles belong to referenced indexes, which are `Sync`.
unsafe impl Send for ReshardHandles {}

// endregion: Tapes

// region: Matches

/// One query's matches, or one left row's pairs, trimmed to the cells the call wrote.
///
/// Rust-only.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct QueryMatches<'matches> {
    /// The keys written for this query: nearest first for a search, largest score first for a
    /// `maxsim`, and in traversal order for a `multiply`.
    pub keys: &'matches [Key],
    /// Positionally paired with the keys: distances for a search, summed similarities for a
    /// `maxsim`, and similarities for a `multiply`.
    pub distances: &'matches [Distance],
}

/// The ranked matches a search or a stream pull wrote into the arena it points into; see
/// [`DenseVerbs::search`](crate::DenseVerbs::search).
pub struct Matches<'arena> {
    pub(crate) arena: &'arena Arena<'arena>,
    pub(crate) view: usearch_matches_view_t,
    pub(crate) queries: usize,
}

// Safety: the view points into the arena referenced beside it, which is `Sync`.
unsafe impl Send for Matches<'_> {}

impl<'arena> Matches<'arena> {
    /// An empty slot a call fills in `arena`.
    pub(crate) fn new(arena: &'arena Arena<'arena>) -> Self {
        Matches { arena, view: usearch_matches_view_t::default(), queries: 0 }
    }

    /// Queries the call resolved.
    pub fn queries(&self) -> usize {
        self.queries
    }

    /// Matches written for one query, refused with [`Status::InvalidArgument`] when `query` is not
    /// below [`Matches::queries`].
    pub fn query(&self, query: usize) -> Result<QueryMatches<'arena>> {
        match query < self.queries {
            // Safety: the arena holds a finished call's output.
            true => Ok(unsafe { self.view.query(query) }),
            false => Err(fail(Status::InvalidArgument, "the query is not below the call's count")),
        }
    }

    /// Every query's matches, in query order.
    pub fn iter(&self) -> impl Iterator<Item = QueryMatches<'arena>> {
        let view = self.view;
        // Safety: the arena holds a finished call's output.
        (0..self.queries).map(move |query| unsafe { view.query(query) })
    }

    /// Exports every query's padded keys, `queries` by `columns`, as one Arrow
    /// `FixedSizeBinary(16)` column tagged `arrow.uuid` in place; each query's count bounds its
    /// own row.
    ///
    /// The export points into the arena, so no verb reaches it while the export lives:
    ///
    /// ```compile_fail
    /// use usearch::*;
    /// # fn main() -> Result<()> {
    /// let options = DenseOptions { dimensions: 2, ..DenseOptions::default() };
    /// let index = DenseIndex::init_ram(&options, &[])?;
    /// let slab = [1.0f32, 0.0];
    /// let mut arena = Arena::new()?;
    /// let matches = index.search(&rows(&slab, 2)?, 1, &mut arena)?;
    /// let export = matches.keys_to_arrow()?;
    /// index.search(&rows(&slab, 2)?, 1, &mut arena)?;
    /// drop(export);
    /// # Ok(())
    /// # }
    /// ```
    pub fn keys_to_arrow(&self) -> Result<ArrowExport<'arena>> {
        let keys = usearch_keys_view_t { data: self.view.keys, stride: core::mem::size_of::<usearch_key_t>() };
        let count = self.queries * self.view.columns;
        let mut schema = ArrowSchema::default();
        let mut array = ArrowDeviceArray::default();
        let status = unsafe { usearch_keys_view_to_arrow(keys, count, self.arena.handle, &mut schema, &mut array) };
        arrow_checked(status)?;
        Ok(ArrowExport { arena: PhantomData, schema, array })
    }

    /// Exports every query's padded distances as one Arrow `FixedSizeList` of `columns` `f64` per
    /// query in place, pointing into the arena as [`Matches::keys_to_arrow`] does.
    pub fn distances_to_arrow(&self) -> Result<ArrowExport<'arena>> {
        let columns = self.view.columns;
        let distances = usearch_rows_view_t {
            data: self.view.scores.cast(),
            stride: columns * core::mem::size_of::<Distance>(),
            columns,
            scalar: Scalar::F64 as usearch_scalar_t,
        };
        let mut schema = ArrowSchema::default();
        let mut array = ArrowDeviceArray::default();
        let status =
            unsafe { usearch_rows_view_to_arrow(distances, self.queries, self.arena.handle, &mut schema, &mut array) };
        arrow_checked(status)?;
        Ok(ArrowExport { arena: PhantomData, schema, array })
    }
}

// endregion: Matches

// region: Arrow

/// A column's type in Apache Arrow's C data interface, laid out as `arrow::ffi::FFI_ArrowSchema`
/// and released when dropped.
#[repr(C)]
pub struct ArrowSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchema,
    dictionary: *mut ArrowSchema,
    release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    private_data: *mut c_void,
}

/// A column's buffers in Apache Arrow's C data interface, laid out as `arrow::ffi::FFI_ArrowArray`
/// and released when dropped.
#[repr(C)]
pub struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *mut *const c_void,
    children: *mut *mut ArrowArray,
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
}

/// An [`ArrowArray`] and the device holding its buffers, as Arrow's C device data interface lays
/// them out.
#[repr(C)]
pub struct ArrowDeviceArray {
    array: ArrowArray,
    device_id: i64,
    device_type: i32,
    sync_event: *mut c_void,
    reserved: [i64; 3],
}

impl Default for ArrowSchema {
    /// A released schema, which every converter refuses.
    fn default() -> Self {
        ArrowSchema {
            format: ptr::null(),
            name: ptr::null(),
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl Default for ArrowArray {
    /// A released array, which every converter refuses.
    fn default() -> Self {
        ArrowArray {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl Default for ArrowDeviceArray {
    /// A released array on no device.
    fn default() -> Self {
        ArrowDeviceArray {
            array: ArrowArray::default(),
            device_id: 0,
            device_type: 0,
            sync_event: ptr::null_mut(),
            reserved: [0; 3],
        }
    }
}

impl ArrowSchema {
    /// The column's format string, or none once released.
    pub fn format(&self) -> Option<&CStr> {
        // Safety: a live schema's format is a null-terminated string it owns.
        self.release.map(|_| unsafe { CStr::from_ptr(self.format) })
    }
}

impl ArrowArray {
    /// Rows the column holds.
    pub fn length(&self) -> i64 {
        self.length
    }
}

impl ArrowDeviceArray {
    /// The column itself.
    pub fn array(&self) -> &ArrowArray {
        &self.array
    }
}

impl Drop for ArrowSchema {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

impl Drop for ArrowArray {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

// Safety: Arrow's C data interface lets a producer's release run on any thread.
unsafe impl Send for ArrowSchema {}
unsafe impl Send for ArrowArray {}
unsafe impl Send for ArrowDeviceArray {}

/// One Arrow column a producer handed over through the C data interface, which `arrow::ffi` pairs
/// as `FFI_ArrowSchema` and `FFI_ArrowArray`.
///
/// # Safety
/// Both pointers must address live structures laid out as [`ArrowSchema`] and [`ArrowArray`] and
/// following Arrow's C data interface, whose host buffers stay valid and unwritten for as long as
/// `self` stays referenced.
pub unsafe trait ArrowColumn {
    /// The column's type.
    fn schema(&self) -> *const ArrowSchema;
    /// The column's buffers.
    fn array(&self) -> *const ArrowArray;
}

/// Turns a converter's or an exporter's status into a `Result`.
pub(crate) fn arrow_checked(status: usearch_status_t) -> Result<()> {
    Status::from_raw(status).map_err(|status| fail(status, "Arrow refused the column"))
}

/// The rows of one Arrow fixed-size list column, read in place through its buffers, which a verb
/// takes wherever it takes a matrix.
///
/// Rust-only.
pub struct ArrowRows<'column, Value> {
    batch: Batch,
    column: PhantomData<&'column Value>,
}

impl<'column, Value: Element> ArrowRows<'column, Value> {
    /// Views `column`'s rows, refused with [`Status::UnsupportedLayout`] for another Arrow type,
    /// [`Status::InvalidArgument`] for a null, and [`Status::UnknownKind`] for rows of another
    /// scalar than `Value`.
    pub fn new<Column: ArrowColumn + ?Sized>(column: &'column Column) -> Result<Self> {
        let mut rows = usearch_rows_view_t::default();
        let mut count = 0;
        arrow_checked(unsafe { usearch_rows_view_from_arrow(column.schema(), column.array(), &mut rows, &mut count) })?;
        if rows.scalar != Value::SCALAR as usearch_scalar_t {
            return Err(fail(Status::UnknownKind, "the Arrow rows hold another scalar"));
        }
        Ok(ArrowRows { batch: Batch { rows, count }, column: PhantomData })
    }
}

impl<Value: Element> RowsView for ArrowRows<'_, Value> {
    type Value = Value;
    fn batch(&self) -> Result<Batch> {
        Ok(self.batch)
    }
}

/// Views the keys of one Arrow `fixed_size_binary(16)` column in place, tagged `arrow.uuid` or not,
/// refused as [`ArrowRows::new`] refuses.
pub fn keys_from_arrow<Column: ArrowColumn + ?Sized>(column: &Column) -> Result<&[Key]> {
    let mut keys = usearch_keys_view_t::default();
    let mut count = 0;
    arrow_checked(unsafe { usearch_keys_view_from_arrow(column.schema(), column.array(), &mut keys, &mut count) })?;
    // Safety: the column's promise keeps the keys alive through its reference; a key is its bytes.
    Ok(unsafe { arena_slice(keys.data, count) })
}

/// A column exported as Arrow in place, reading the arena it sits in until that arena's next call.
///
/// Rust-only.
pub struct ArrowExport<'arena> {
    pub(crate) arena: PhantomData<&'arena Arena<'arena>>,
    pub(crate) schema: ArrowSchema,
    pub(crate) array: ArrowDeviceArray,
}

impl ArrowExport<'_> {
    /// The column's type.
    pub fn schema(&self) -> &ArrowSchema {
        &self.schema
    }

    /// The column's buffers and the device holding them.
    pub fn array(&self) -> &ArrowDeviceArray {
        &self.array
    }

    /// Hands the column over past the arena's lifetime, for a consumer that takes the structures by
    /// value.
    ///
    /// # Safety
    ///
    /// The arena must outlive the parts and take no call until they are released, as nothing
    /// refuses one, and its buffers must be host memory.
    pub unsafe fn into_parts(self) -> (ArrowSchema, ArrowArray) {
        (self.schema, self.array.array)
    }
}

// endregion: Arrow

// region: Classes

/// The CPU families USearch ships tuned variants for: dispatch reports its pick in this
/// vocabulary, and a RAM or NVMe placement or executor may cap dispatch at one.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum CPUClass {
    /// The portable floor: no SIMD assumptions.
    Serial = 1,
    /// The one class of a monoversioned build, with every capability its floor implies or its
    /// preset vouches for.
    Monoversioned = 2,
    /// Haswell-class x86: AVX2 & FMA, the 256-bit floor.
    IntelHaswell = 3,
    /// Sierra Forest-class x86 - the E-core Xeons: AVX-VNNI and its successors without AVX-512.
    IntelSierra = 4,
    /// Skylake-class x86: the AVX-512 foundation tier.
    IntelSkylake = 5,
    /// Icelake-class x86: AVX-512 VNNI & VPOPCNT.
    IntelIcelake = 6,
    /// Genoa-class x86 - Zen 4 and newer: native bf16 dot products over AVX-512.
    AMDGenoa = 7,
    /// Sapphire Rapids-class x86: the Genoa bundle plus AVX-512 FP16 and the AMX-BF16 & AMX-INT8
    /// tiles.
    IntelSapphire = 8,
    /// Granite Rapids-class x86: the Sapphire Rapids bundle plus the AMX-FP16 tiles.
    IntelGranite = 9,
    /// Diamond Rapids-class x86: the Granite Rapids bundle plus AVX10.2's FP8 conversions.
    IntelDiamond = 10,
    /// Graviton2-class Arm - Neoverse N1, the bare NEON floor.
    AWSG2 = 12,
    /// Armv8.6+ NEON: dot products, bf16 & f16 fused multiplies - Apple M2 class silicon.
    AppleM2 = 13,
    /// Apple M4-class Arm: the Armv8.6 NEON bundle plus the SME outer-product tiles.
    AppleM4 = 14,
    /// Graviton5-class Arm - Neoverse V3, Armv9.2.
    AWSG5 = 15,
    /// Vera-class Arm - NVIDIA's Armv9.2-generation cores with FP8 dot products.
    NVIDIAVera = 16,
    /// Alibaba C930-class RISC-V - T-Head's XuanTie core, RVA23 vectors.
    AlibabaC930 = 17,
    /// IBM POWER10-class Power, ISA 3.1: VSX vectors, MMA accumulators and `pause_short` waits.
    IBMPower10 = 18,
    /// WebAssembly every engine loads: 32-bit, single-threaded, the fixed-width simd128 tier.
    Wasm32V128 = 19,
    /// WebAssembly with relaxed SIMD, shared memory for real thread pools, and 64-bit
    /// addressing.
    Wasm64Extended = 20,
}

impl fmt::Display for CPUClass {
    /// Writes the class's name, as an index description reports it.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_cpu_class_name, *self as c_int))
    }
}

/// The GPU families USearch ships tuned variants for, the device twin of [`CPUClass`]: a VRAM
/// placement or executor may cap dispatch at one.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum GPUClass {
    /// Ampere-class NVIDIA and every GPU without a class of its own - A100, A10 & RTX 30.
    NVIDIAAmpere = 1,
    /// Ada-class NVIDIA - L4, L40S & RTX 40: the fp8 conversions.
    NVIDIAAda = 2,
    /// Hopper-class NVIDIA - H100 & H200.
    NVIDIAHopper = 3,
    /// Blackwell-class NVIDIA - B200, B300 & GB300.
    NVIDIABlackwell = 4,
    /// Consumer Blackwell NVIDIA - RTX 50, RTX PRO 6000 & DGX Spark.
    NVIDIABlackwellRTX = 5,
    /// Rubin-class NVIDIA - the Vera-Rubin platform's device half.
    NVIDIARubin = 6,
    /// CDNA3 AMD - MI300X & MI325X: gfx942, FNUZ fp8.
    AMDCDNA3 = 7,
    /// CDNA4 AMD - MI350X & MI355X: gfx950, OCP fp8.
    AMDCDNA4 = 8,
    /// CDNA5 AMD - MI400: gfx1250.
    AMDCDNA5 = 9,
    /// Apple7-family GPUs and every Apple GPU without a class of its own - M1 & M2.
    AppleMetal7 = 10,
    /// Apple9-family GPUs - M3 & M4.
    AppleMetal9 = 11,
    /// Apple10-family GPUs - M5.
    AppleMetal10 = 12,
}

impl fmt::Display for GPUClass {
    /// Writes the class's name, as an index description reports it.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_gpu_class_name, *self as c_int))
    }
}

// endregion: Classes

// region: Placements

/// The memory domains a RAM placement or executor names, and the ceiling on its class.
pub(crate) fn ram_options(memory_domains: &[i32], cpu_class_ceiling: Option<CPUClass>) -> usearch_ram_options_t {
    usearch_ram_options_t {
        memory_domains: memory_domains.as_ptr(),
        memory_domains_count: memory_domains.len(),
        shard_capacity: 0,
        cpu_class_ceiling: cpu_class_ceiling.map_or(0, |cpu_class_ceiling| cpu_class_ceiling as usearch_cpu_class_t),
        memory_limit: 0,
    }
}

/// The devices a VRAM placement or executor names, and the ceiling on its class.
pub(crate) fn vram_options(devices: &[i32], gpu_class_ceiling: Option<GPUClass>) -> usearch_vram_options_t {
    usearch_vram_options_t {
        devices: devices.as_ptr(),
        devices_count: devices.len(),
        shard_capacity: 0,
        gpu_class_ceiling: gpu_class_ceiling.map_or(0, |gpu_class_ceiling| gpu_class_ceiling as usearch_gpu_class_t),
    }
}

/// The paths an NVMe placement or executor names, pointing into `views`, which must outlive the
/// call, and the ceiling on the class it dispatches as.
pub(crate) fn nvme_options(
    views: &[usearch_bytes_view_t],
    cpu_class_ceiling: Option<CPUClass>,
) -> usearch_nvme_options_t {
    usearch_nvme_options_t {
        paths: views.as_ptr(),
        paths_count: views.len(),
        cpu_class_ceiling: cpu_class_ceiling.map_or(0, |cpu_class_ceiling| cpu_class_ceiling as usearch_cpu_class_t),
    }
}

/// Views a path as the bytes the C interface exchanges, which must be UTF-8.
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

/// Whether a predicate admits one candidate key or passes over it.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum Admission {
    /// Skip the key; the walk still routes through it.
    #[default]
    Reject = 0,
    /// Report the key among the matches.
    Admit = 1,
}

/// How costly one predicate call is, which decides how many keys each call carries.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
pub enum PredicateCost {
    /// Asked as candidates arrive, a few keys a call, with nothing buffered.
    #[default]
    Cheap = 0,
    /// Asked once per traversal round or scan run about every candidate it gathered, for handlers
    /// that pay per call, such as a lookup in an external database.
    Expensive = 1,
}

/// A periodic report of finished and total units, passed as the C struct every reporting verb
/// takes; `Break` stops the call at the next increment wherever the verb can stop, which an
/// insertion never can.
///
/// The handler runs on whichever thread advanced the work, never on two at once, so it owes `Send`
/// and may be `FnMut`. A panic inside it aborts the process, unable to unwind through the library.
pub struct Progress<'handler> {
    raw: usearch_progress_t,
    handler: PhantomData<&'handler mut ()>,
}

// Safety: it points at a `Send` handler it references exclusively.
unsafe impl Send for Progress<'_> {}

impl<'handler> Progress<'handler> {
    /// Passes `handler`, called with the units finished so far and the total the call finishes.
    pub fn new<Handler: FnMut(usize, usize) -> ControlFlow<()> + Send>(handler: &'handler mut Handler) -> Self {
        let raw = usearch_progress_t {
            callback: Some(progress_trampoline::<Handler>),
            state: ptr::from_mut(handler).cast(),
            steps: 0,
        };
        Progress { raw, handler: PhantomData }
    }
}

/// # Safety
/// `state` must address the `Handler` a [`Progress`] passed to the call.
unsafe extern "C" fn progress_trampoline<Handler: FnMut(usize, usize) -> ControlFlow<()>>(
    finished: usize,
    total: usize,
    state: *mut c_void,
) -> c_int {
    unsafe {
        let handler = &mut *state.cast::<Handler>();
        c_int::from(handler(finished, total).is_break())
    }
}

/// Passes an optional progress and step count as reporting verbs take them; `None` passes nulls.
pub(crate) fn lower_progress(progress: &Option<Progress>, steps: usize) -> usearch_progress_t {
    match progress {
        Some(progress) => usearch_progress_t { steps, ..progress.raw },
        None => usearch_progress_t { callback: None, state: ptr::null_mut(), steps },
    }
}

/// Which candidate keys a search may report, passed as the C struct every filtering verb takes.
///
/// The handler runs on the executor's threads together, so it owes `Sync`; a panic inside it
/// aborts the process.
#[derive(Clone, Copy)]
pub struct Predicate<'handler> {
    raw: usearch_predicate_t,
    handler: PhantomData<&'handler ()>,
}

// Safety: it points at a `Sync` handler it references shared.
unsafe impl Send for Predicate<'_> {}
unsafe impl Sync for Predicate<'_> {}

impl<'handler> Predicate<'handler> {
    /// Answers one key at a time, asked as candidates are scored.
    pub fn each<Handler: Fn(Key) -> bool + Sync>(handler: &'handler Handler) -> Self {
        Self::lowered(each_trampoline::<Handler>, handler, PredicateCost::Cheap)
    }

    /// Answers a batch at a time, every admission starting as [`Admission::Reject`], asked once per
    /// round unless told it is cheap.
    pub fn batch<Handler: Fn(&[Key], &mut [Admission]) + Sync>(handler: &'handler Handler) -> Self {
        Self::lowered(batch_trampoline::<Handler>, handler, PredicateCost::Expensive)
    }

    /// Asks the handler as candidates arrive, a few keys a call.
    pub fn cheap(mut self) -> Self {
        self.raw.cost = PredicateCost::Cheap as usearch_predicate_cost_t;
        self
    }

    fn lowered<Handler>(
        callback: unsafe extern "C" fn(*const Key, *mut c_int, usize, *mut c_void),
        handler: &'handler Handler,
        cost: PredicateCost,
    ) -> Self {
        let raw = usearch_predicate_t {
            callback: Some(callback),
            state: ptr::from_ref(handler).cast_mut().cast(),
            cost: cost as usearch_predicate_cost_t,
        };
        Predicate { raw, handler: PhantomData }
    }
}

/// # Safety
/// `state` must address the `Handler` a [`Predicate`] passed to the call, and `keys` and
/// `admissions` must each address `count` elements.
unsafe extern "C" fn each_trampoline<Handler: Fn(Key) -> bool>(
    keys: *const Key,
    admissions: *mut c_int,
    count: usize,
    state: *mut c_void,
) {
    unsafe {
        let handler = &*state.cast::<Handler>();
        let keys = core::slice::from_raw_parts(keys, count);
        let admissions = core::slice::from_raw_parts_mut(admissions, count);
        for (key, admission) in keys.iter().zip(admissions) {
            *admission = c_int::from(handler(*key));
        }
    }
}

/// # Safety
/// `state` must address the `Handler` a [`Predicate`] passed to the call, and `keys` and
/// `admissions` must each address `count` elements.
unsafe extern "C" fn batch_trampoline<Handler: Fn(&[Key], &mut [Admission])>(
    keys: *const Key,
    admissions: *mut c_int,
    count: usize,
    state: *mut c_void,
) {
    unsafe {
        let handler = &*state.cast::<Handler>();
        // Rejection is zero, so every slot holds a valid admission before the handler reads it.
        ptr::write_bytes(admissions, 0, count);
        let admissions = core::slice::from_raw_parts_mut(admissions.cast::<Admission>(), count);
        handler(core::slice::from_raw_parts(keys, count), admissions);
    }
}

/// Passes an optional predicate as the struct every filtering verb takes; `None` passes nulls.
pub(crate) fn lower_predicate(predicate: &Option<Predicate>) -> usearch_predicate_t {
    match predicate {
        Some(predicate) => predicate.raw,
        None => usearch_predicate_t { callback: None, state: ptr::null_mut(), cost: usearch_predicate_cost_cheap_k },
    }
}

/// Passes an optional output as the pointer the library writes it through; `None` passes null.
pub(crate) fn output_slot<Value>(output: &mut Option<&mut Value>) -> *mut Value {
    output.as_deref_mut().map_or(ptr::null_mut(), ptr::from_mut)
}

// endregion: Callbacks

// region: Phases

/// The kind of work a refine or reshard plan is doing, one value shared by every residency.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum Phase {
    /// No plan, or one that has not started.
    #[default]
    None = 0,
    /// A reshard moving keys and values into a target, with the links it keeps.
    Copy = 1,
    /// A reshard on NVMe building blocks of a target's rows in RAM and writing their links.
    Build = 2,
    /// A reshard cutting a source into one part per target.
    Partition = 3,
    /// A reshard moving each part into its target.
    Scatter = 4,
    /// A round's health check, opening or closing it.
    Health = 5,
    /// A round linking each island with the root's component.
    Heal = 6,
    /// A round drawing the hubs its gather streams entries past.
    Hubs = 7,
    /// A round gathering every entry into a neighborhood.
    Gather = 8,
    /// A round re-stitching each neighborhood.
    Restitch = 9,
    /// A round on NVMe rewriting each tile of links from what its restitch changed.
    Write = 10,
    /// The plan ended its round, and nothing more is pending.
    Done = 11,
}

impl Phase {
    fn from_raw(value: c_int) -> Option<Self> {
        let phase = match value {
            0 => Phase::None,
            1 => Phase::Copy,
            2 => Phase::Build,
            3 => Phase::Partition,
            4 => Phase::Scatter,
            5 => Phase::Health,
            6 => Phase::Heal,
            7 => Phase::Hubs,
            8 => Phase::Gather,
            9 => Phase::Restitch,
            10 => Phase::Write,
            11 => Phase::Done,
            _ => return None,
        };
        Some(phase)
    }
}

impl fmt::Display for Phase {
    /// Writes the phase's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_phase_name, *self as c_int))
    }
}

impl FromStr for Phase {
    type Err = Error;

    /// Reads a phase back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_phase_from_name, Phase::from_raw)
    }
}

/// Where a plan stands: its phase, and that phase's own count.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub struct PhaseProgress {
    /// The kind of work in hand.
    pub phase: Phase,
    /// This phase's position among the ones the plan runs.
    pub ordinal: usize,
    /// Units of this phase done, entries or islands.
    pub completed: usize,
    /// Units of this phase in all.
    pub total: usize,
}

// endregion: Phases

// region: Executors

/// What an executor does with its threads or streams once it goes idle.
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
    /// Writes the mood's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_mood_name, *self as c_int))
    }
}

impl FromStr for Mood {
    type Err = Error;

    /// Reads a mood back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_mood_from_name, Mood::from_raw)
    }
}

/// Where an executor computes, and with it where the arenas it makes keep their memory.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum Residency {
    /// Volatile memory, traversed by CPU threads.
    Ram = 1,
    /// Accelerator memory, traversed by GPU warps on a chosen device.
    Vram = 2,
    /// Persisted storage, traversed through pluggable I/O engines.
    Nvme = 3,
}

impl Residency {
    fn from_raw(value: usearch_residency_t) -> Option<Self> {
        match value {
            1 => Some(Residency::Ram),
            2 => Some(Residency::Vram),
            3 => Some(Residency::Nvme),
            _ => None,
        }
    }
}

impl fmt::Display for Residency {
    /// Writes the residency's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_residency_name, *self as c_int))
    }
}

impl FromStr for Residency {
    type Err = Error;

    /// Reads a residency back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_residency_from_name, Residency::from_raw)
    }
}

/// How a search reaches the indexed entries it scores.
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[non_exhaustive]
pub enum Traversal {
    /// Walks the graph, scoring the entries the expansion reaches - approximate.
    #[default]
    Walk = 1,
    /// Scores every entry published when the query's scan starts - exact, on RAM and NVMe handles.
    Scan = 2,
}

impl Traversal {
    fn from_raw(value: usearch_traversal_t) -> Option<Self> {
        match value {
            1 => Some(Traversal::Walk),
            2 => Some(Traversal::Scan),
            _ => None,
        }
    }
}

impl fmt::Display for Traversal {
    /// Writes the traversal's name, which `from_str` reads back.
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(spelling(usearch_traversal_name, *self as c_int))
    }
}

impl FromStr for Traversal {
    type Err = Error;

    /// Reads a traversal back from the name `Display` writes, refusing any other.
    fn from_str(text: &str) -> Result<Self> {
        read_named(text, usearch_traversal_from_name, Traversal::from_raw)
    }
}

/// The host threads or device streams an [`On`] runs its calls on, and the traversal scratch each
/// of them walks with.
///
/// One executor serves any number of index handles, so separate categories of indexes share one
/// machine's cores or devices instead of each claiming a private set. Its kind is fixed by the
/// constructor that produced it and must match the residency of every handle it drives.
///
/// [`Executor::default`] is the null executor: it owns nothing, and every call
/// handed it runs on the calling thread.
///
/// One call holds an executor at a time, and a second call finding it held fails with
/// [`Status::Contended`], as nothing queues, so threads wanting to overlap take one apiece; it is
/// `Send` but not `Sync`. A deferred call on a caller-exclusive pool runs on the pool's thread 0,
/// so polling alone finishes it, while a single thread or a VRAM executor runs it at submission.
/// [`Arena::new_in`] places an arena where it computes, and the arena refers to it, so the executor
/// outlives every arena it made. Dropping an executor drains it, then frees it.
#[derive(Debug)]
pub struct Executor {
    pub(crate) handle: usearch_executor_t,
}

// Safety: nothing in an executor is tied to the thread that created it, so it may move to another;
// staying out of `Sync` keeps submissions to it on one thread at a time.
unsafe impl Send for Executor {}

impl Default for Executor {
    fn default() -> Self {
        Executor::adopt(ptr::null_mut())
    }
}

impl Executor {
    /// Owns the handle a constructor filled, which is null for the executor that owns nothing.
    fn adopt(handle: usearch_executor_t) -> Self {
        Executor { handle }
    }

    /// Spawns host workers for RAM-resident indexes, interleaved across the compute domains local
    /// to `memory_domains`, which is the same list the indexes were initialized with.
    ///
    /// Zero `threads` takes 7/8 of the cores those domains hold, since full occupancy starves
    /// the ordered publication path.
    ///
    /// A `cpu_class_ceiling` caps the class the threads' waiters and atomics dispatch as,
    /// narrowing to what the machine runs.
    pub fn for_ram(memory_domains: &[i32], cpu_class_ceiling: Option<CPUClass>, threads: usize) -> Result<Self> {
        let mut handle = ptr::null_mut();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_executor_for_ram(&mut usearch_executor_for_ram_t {
                executor: &mut handle,
                outcome: &mut outcome,
                ram: ram_options(memory_domains, cpu_class_ceiling),
                threads,
                scratch_limit: 0,
            })
        };
        let executor = Executor::adopt(handle);
        check(outcome).map(|()| executor)
    }

    /// Spawns host workers for NVMe-resident indexes, one submission ring apiece, across the
    /// compute domains local to the devices `paths` names.
    ///
    /// Zero `threads` derives from those devices' queue capabilities.
    ///
    /// A `cpu_class_ceiling` caps the class the threads' waiters and atomics dispatch as,
    /// narrowing to what the machine runs.
    pub fn for_nvme(paths: &[impl AsRef<Path>], cpu_class_ceiling: Option<CPUClass>, threads: usize) -> Result<Self> {
        let views = path_views(paths)?;
        let mut handle = ptr::null_mut();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_executor_for_nvme(&mut usearch_executor_for_nvme_t {
                executor: &mut handle,
                outcome: &mut outcome,
                nvme: nvme_options(&views, cpu_class_ceiling),
                ram: ram_options(&[], None),
                threads,
                nvme_class: 0,
                scratch_limit: 0,
            })
        };
        let executor = Executor::adopt(handle);
        check(outcome).map(|()| executor)
    }

    /// Opens one non-blocking stream on every named device, created and destroyed with the
    /// executor, all of them driven by the one host thread that calls the verbs - which is why this
    /// takes no thread count.
    ///
    /// Zero `scratch_limit` lets scratch grow as geometry asks. A `gpu_class_ceiling` no named
    /// device loads is refused with [`Status::Unsupported`].
    pub fn for_vram(devices: &[i32], gpu_class_ceiling: Option<GPUClass>, scratch_limit: usize) -> Result<Self> {
        unsafe { Self::for_vram_on_streams(devices, gpu_class_ceiling, &[], scratch_limit) }
    }

    /// Names the caller's CUDA streams, one per device in `devices` order, in place of the ones
    /// [`Executor::for_vram`] opens; the library answers [`Status::Unsupported`] until caller
    /// streams are wired.
    ///
    /// # Safety
    /// Every stream must be a live CUDA stream on its device, which the caller destroys only after
    /// dropping the executor.
    pub unsafe fn for_vram_on_streams(
        devices: &[i32],
        gpu_class_ceiling: Option<GPUClass>,
        streams: &[*mut c_void],
        scratch_limit: usize,
    ) -> Result<Self> {
        unsafe {
            if !streams.is_empty() && streams.len() != devices.len() {
                return Err(fail(Status::ShapeMismatch, "pass no streams or one per device"));
            }
            let mut handle = ptr::null_mut();
            let mut outcome = usearch_outcome_t::default();
            usearch_executor_for_vram(&mut usearch_executor_for_vram_t {
                executor: &mut handle,
                outcome: &mut outcome,
                vram: vram_options(devices, gpu_class_ceiling),
                // The C side reads one stream per device from any non-null pointer.
                streams: match streams.is_empty() {
                    true => ptr::null(),
                    false => streams.as_ptr(),
                },
                scratch_limit,
            });
            let executor = Executor::adopt(handle);
            check(outcome).map(|()| executor)
        }
    }

    /// Waits until no call holds this executor, then adopts `mood` until the next call; a failure
    /// is the executor's own, such as a lost device, never one of a drained call.
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
    fn drop(&mut self) {
        unsafe { usearch_executor_free(self.handle) }
    }
}

// endregion: Executors

// region: Arenas

/// Memory the library writes every output whose length scales with a call into, reused across calls.
///
/// An executor places it where it computes, host pages or its GPU's memory, decided once, and
/// nothing is allocated until a call writes into it. Every call rewinds and refills it,
/// invalidating the previous call's outputs, so a steady loop allocates once.
/// Every verb, blocking or deferred, takes it as `&mut Arena` and answers outputs that point into
/// it in turn.
/// A [`Residency::Vram`] arena holds device memory, which the host accessors of its outputs must
/// not read; [`Arena::describe`] says where they live. It grows, fills and frees on the first
/// stream of the executor that made it, so it serves only calls on that executor, any other refused
/// with [`Status::Device`].
///
/// `'executor` refers to the executor that made it, which therefore outlives it, and dropping it
/// waits for the calls still writing into it:
///
/// ```compile_fail,E0505
/// use usearch::*;
/// # fn main() -> Result<()> {
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let arena = Arena::new_in(&executor)?;
/// drop(executor);
/// drop(arena);
/// # Ok(())
/// # }
/// ```
#[derive(Debug)]
pub struct Arena<'executor> {
    pub(crate) handle: usearch_arena_t,
    executor: PhantomData<&'executor ()>,
}

// Safety: an arena serves one call at a time, which the `&mut` every verb takes already enforces,
// and a shared reference only asks the library where it lives.
unsafe impl Send for Arena<'_> {}
unsafe impl Sync for Arena<'_> {}

impl Arena<'static> {
    /// Makes an arena of host pages for calls on the calling thread.
    pub fn new() -> Result<Self> {
        Arena::made_by(ptr::null_mut())
    }
}

impl<'executor> Arena<'executor> {
    /// Makes an arena where `executor` computes, serving the calls of its residency, or only those
    /// on `executor` itself for a GPU arena.
    pub fn new_in(executor: &'executor Executor) -> Result<Self> {
        Arena::made_by(executor.handle)
    }

    fn made_by(executor: usearch_executor_t) -> Result<Self> {
        let mut arena = Arena { handle: ptr::null_mut(), executor: PhantomData };
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_arena_init(&mut usearch_arena_init_t { arena: &mut arena.handle, executor, outcome: &mut outcome })
        };
        check(outcome).map(|()| arena)
    }

    /// Where the outputs live, as the executor that made the arena decided.
    pub fn describe(&self) -> Result<ArenaDescription> {
        let mut description = usearch_arena_description_t::default();
        let mut outcome = usearch_outcome_t::default();
        unsafe {
            usearch_arena_describe(&mut usearch_arena_describe_t {
                arena: self.handle,
                outcome: &mut outcome,
                description: &mut description,
            })
        };
        check(outcome)?;
        let residency = Residency::from_raw(description.residency)
            .ok_or_else(|| fail(Status::UnknownKind, "the arena names no known residency"))?;
        Ok(ArenaDescription {
            residency,
            device: match residency {
                Residency::Vram => u32::try_from(description.device).ok(),
                _ => None,
            },
        })
    }
}

impl Drop for Arena<'_> {
    fn drop(&mut self) {
        unsafe { usearch_arena_free(self.handle) }
    }
}

/// Where an [`Arena`]'s outputs live.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub struct ArenaDescription {
    /// [`Residency::Ram`] for host pages, [`Residency::Vram`] for one device's memory.
    pub residency: Residency,
    /// The device ordinal, as [`Executor::for_vram`] takes it; `None` for host pages.
    pub device: Option<u32>,
}

/// Views `length` values the library wrote into an arena; a failed or empty call leaves null.
///
/// # Safety
/// `data` must address `length` initialized values that live as long as the arena reference.
pub(crate) unsafe fn arena_slice<'arena, Value>(data: *const Value, length: usize) -> &'arena [Value] {
    unsafe {
        match length {
            0 => &[],
            length => core::slice::from_raw_parts(data, length),
        }
    }
}

// endregion: Arenas

// region: Receivers

/// An index referenced with the executor its calls run on and how they complete, shared by every
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
    fn clone(&self) -> Self {
        *self
    }
}

impl<Index, Completion> Copy for On<'_, Index, Completion> {}

/// The index a verb reaches and the executor it runs on, as its receiver passed them.
#[derive(Clone, Copy)]
pub(crate) struct Handles {
    pub(crate) handle: *mut c_void,
    pub(crate) executor: usearch_executor_t,
}

// Safety: an index is shared across threads, and an executor goes with the calls submitted to it.
unsafe impl Send for Handles {}

impl Handles {
    pub(crate) fn of(handle: *mut c_void, executor: usearch_executor_t) -> Self {
        Handles { handle, executor }
    }
}

/// The completion of a blocking receiver, whose verbs return once their call is final and answer
/// what it produced.
///
/// Rust-only.
#[derive(Debug)]
pub struct Blocking;

impl Completion for Blocking {
    type Deferrable<'index, Verb: Calling> = Verb::Output;

    fn answer<'index, Verb: Calling>(mut verb: Verb) -> Result<Self::Deferrable<'index, Verb>> {
        let mut outcome = usearch_outcome_t::default();
        let mut arguments = verb.arguments(&mut outcome, usearch_completion_blocking_k);
        // Safety: the arguments point into `verb` and `outcome`, which stay in place until the call
        // returns, and a blocking call is final on return.
        unsafe { Verb::call(&mut arguments) };
        verb.finish(check(outcome))
    }
}

impl<Input> Accepts<Input> for Blocking {}

/// The completion of a deferring receiver, whose verbs answer a [`Deferred`] that submits the call
/// once first polled or waited on.
///
/// Rust-only.
#[derive(Debug)]
pub struct Deferring;

impl Completion for Deferring {
    type Deferrable<'index, Verb: Calling> = Deferred<'index, Verb>;

    fn answer<'index, Verb: Calling>(verb: Verb) -> Result<Self::Deferrable<'index, Verb>> {
        Ok(Deferred::new(verb))
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Unsubmitted,
    Submitted,
    Answered,
}

/// What the library reads and writes while a call is in flight, reached only through raw pointers.
struct Call<Verb: Calling> {
    outcome: usearch_outcome_t,
    arguments: MaybeUninit<Verb::Arguments>,
    /// The owned inputs, the output slots and the arena reference.
    verb: ManuallyDrop<Verb>,
}

/// A call submitted on its first poll or on [`Deferred::wait`], at the address it is pinned at, so
/// its outcome, arguments, inputs and output slots live inline and need no allocation.
///
/// It is the [`Future`] of its call, so `.await` submits it and polls its status word until final,
/// while `wait` blocks the thread without spinning, and both hand back what the call produced.
/// Dropping a submitted one waits for its call, and dropping an unsubmitted one runs nothing.
/// `'index` refers to the [`On`] that made it, and it references its arena as a blocking call does.
///
/// The library reads its inputs until it is final, so a deferred call keeps no local reference:
///
/// ```compile_fail,E0597
/// use usearch::*;
/// # fn main() -> Result<()> {
/// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 2, ..DenseOptions::default() }, &[])?;
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let slab = vec![0.0f32, 1.0];
/// let call = index.deferred(&executor).search(rows(&slab, 2)?, 1, &mut arena)?;
/// call.wait()?;
/// # Ok(())
/// # }
/// ```
///
/// And an arena serves one call at a time, so two calls in flight take one apiece:
///
/// ```compile_fail,E0499
/// use usearch::*;
/// # fn main() -> Result<()> {
/// static SLAB: [f32; 2] = [0.0, 1.0];
/// let index = DenseIndex::init_ram(&DenseOptions { dimensions: 2, ..DenseOptions::default() }, &[])?;
/// let executor = Executor::for_ram(&[], None, 0)?;
/// let mut arena = Arena::new_in(&executor)?;
/// let first = index.deferred(&executor).search(rows(&SLAB, 2)?, 1, &mut arena)?;
/// let second = index.deferred(&executor).search(rows(&SLAB, 2)?, 1, &mut arena)?;
/// first.wait()?;
/// second.wait()?;
/// # Ok(())
/// # }
/// ```
///
/// Rust-only.
#[must_use = "a deferred call runs only once awaited or waited on"]
pub struct Deferred<'index, Verb: Calling> {
    /// The library writes here while `stage` is `Submitted`.
    call: UnsafeCell<Call<Verb>>,
    stage: Stage,
    receiver: PhantomData<&'index ()>,
    _pinned: PhantomPinned,
}

// Safety: until submission it is plain data, and once submitted it is pinned and only the library
// reaches the call.
unsafe impl<Verb: Calling + Send> Send for Deferred<'_, Verb> {}

impl<Verb: Calling> Deferred<'_, Verb> {
    fn new(verb: Verb) -> Self {
        Deferred {
            call: UnsafeCell::new(Call {
                outcome: usearch_outcome_t::default(),
                arguments: MaybeUninit::uninit(),
                verb: ManuallyDrop::new(verb),
            }),
            stage: Stage::Unsubmitted,
            receiver: PhantomData,
            _pinned: PhantomPinned,
        }
    }

    fn outcome(&self) -> *mut usearch_outcome_t {
        unsafe { ptr::addr_of_mut!((*self.call.get()).outcome) }
    }

    /// Hands the call to the library at the address it is pinned at.
    fn submit(self: Pin<&mut Self>) {
        // Safety: nothing moves out, and the pointers taken here stay valid while it is pinned.
        let this = unsafe { self.get_unchecked_mut() };
        let call = this.call.get();
        unsafe {
            let outcome = ptr::addr_of_mut!((*call).outcome);
            let arguments = (*call).verb.arguments(outcome, usearch_completion_deferred_k);
            let arguments = (*call).arguments.write(arguments) as *mut Verb::Arguments;
            this.stage = Stage::Submitted;
            Verb::call(arguments);
        }
    }

    /// Takes the verb back from a final call, which the library touches no more, beside its status.
    fn answered(self: Pin<&mut Self>) -> (Result<()>, Verb) {
        // Safety: the call is final, so nothing reads the verb once it moves out.
        let this = unsafe { self.get_unchecked_mut() };
        this.stage = Stage::Answered;
        let call = this.call.get();
        unsafe { (check(ptr::read(ptr::addr_of!((*call).outcome))), ManuallyDrop::take(&mut (*call).verb)) }
    }

    /// Submits the call, then blocks until it is final and hands back what it produced.
    pub fn wait(self) -> Result<Verb::Output> {
        let mut this = pin!(self);
        this.as_mut().submit();
        unsafe { usearch_outcome_wait(this.outcome()) };
        let (status, verb) = this.answered();
        verb.finish(status)
    }
}

impl<Verb: Submitting> Deferred<'_, Verb> {
    /// Submits the call, then blocks until it is final and hands back what it produced beside the
    /// inputs it owned, like a plan a stop left live.
    pub fn wait_with_inputs(self) -> (Result<Verb::Output>, Verb::Inputs) {
        let mut this = pin!(self);
        this.as_mut().submit();
        unsafe { usearch_outcome_wait(this.outcome()) };
        let (status, verb) = this.answered();
        verb.finish_with_inputs(status)
    }
}

impl<Verb: Calling> Future for Deferred<'_, Verb> {
    type Output = Result<Verb::Output>;

    /// Submits the call on the first poll, then reads its status word; a call not yet final asks to
    /// be polled again, since the library has no wake-up.
    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        match self.stage {
            Stage::Answered => return Poll::Ready(Err(fail(Status::InvalidArgument, "the call was already awaited"))),
            Stage::Unsubmitted => self.as_mut().submit(),
            Stage::Submitted => {}
        }
        if unsafe { usearch_outcome_status(self.outcome()) } == usearch_status_deferred_k {
            context.waker().wake_by_ref();
            return Poll::Pending;
        }
        let (status, verb) = self.answered();
        Poll::Ready(verb.finish(status))
    }
}

impl<Verb: Calling> Drop for Deferred<'_, Verb> {
    fn drop(&mut self) {
        match self.stage {
            Stage::Answered => return,
            Stage::Submitted => unsafe {
                usearch_outcome_wait(self.outcome());
            },
            Stage::Unsubmitted => {}
        }
        // Safety: the call is final or was never submitted, so the library reads the verb no more.
        unsafe { ManuallyDrop::drop(&mut (*self.call.get()).verb) };
    }
}

// endregion: Receivers

// region: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DenseVerbs;
    use crate::testing::skipped;

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
    fn class_names_read_back() {
        type FromName = unsafe extern "C" fn(usearch_bytes_view_t, *mut c_int) -> usearch_status_t;
        type Name = unsafe extern "C" fn(c_int) -> *const c_char;
        let classes: [(FromName, Name); 3] = [
            (usearch_cpu_class_from_name, usearch_cpu_class_name),
            (usearch_gpu_class_from_name, usearch_gpu_class_name),
            (usearch_nvme_class_from_name, usearch_nvme_class_name),
        ];
        // Classes keep their registry values forever; a class this build lacks spells "unknown".
        for (from_name, name) in classes {
            for code in (1..=c_int::from(u8::MAX)).filter(|&code| spelling(name, code) != "unknown") {
                let spelled = spelling(name, code);
                assert_eq!(read_named(spelled, from_name, Some), Ok(code), "{spelled}");
            }
            assert!(read_named("unknown", from_name, Some).is_err());
        }
        assert_eq!(CPUClass::Serial.to_string(), "serial");
        assert_eq!(CPUClass::Monoversioned.to_string(), "monoversioned");
        assert_eq!(CPUClass::IBMPower10.to_string(), "ibm-power10");
        assert_eq!(GPUClass::AppleMetal10.to_string(), "apple-metal10");
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
        for code in 0..=11 {
            let phase = Phase::from_raw(code).unwrap();
            assert_eq!(phase.to_string().parse(), Ok(phase), "{phase:?}");
        }

        assert!("success".parse::<Status>().is_err(), "success names no failure");
        assert!("shape mismatch".parse::<Status>().is_err());
        assert!("unknown".parse::<Scalar>().is_err());
        assert!("parking".parse::<Mood>().is_err());
        assert!("gpu".parse::<Residency>().is_err());
        assert!("exact".parse::<Traversal>().is_err());
        assert!("stage".parse::<Phase>().is_err());
    }

    #[test]
    fn scalar_tags_match_their_storage_types() {
        assert_eq!(<f32 as Element>::SCALAR, Scalar::F32);
        assert_eq!(<bf16 as Element>::SCALAR, Scalar::BF16);
        assert_eq!(<i8 as Element>::SCALAR, Scalar::I8);
        assert_eq!(<u1x8 as Element>::SCALAR, Scalar::U1x8);
    }

    #[test]
    fn packed_slabs_view_as_matrices() {
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
    }

    #[test]
    fn packed_slabs_reject_a_zero_width() {
        // The width divides the slab, so a zero would be a division rather than an empty run.
        let failure = rows(&[1.0f32, 2.0], 0).unwrap_err();
        assert_eq!(failure.status, Status::ShapeMismatch);
    }

    /// An arena lives where the executor that made it computes.
    #[test]
    fn arenas_live_where_their_executor_computes() {
        let host = ArenaDescription { residency: Residency::Ram, device: None };
        assert_eq!(Arena::new().unwrap().describe().unwrap(), host);

        let executor = Executor::for_ram(&[], None, 1).unwrap();
        assert_eq!(Arena::new_in(&executor).unwrap().describe().unwrap(), host);

        let Ok(gpu) = Executor::for_vram(&[0], None, 0) else {
            return skipped("no GPU device");
        };
        assert_eq!(
            Arena::new_in(&gpu).unwrap().describe().unwrap(),
            ArenaDescription { residency: Residency::Vram, device: Some(0) }
        );
    }

    #[test]
    fn errors_hold_a_status() {
        let failure = fail(Status::ShapeMismatch, "something specific went wrong");
        assert_eq!(failure.status, Status::ShapeMismatch);
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
            let raw = usearch_outcome_t { status: status as i32, message: ptr::null() };
            assert_eq!(check(raw).unwrap_err().status, status);
        }
        // A final outcome is never deferred, and a status this build has no name for is not
        // success either, so both read as the contract violation they are.
        for status in [Status::Deferred as i32, 4242] {
            let raw = usearch_outcome_t { status, message: ptr::null() };
            assert_eq!(check(raw).unwrap_err().status, Status::Internal);
        }
    }

    #[test]
    fn placements_marshal_as_list_views() {
        let derived = ram_options(&[], None);
        assert_eq!(derived.memory_domains_count, 0, "an empty list derives");
        assert_eq!(derived.cpu_class_ceiling, 0, "no ceiling dispatches as the machine");

        let devices = [0, 1, 2, 3];
        let capped = vram_options(&devices, Some(GPUClass::NVIDIAHopper));
        assert_eq!(capped.devices_count, 4);
        assert_eq!(unsafe { *capped.devices.add(3) }, 3);
        assert_eq!(capped.gpu_class_ceiling, 3, "a ceiling crosses as its C value");

        let views = path_views(&["/mnt/nvme0", "/mnt/nvme1"]).unwrap();
        let drives = nvme_options(&views, Some(CPUClass::IntelSkylake));
        assert_eq!(drives.paths_count, 2);
        assert_eq!(drives.cpu_class_ceiling, 5);
        assert_eq!(unsafe { (*drives.paths).length }, "/mnt/nvme0".len());
    }

    unsafe extern "C" fn leave_schema(schema: *mut ArrowSchema) {
        unsafe {
            (*schema).release = None;
        }
    }
    unsafe extern "C" fn leave_array(array: *mut ArrowArray) {
        unsafe {
            (*array).release = None;
        }
    }

    /// An Arrow column laid out by hand over `T`'s bytes, as a foreign producer hands one over:
    /// `width` of `format`'s values per row, or one value per row without an element.
    struct HandMadeColumn<'bytes> {
        schema: ArrowSchema,
        array: ArrowArray,
        _element: Option<(Box<ArrowSchema>, Box<[*mut ArrowSchema; 1]>)>,
        _values: Option<(Box<ArrowArray>, Box<[*mut ArrowArray; 1]>)>,
        _buffers: Box<[*const c_void; 2]>,
        _bytes: PhantomData<&'bytes [u8]>,
    }

    // Safety: the fields hold every node, and the reference holds the bytes.
    unsafe impl ArrowColumn for HandMadeColumn<'_> {
        fn schema(&self) -> *const ArrowSchema {
            &self.schema
        }
        fn array(&self) -> *const ArrowArray {
            &self.array
        }
    }

    fn hand_made_schema(format: &'static CStr) -> ArrowSchema {
        ArrowSchema { format: format.as_ptr(), release: Some(leave_schema), ..ArrowSchema::default() }
    }

    fn hand_made_array(length: i64, buffers: *mut *const c_void) -> ArrowArray {
        ArrowArray { length, n_buffers: 2, buffers, release: Some(leave_array), ..ArrowArray::default() }
    }

    /// A `w:16` column of `keys`.
    fn hand_made_keys(keys: &[Key]) -> HandMadeColumn<'_> {
        let mut buffers = Box::new([ptr::null(), keys.as_ptr().cast::<c_void>()]);
        let array = hand_made_array(keys.len() as i64, buffers.as_mut_ptr());
        HandMadeColumn {
            schema: hand_made_schema(c"w:16"),
            array,
            _element: None,
            _values: None,
            _buffers: buffers,
            _bytes: PhantomData,
        }
    }

    /// A `+w:4` column of `f32` rows over `slab`, its first `skipped` rows sliced away.
    fn hand_made_rows(slab: &[f32], skipped: i64) -> HandMadeColumn<'_> {
        let mut element = Box::new(hand_made_schema(c"f"));
        let mut element_pointers = Box::new([&mut *element as *mut ArrowSchema]);
        let mut buffers = Box::new([ptr::null(), slab.as_ptr().cast::<c_void>()]);
        let mut values = Box::new(hand_made_array(slab.len() as i64, buffers.as_mut_ptr()));
        let mut value_pointers = Box::new([&mut *values as *mut ArrowArray]);
        let mut schema = hand_made_schema(c"+w:4");
        schema.n_children = 1;
        schema.children = element_pointers.as_mut_ptr();
        let array = ArrowArray {
            length: slab.len() as i64 / 4 - skipped,
            offset: skipped,
            n_buffers: 1,
            n_children: 1,
            buffers: buffers.as_mut_ptr(),
            children: value_pointers.as_mut_ptr(),
            release: Some(leave_array),
            ..ArrowArray::default()
        };
        HandMadeColumn {
            schema,
            array,
            _element: Some((element, element_pointers)),
            _values: Some((values, value_pointers)),
            _buffers: buffers,
            _bytes: PhantomData,
        }
    }

    /// A foreign producer's column inserts in place and answers at its own keys, past its offset.
    #[test]
    fn arrow_columns_insert_in_place() {
        let options = crate::DenseOptions { dimensions: 4, ..crate::DenseOptions::default() };
        let index = crate::DenseIndex::init_ram(&options, &[]).unwrap();
        let slab = [0.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let keys = [Key::from(7u64), Key::from(9u64)];
        let vectors = hand_made_rows(&slab, 1);
        let identities = hand_made_keys(&keys);
        assert_eq!(keys_from_arrow(&identities).unwrap(), &keys);
        index.insert(keys_from_arrow(&identities).unwrap(), ArrowRows::<f32>::new(&vectors).unwrap()).unwrap();

        let mut arena = Arena::new().unwrap();
        let matches = index.search(&rows(&slab[8..], 4).unwrap(), 1, &mut arena).unwrap();
        assert_eq!(matches.query(0).unwrap().keys, &[Key::from(9u64)]);
        assert_eq!(ArrowRows::<f64>::new(&vectors).err().map(|error| error.status), Some(Status::UnknownKind));
    }

    /// Padded matches export their keys and distances as two columns, one row per query.
    #[test]
    fn matches_export_keys_and_distances_apart() {
        let options = crate::DenseOptions { dimensions: 4, ..crate::DenseOptions::default() };
        let index = crate::DenseIndex::init_ram(&options, &[]).unwrap();
        let slab = [1.0f32, 0.0, 0.0, 0.0];
        index.insert([Key::from(3u64)], &rows(&slab, 4).unwrap()).unwrap();

        let mut arena = Arena::new().unwrap();
        let matches = index.search(&rows(&slab, 4).unwrap(), 1, &mut arena).unwrap();
        let keys = matches.keys_to_arrow().unwrap();
        assert_eq!(keys.schema().format(), Some(c"w:16"));
        let distances = matches.distances_to_arrow().unwrap();
        assert_eq!(distances.schema().format(), Some(c"+w:1"));
        assert_eq!(distances.array().array.length(), 1);
    }
}

// endregion: Tests
