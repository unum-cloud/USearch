//! Helpers the crate's tests share: the run's seed and streams, scratch directories, announced
//! skips, and the executors, keys and tolerances every index module's tests reach for.
//!
//! File: rust/testing.rs
//! Author: Ash Vardanian
//! Date: November 16, 2026

#![allow(dead_code, reason = "the Shared Testing region is the same in every sibling library, used or not")]

use crate::{Error, Executor, Key, Result, Status};

/// The environment variable the run's seed is read from, as in the C and C++ suites.
const SEED_VARIABLE: &str = "USEARCH_SEED";

// region Shared Testing
/// Announces a skipped test on stderr, so a run shows what it left out rather than passing in
/// silence.
pub(crate) fn skipped(reason: &str) {
    eprintln!("skipped: {reason}");
}

/// The run's seed, read once from the variable [`SEED_VARIABLE`] names: 42 when unset, drawn afresh
/// for `random` and then printed, so the run can be repeated.
pub(crate) fn seed() -> u64 {
    static SEED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SEED.get_or_init(read_seed)
}

/// Parses [`SEED_VARIABLE`] once for [`seed`].
fn read_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};

    const EXPECTED: &str = "an unsigned integer or random";
    let text = std::env::var(SEED_VARIABLE).unwrap_or_default();
    if text.is_empty() {
        return 42;
    }
    if text != "random" {
        let Ok(parsed) = text.parse::<u64>() else {
            panic!("{SEED_VARIABLE}=\"{text}\" does not parse, expected {EXPECTED}");
        };
        return parsed;
    }
    let hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let drawn = hasher.finish();
    eprintln!("{SEED_VARIABLE}={drawn}");
    drawn
}

/// SplitMix64's finalizer, a bijection that spreads every input bit over all 64 output bits.
pub(crate) const fn split_mix_bits(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

/// The `index`-th stream of the family `name` under `seed`, identical in C, C++, Python and
/// Mojo.
pub(crate) fn stream_key(seed: u64, name: &str, index: u64) -> u64 {
    let mut hashed = 0xCBF2_9CE4_8422_2325_u64;
    for byte in name.bytes() {
        hashed = (hashed ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3);
    }
    split_mix_bits(split_mix_bits(seed ^ hashed).wrapping_add(index))
}

/// The `index`-th stream of the running test, keyed by the name libtest gives its thread.
pub(crate) fn case_stream_key(index: u64) -> u64 {
    let thread = std::thread::current();
    let name = thread.name().unwrap_or("main");
    stream_key(seed(), name, index)
}

/// A directory of its own under the system's temporary one, removed when dropped.
pub(crate) struct ScratchDirectory {
    path: std::path::PathBuf,
}

impl ScratchDirectory {
    /// Creates `<temp>/<name>-<pid>-<nanos>`, so parallel tests never share one.
    pub(crate) fn new(name: &str) -> Self {
        let since = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH);
        let nanos = since.map_or(0, |elapsed| elapsed.as_nanos());
        let leaf = format!("{name}-{}-{nanos}", std::process::id());
        let path = std::env::temp_dir().join(leaf);
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    /// Where the directory is.
    pub(crate) fn path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for ScratchDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

/// Polls `future` to completion on this thread with a no-op waker.
pub(crate) fn block_on<F: core::future::Future>(future: F) -> F::Output {
    let mut future = core::pin::pin!(future);
    let waker = core::task::Waker::noop();
    let mut context = core::task::Context::from_waker(waker);
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}
// endregion Shared Testing

/// One executor per test, since an executor runs one call at a time.
pub(crate) fn executor() -> Executor {
    Executor::for_ram(&[], None, 0).expect("the executor must spawn")
}

/// `count` consecutive keys from `first`.
pub(crate) fn keys(first: u64, count: usize) -> Vec<Key> {
    (first..first + count as u64).map(Key::from).collect()
}

/// Whether `answer` is the library's refusal with `status`.
pub(crate) fn refused(answer: Result<impl Sized>, status: Status) -> bool {
    matches!(answer, Err(Error { status: answered, .. }) if answered == status)
}

/// Panics unless `|actual - wanted| <= tolerance * max(1, |actual|, |wanted|)`, the rule every
/// suite's `expect_near` keeps.
#[track_caller]
pub(crate) fn assert_close(actual: f64, wanted: f64, tolerance: f64) {
    let bound = tolerance * actual.abs().max(wanted.abs()).max(1.0);
    assert!((actual - wanted).abs() <= bound, "expected {wanted} within {tolerance}, got {actual}");
}

/// Panics unless the rows have one length and every value is close to its counterpart.
#[track_caller]
pub(crate) fn assert_close_rows(actual: &[f32], wanted: &[f32], tolerance: f64) {
    assert_eq!(actual.len(), wanted.len(), "expected {} values, got {}", wanted.len(), actual.len());
    for (actual, wanted) in actual.iter().zip(wanted) {
        assert_close(f64::from(*actual), f64::from(*wanted), tolerance);
    }
}
