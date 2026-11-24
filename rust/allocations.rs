//! Counts what a warm call allocates, which is nothing: the binding's heap through a counting
//! global allocator, and every heap allocation in the process, the C library's included, through
//! the platform's own hook, macOS's `malloc_logger` or an in-binary `malloc` that glibc lets the
//! executable define, no interposition library on either.
//!
//! Executors, arenas, blocking calls, deferred calls and their callbacks keep the binding's state
//! inline, and a warm call naming an executor allocates nothing on either side. A call without an
//! executor runs on the calling thread, and the library builds its one arena slot and its scratch
//! for that call alone, so it allocates a stated, bounded number of times, and the binding none.
//!
//! It installs a counting global allocator, so it runs as a program of its own, outside libtest.
//!
//! File: rust/allocations.rs
//! Author: Ash Vardanian
//! Date: November 18, 2026

#[cfg(all(target_os = "linux", target_env = "gnu"))]
use core::ffi::c_void;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use usearch::*;

struct Counting;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

/// Heap allocations anywhere in the process, the C library's included, once `count_heap` installed
/// the platform's hook.
static HEAP_ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

/// The record `ram_executor_t::on_calling_thread` reserves for the call's one arena.
const CALLING_THREAD_ARENA_SLOTS: usize = 1;

/// The chunks that arena grows through for this test's search, from empty, since nothing earlier
/// left it a high-water mark.
const CALLING_THREAD_ARENA_CHUNKS: usize = 5;

#[cfg(target_os = "macos")]
type MallocLogger = unsafe extern "C" fn(u32, usize, usize, usize, usize, u32);

#[cfg(target_os = "macos")]
unsafe extern "C" {
    /// libmalloc calls it on every allocation and free in every default zone when set.
    static mut malloc_logger: Option<MallocLogger>;
}

/// `MALLOC_LOG_TYPE_ALLOCATE` from libmalloc's `stack_logging.h`; a `realloc` carries it too.
#[cfg(target_os = "macos")]
const MALLOC_LOG_ALLOCATE: u32 = 2;

#[cfg(target_os = "macos")]
unsafe extern "C" fn count_logged(kind: u32, _: usize, _: usize, _: usize, _: usize, _: u32) {
    if kind & MALLOC_LOG_ALLOCATE != 0 {
        HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    }
}

/// Installs the counting logger, which the process keeps until it exits, and answers whether the
/// process heap is counted.
#[cfg(target_os = "macos")]
fn count_heap() -> bool {
    unsafe { malloc_logger = Some(count_logged) };
    true
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
unsafe extern "C" {
    fn __libc_malloc(size: usize) -> *mut c_void;
    fn __libc_calloc(count: usize, size: usize) -> *mut c_void;
    fn __libc_realloc(pointer: *mut c_void, size: usize) -> *mut c_void;
    fn __libc_memalign(alignment: usize, size: usize) -> *mut c_void;
}

// glibc resolves `malloc` to the executable's definition for every shared object, so these see
// the C library's and libstdc++'s allocations too.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    unsafe { __libc_malloc(size) }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn calloc(count: usize, size: usize) -> *mut c_void {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    unsafe { __libc_calloc(count, size) }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn realloc(pointer: *mut c_void, size: usize) -> *mut c_void {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    unsafe { __libc_realloc(pointer, size) }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn aligned_alloc(alignment: usize, size: usize) -> *mut c_void {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    unsafe { __libc_memalign(alignment, size) }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn memalign(alignment: usize, size: usize) -> *mut c_void {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    unsafe { __libc_memalign(alignment, size) }
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
#[unsafe(no_mangle)]
unsafe extern "C" fn posix_memalign(output: *mut *mut c_void, alignment: usize, size: usize) -> i32 {
    HEAP_ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    let allocated = unsafe { __libc_memalign(alignment, size) };
    if allocated.is_null() {
        return 12; // ENOMEM
    }
    unsafe { *output = allocated };
    0
}

/// The definitions above count from the first allocation on, so the process heap is counted.
#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn count_heap() -> bool {
    true
}

/// No hook on this platform, so only the binding's own heap is counted.
#[cfg(not(any(target_os = "macos", all(target_os = "linux", target_env = "gnu"))))]
fn count_heap() -> bool {
    false
}

static SLAB: [f32; 8] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];

fn admit_all(_: Key) -> bool {
    true
}

/// Polls `future` to completion on this thread with a no-op waker.
fn block_on<Task: Future>(future: Task) -> Task::Output {
    let mut future = core::pin::pin!(future);
    let mut context = core::task::Context::from_waker(core::task::Waker::noop());
    loop {
        if let core::task::Poll::Ready(output) = future.as_mut().poll(&mut context) {
            return output;
        }
        std::thread::yield_now();
    }
}

/// One search without an executor, on the calling thread, into an arena made without one.
fn calling_thread_round(index: &DenseIndex, arena: &mut Arena<'_>) -> Result<()> {
    let alone = index.search(rows(&SLAB, 4)?, 2, arena)?;
    assert_eq!(alone.query(0)?.keys.first(), Some(&Key::from(1u64)));
    Ok(())
}

/// One round of every path the budget covers, checking each answer.
fn round(index: &DenseIndex, executor: &Executor, arena: &mut Arena<'_>) -> Result<()> {
    let config = DenseSearchConfig { predicate: Some(Predicate::each(&admit_all)), ..DenseSearchConfig::from(2) };
    let waited = index.deferred(executor).search(rows(&SLAB, 4)?, config, arena)?.wait()?;
    assert_eq!(waited.query(0)?.keys.first(), Some(&Key::from(1u64)));
    let awaited = block_on(index.deferred(executor).search(rows(&SLAB, 4)?, 2, arena)?)?;
    assert_eq!(awaited.query(1)?.keys.first(), Some(&Key::from(2u64)));
    assert_eq!(index.deferred(executor).count(Key::from(1u64))?.wait()?, 1);
    let blocking = index.on(executor).search(rows(&SLAB, 4)?, 2, arena)?;
    assert_eq!(blocking.queries(), 2);
    Ok(())
}

fn main() -> Result<()> {
    let heap_counted = !cfg!(feature = "stub") && count_heap();
    if cfg!(feature = "stub") {
        println!("heap counts are asserted for the engine only, since the stub runs on standard containers");
    }
    let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    index.insert([Key::from(1u64), Key::from(2u64)], &rows(&SLAB, 4)?)?;
    let executor = Executor::for_ram(&[], None, 0)?;
    let mut arena = Arena::new_in(&executor)?;

    round(&index, &executor, &mut arena)?;
    let (binding_before, heap_before) = (ALLOCATIONS.load(Ordering::Relaxed), HEAP_ALLOCATIONS.load(Ordering::Relaxed));
    for _ in 0..50 {
        round(&index, &executor, &mut arena)?;
    }
    let heap = HEAP_ALLOCATIONS.load(Ordering::Relaxed) - heap_before;
    let binding = ALLOCATIONS.load(Ordering::Relaxed) - binding_before;
    assert_eq!(binding, 0, "50 warm rounds allocated {binding} times in the binding");
    if heap_counted {
        assert_eq!(heap, 0, "50 warm rounds allocated {heap} times in the process");
    }
    println!(
        "deferred and blocking calls on an executor: {binding} binding and {heap} heap allocations over 50 warm rounds"
    );

    // Making an arena allocates in the library, by contract, and nothing in the binding.
    drop(Arena::new_in(&executor)?);
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..50 {
        drop(Arena::new_in(&executor)?);
    }
    let binding = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(binding, 0, "50 arenas allocated {binding} times in the binding");

    let mut alone = Arena::new()?;
    calling_thread_round(&index, &mut alone)?;
    let (binding_before, heap_before) = (ALLOCATIONS.load(Ordering::Relaxed), HEAP_ALLOCATIONS.load(Ordering::Relaxed));
    for _ in 0..50 {
        calling_thread_round(&index, &mut alone)?;
    }
    let heap = HEAP_ALLOCATIONS.load(Ordering::Relaxed) - heap_before;
    let binding = ALLOCATIONS.load(Ordering::Relaxed) - binding_before;
    assert_eq!(binding, 0, "50 calls without an executor allocated {binding} times in the binding");
    if heap_counted {
        let bound = CALLING_THREAD_ARENA_SLOTS + CALLING_THREAD_ARENA_CHUNKS;
        assert!(
            heap <= 50 * bound,
            "50 calls without an executor allocated {heap} times, past {CALLING_THREAD_ARENA_SLOTS} arena slot and \
             {CALLING_THREAD_ARENA_CHUNKS} arena chunks a call"
        );
    }
    println!("calls without an executor: {binding} binding and {heap} heap allocations over 50 calls");
    Ok(())
}
