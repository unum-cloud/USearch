//! Counts what the binding allocates on its own heap once warm, which is nothing: executors,
//! arenas, blocking calls, deferred calls and their callbacks all keep their state inline.
//!
//! It installs a counting global allocator, so it runs as a program of its own, outside libtest.
//!
//! File: rust/allocations.rs
//! Author: Ash Vardanian
//! Date: November 18, 2026

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
    drop(Arena::new_in(executor)?);
    Ok(())
}

fn main() -> Result<()> {
    let index = DenseIndex::init_ram(&DenseOptions { dimensions: 4, ..DenseOptions::default() }, &[])?;
    index.insert([Key::from(1u64), Key::from(2u64)], &rows(&SLAB, 4)?)?;
    let executor = Executor::for_ram(&[], None, 0)?;
    let mut arena = Arena::new_in(&executor)?;

    round(&index, &executor, &mut arena)?;
    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..50 {
        round(&index, &executor, &mut arena)?;
    }
    let allocations = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(allocations, 0, "50 warm rounds allocated {allocations} times");
    println!("deferred, blocking and arena paths: 0 allocations over 50 warm rounds");
    Ok(())
}
