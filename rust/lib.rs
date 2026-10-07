//! USearch - Billion-Scale Multi-Modal Search, Clustering, & Sparse Algebra Engine for CPUs, GPUs, & NVMe Devices.
//!
//! The crate links the pre-instantiated C library and consumes it through its stable C headers,
//! so no index template is instantiated here and nothing but the vocabulary below crosses the
//! boundary. One module per modality, named after the header it mirrors.
//!
//! ```no_run
//! use usearch::{DenseIndex, Executor, ScalarKind};
//! use usearch::dense::{Metric, Options};
//!
//! let mut options = Options::default();
//! options.dimensions = 768;
//! options.metric = Metric::Angular;
//! options.quantization = ScalarKind::I8;
//!
//! let index = DenseIndex::new(&options)?;
//! let executor = Executor::for_ram(&[], 0, 0)?;
//! index.reserve(&executor, 1_000_000)?;
//! # Ok::<(), usearch::Error>(())
//! ```

pub mod types;

#[path = "dense_index.rs"]
pub mod dense;

pub use dense::DenseIndex;
pub use types::{
    capabilities_available, capabilities_compiled, rows, rows_mut, version, Arena, Continuation, Distance, Error,
    Executor, Key, Mood, Progress, Residency, Result, Scalar, ScalarKind, Status,
};
