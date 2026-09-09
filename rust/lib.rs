//! USearch - Smaller & Faster Single-File Vector Search Engine from Unum.
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
//! index.reserve(1_000_000)?;
//! let executor = Executor::for_ram(&[], 0)?;
//! # Ok::<(), usearch::Error>(())
//! ```

pub mod types;

#[path = "dense_index.rs"]
pub mod dense;

pub use dense::DenseIndex;
pub use types::{
    capabilities_available, capabilities_compiled, rows, rows_mut, version, Continuation, Distance, Error, Executor,
    Key, Residency, Result, Scalar, ScalarKind, Status,
};
