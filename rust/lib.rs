//! USearch - Smaller & Faster Single-File Vector Search Engine from Unum.
//!
//! The crate links the pre-instantiated C library and consumes it through its stable C headers,
//! so no index template is instantiated here and nothing but the vocabulary below crosses the
//! boundary. One module per modality, named after the header it mirrors.
//!
//! ```no_run
//! use usearch::{DenseIndex, ScalarKind};
//! use usearch::dense::{Metric, Options};
//!
//! let index = DenseIndex::new(&Options {
//!     dimensions: 768,
//!     metric: Metric::Angular,
//!     quantization: ScalarKind::I8,
//!     ..Options::default()
//! })?;
//! index.reserve(1_000_000)?;
//! # Ok::<(), usearch::Error>(())
//! ```

pub mod types;

#[path = "dense_index.rs"]
pub mod dense;

pub use dense::Index as DenseIndex;
pub use types::{
    capabilities_available, capabilities_compiled, rows, rows_mut, version, Addressing, Device, Distance, Error,
    HugePages, IoMode, Key, Residency, Result, Scalar, ScalarKind, Status,
};
