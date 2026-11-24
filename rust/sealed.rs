//! The traits only this crate implements: public within this private module, so callers see them
//! in bounds yet can neither name nor implement them.
//!
//! File: rust/sealed.rs
//! Author: Ash Vardanian
//! Date: November 18, 2026

use crate::types::{Batch, Element, Result, usearch_completion_t, usearch_outcome_t};

/// One verb over one `#[repr(C)]` argument struct: where the arguments point, which C entry
/// point takes them, and what a final call hands back.
pub trait Calling: Sized {
    /// The argument struct of the verb's C entry point.
    type Arguments;
    /// What a successful call hands back.
    type Output;

    /// The arguments over the inputs and outputs `self` holds, valid while `self` stays in place.
    fn arguments(&mut self, outcome: *mut usearch_outcome_t, completion: usearch_completion_t) -> Self::Arguments;

    /// Passes `arguments` to the verb's C entry point.
    ///
    /// # Safety
    /// Everything `arguments` points to stays alive and in place until the outcome is final.
    unsafe fn call(arguments: *mut Self::Arguments);

    /// What the verb hands back once its outcome is `status`.
    fn finish(self, status: Result<()>) -> Result<Self::Output>;
}

/// A verb whose inputs a deferred call hands back beside its answer, like a plan a stop left live.
pub trait Submitting: Calling {
    /// What the verb owned.
    type Inputs;

    /// What `finish` hands back, beside the inputs.
    fn finish_with_inputs(self, status: Result<()>) -> (Result<Self::Output>, Self::Inputs);
}

/// How the verbs of a receiver complete, which decides what each of them answers.
pub trait Completion: Sized {
    /// The output itself when blocking, a `Deferred` holding the verb when deferring.
    type Deferrable<'index, Verb: Calling>;

    /// Runs `verb`, or holds it until it is first polled or waited on.
    fn answer<'index, Verb: Calling>(verb: Verb) -> Result<Self::Deferrable<'index, Verb>>;
}

/// The inputs a completion takes: a blocking receiver takes references and owned values alike,
/// while a deferring one takes only what it owns, since the library reads its inputs after the
/// verb returns. Both reference the arena, which no call reads as an input.
#[diagnostic::on_unimplemented(
    message = "a deferred receiver needs owned inputs",
    label = "the library would still read this reference after the verb returns",
    note = "pass keys in a `Vec` or an array, rows in a NumKong `Matrix` or a `'static` `MatrixView`, a `PathBuf`, \
            `String` or `'static` path, and `'static` configs, handlers and value lookups"
)]
pub trait Accepts<Input> {}

/// A matrix a verb reads its rows from, referenced or owned.
pub trait RowsView {
    /// The scalar type the rows hold.
    type Value: Element;
    /// The rows and their count, refused when the layout cannot cross the boundary.
    fn batch(&self) -> Result<Batch>;
}
