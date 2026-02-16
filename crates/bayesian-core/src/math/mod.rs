//! Mathematical special functions for probability distributions.
//!
//! This module provides implementations of special functions needed for
//! computing log-probabilities of various distributions.

pub mod special;

pub use special::{digamma, ln_beta, ln_gamma};
