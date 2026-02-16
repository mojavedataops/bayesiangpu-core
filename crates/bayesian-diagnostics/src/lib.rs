//! MCMC Convergence Diagnostics for BayesianGPU
//!
//! This crate provides diagnostic tools for assessing MCMC sampling quality:
//!
//! - **R-hat (Gelman-Rubin)**: Measures convergence by comparing between-chain
//!   and within-chain variance. Values close to 1.0 indicate convergence.
//!
//! - **ESS (Effective Sample Size)**: Estimates the number of independent samples
//!   in an autocorrelated chain. Low ESS suggests more samples are needed.
//!
//! - **Divergences**: Tracks numerical instabilities during sampling that may
//!   indicate problems with the model or sampler configuration.
//!
//! - **Posterior Summary**: Computes mean, standard deviation, quantiles, and
//!   combines all diagnostics into a comprehensive summary.
//!
//! - **LOO-CV**: Leave-One-Out Cross-Validation via Pareto Smoothed Importance
//!   Sampling (PSIS) for model comparison.
//!
//! - **WAIC**: Widely Applicable Information Criterion for model comparison.
//!
//! # Quick Start
//!
//! ```rust
//! use bayesian_diagnostics::{rhat, ess, summarize};
//!
//! // MCMC samples from multiple chains (need sufficient samples for ESS)
//! let chains: Vec<Vec<f64>> = (0..4).map(|c| {
//!     (0..200).map(|i| 1.0 + 0.05 * (((i + c * 7) % 100) as f64 / 50.0 - 1.0)).collect()
//! }).collect();
//!
//! // Check convergence
//! let rhat_val = rhat(&chains);
//! println!("R-hat: {:.3}", rhat_val);
//! assert!(rhat_val < 1.1, "Chains have not converged");
//!
//! // Check effective sample size
//! let ess_val = ess(&chains);
//! println!("ESS: {:.0}", ess_val);
//! assert!(ess_val >= 10.0, "Need more samples");
//!
//! // Get full summary
//! let summary = summarize(&chains);
//! println!("Mean: {:.3} +/- {:.3}", summary.mean, summary.std);
//! println!("95% CI: [{:.3}, {:.3}]", summary.q2_5, summary.q97_5);
//! ```
//!
//! # Model Comparison
//!
//! ```rust
//! use bayesian_diagnostics::loo::loo;
//! use bayesian_diagnostics::waic::waic;
//!
//! // Compute pointwise log-likelihood for each observation and sample
//! let log_lik: Vec<Vec<f64>> = (0..100).map(|s| {
//!     (0..20).map(|n| -0.5 * ((s as f64 * 0.01 - n as f64 * 0.05).powi(2))).collect()
//! }).collect();
//!
//! // LOO-CV (recommended)
//! let loo_result = loo(&log_lik);
//! println!("ELPD-LOO: {:.1} +/- {:.1}", loo_result.elpd_loo, loo_result.se_elpd_loo);
//! println!("LOOIC: {:.1}", loo_result.looic);
//!
//! // WAIC (alternative)
//! let waic_result = waic(&log_lik);
//! println!("WAIC: {:.1} +/- {:.1}", waic_result.waic, waic_result.se_waic);
//! ```
//!
//! # Recommended Thresholds
//!
//! | Diagnostic | Good | Acceptable | Poor |
//! |------------|------|------------|------|
//! | R-hat | < 1.01 | < 1.05 | >= 1.1 |
//! | Bulk ESS | > 400 | > 100 | < 100 |
//! | Tail ESS | > 400 | > 100 | < 100 |
//! | Divergences | 0% | < 1% | > 5% |
//! | Pareto k | < 0.7 | < 1.0 | >= 1.0 |
//!
//! # Module Overview
//!
//! - [`rhat`]: R-hat convergence diagnostic
//! - [`ess`]: Effective sample size calculations
//! - [`divergences`]: Divergence tracking and analysis
//! - [`summary`]: Posterior summary statistics
//! - [`loo`]: PSIS-LOO cross-validation for model comparison
//! - [`waic`]: WAIC information criterion for model comparison
//! - [`compare`]: Model comparison utilities

pub mod compare;
pub mod divergences;
pub mod ess;
pub mod loo;
pub mod rhat;
pub mod summary;
pub mod waic;

// Re-export main types and functions for convenience
pub use compare::{
    compare, compare_pseudo_bma, ComparisonResult, ModelComparisonEntry, ModelCriterion,
};
pub use divergences::{analyze_divergences, is_acceptable, DivergenceInfo, DivergenceTracker};
pub use ess::{ess, ess_bulk, ess_from_array, ess_tail, EssResult};
pub use loo::{loo, loo_compare, loo_from_array, pareto_k_threshold, psis, LooResult, PsisResult};
pub use rhat::{
    is_converged, rhat, rhat_detailed, rhat_from_array, rhat_rank_normalized, RhatResult,
};
pub use summary::{
    credible_interval, hpd_interval, summarize, summarize_all, summarize_named, PosteriorSummary,
    SummaryTable,
};
pub use waic::{waic, waic_compare, waic_from_array, WaicResult};

/// Overall diagnostic status for a parameter or set of parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticStatus {
    /// All diagnostics pass recommended thresholds
    Good,
    /// Diagnostics are acceptable but could be improved
    Acceptable,
    /// Diagnostics indicate potential problems
    Warning,
    /// Diagnostics indicate serious problems
    Error,
}

impl DiagnosticStatus {
    /// Get a human-readable description.
    pub fn description(&self) -> &'static str {
        match self {
            DiagnosticStatus::Good => "All diagnostics pass",
            DiagnosticStatus::Acceptable => "Acceptable but could use more samples",
            DiagnosticStatus::Warning => "Some diagnostics indicate potential issues",
            DiagnosticStatus::Error => "Serious sampling issues detected",
        }
    }

    /// Check if status is acceptable (Good or Acceptable).
    pub fn is_ok(&self) -> bool {
        matches!(self, DiagnosticStatus::Good | DiagnosticStatus::Acceptable)
    }
}

/// Comprehensive diagnostic check for a parameter.
///
/// # Arguments
/// * `chains` - Vector of chains
/// * `divergence_info` - Optional divergence information
///
/// # Returns
/// Overall diagnostic status
pub fn check_diagnostics(
    chains: &[Vec<f64>],
    divergence_info: Option<&DivergenceInfo>,
) -> DiagnosticStatus {
    let rhat_val = rhat::rhat(chains);
    let ess_bulk_val = ess::ess_bulk(chains);
    let ess_tail_val = ess::ess_tail(chains);

    // Check R-hat
    let rhat_ok = rhat_val.is_finite() && rhat_val < 1.01;
    let rhat_acceptable = rhat_val.is_finite() && rhat_val < 1.05;

    // Check ESS
    let ess_ok = ess_bulk_val >= 400.0 && ess_tail_val >= 400.0;
    let ess_acceptable = ess_bulk_val >= 100.0 && ess_tail_val >= 100.0;

    // Check divergences
    let divergence_ok = divergence_info.map(|d| d.total == 0).unwrap_or(true);
    let divergence_acceptable = divergence_info.map(|d| d.fraction < 0.01).unwrap_or(true);

    // Determine overall status
    if rhat_ok && ess_ok && divergence_ok {
        DiagnosticStatus::Good
    } else if rhat_acceptable && ess_acceptable && divergence_acceptable {
        DiagnosticStatus::Acceptable
    } else if rhat_val.is_finite() && rhat_val < 1.1 && ess_bulk_val >= 50.0 {
        DiagnosticStatus::Warning
    } else {
        DiagnosticStatus::Error
    }
}

/// Format a diagnostic report for all parameters.
pub fn diagnostic_report(
    summaries: &[PosteriorSummary],
    divergence_info: Option<&DivergenceInfo>,
) -> String {
    let mut report = String::new();

    report.push_str("=== MCMC Diagnostic Report ===\n\n");

    // Divergence summary
    if let Some(div) = divergence_info {
        report.push_str(&format!(
            "Divergences: {} ({:.2}%)\n",
            div.total,
            div.fraction * 100.0
        ));
        if !div.is_acceptable() {
            report.push_str("  WARNING: High divergence rate!\n");
        }
        report.push('\n');
    }

    // Parameter summary table
    report.push_str(&format!(
        "{:>15} {:>8} {:>8} {:>10} {:>10} {:>8}\n",
        "Parameter", "R-hat", "Bulk ESS", "Tail ESS", "Mean", "Std"
    ));
    report.push_str(&"-".repeat(70));
    report.push('\n');

    let mut warnings = Vec::new();

    for s in summaries {
        let name = s.name.as_deref().unwrap_or("?");
        let rhat_flag = if s.rhat > 1.01 { "*" } else { "" };
        let ess_flag = if s.ess_bulk < 100.0 || s.ess_tail < 100.0 {
            "*"
        } else {
            ""
        };

        report.push_str(&format!(
            "{:>15} {:>7.3}{} {:>7.0}{} {:>10.0} {:>10.3} {:>8.3}\n",
            name, s.rhat, rhat_flag, s.ess_bulk, ess_flag, s.ess_tail, s.mean, s.std
        ));

        if !s.warnings().is_empty() {
            for w in s.warnings() {
                warnings.push(format!("  {}: {}", name, w));
            }
        }
    }

    if !warnings.is_empty() {
        report.push_str("\nWarnings:\n");
        for w in warnings {
            report.push_str(&w);
            report.push('\n');
        }
    }

    report.push_str("\n* indicates potential issues\n");

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_good_chains() -> Vec<Vec<f64>> {
        // Well-mixed chains with enough samples for ESS >= 400
        // Use pseudo-random sequence with good mixing properties
        let mut chains = Vec::new();
        for seed in [17u64, 31, 47, 61] {
            let chain: Vec<f64> = (0..500)
                .map(|i| {
                    // Simple LCG for deterministic "random" values
                    let x = ((seed.wrapping_mul(1103515245).wrapping_add(i as u64 * 12345)) % 1000)
                        as f64;
                    1.0 + (x / 1000.0 - 0.5) * 0.1 // Values centered around 1.0
                })
                .collect();
            chains.push(chain);
        }
        chains
    }

    fn make_bad_chains() -> Vec<Vec<f64>> {
        // Chains stuck at different values
        vec![
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        ]
    }

    #[test]
    fn test_check_diagnostics_good() {
        let chains = make_good_chains();
        let status = check_diagnostics(&chains, None);
        assert!(status.is_ok());
    }

    #[test]
    fn test_check_diagnostics_bad() {
        let chains = make_bad_chains();
        let status = check_diagnostics(&chains, None);
        assert_eq!(status, DiagnosticStatus::Error);
    }

    #[test]
    fn test_check_diagnostics_with_divergences() {
        let chains = make_good_chains();
        let div_info = DivergenceInfo::new(vec![0, 0, 0, 0], 10);
        let status = check_diagnostics(&chains, Some(&div_info));
        assert!(status.is_ok());

        let bad_div = DivergenceInfo::new(vec![5, 5, 5, 5], 10);
        let status = check_diagnostics(&chains, Some(&bad_div));
        assert!(!status.is_ok() || status == DiagnosticStatus::Acceptable);
    }

    #[test]
    fn test_diagnostic_report() {
        let chains = make_good_chains();
        let summary = summarize_named(&chains, Some("theta".to_string()));
        let div_info = DivergenceInfo::none(4, 10);

        let report = diagnostic_report(&[summary], Some(&div_info));
        assert!(report.contains("theta"));
        assert!(report.contains("R-hat"));
        assert!(report.contains("Divergences"));
    }

    #[test]
    fn test_diagnostic_status() {
        assert!(DiagnosticStatus::Good.is_ok());
        assert!(DiagnosticStatus::Acceptable.is_ok());
        assert!(!DiagnosticStatus::Warning.is_ok());
        assert!(!DiagnosticStatus::Error.is_ok());
    }
}
