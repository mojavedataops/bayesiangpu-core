//! Special mathematical functions.
//!
//! Provides implementations of the log-gamma function, log-beta function,
//! and digamma function using the Lanczos approximation.

use std::f64::consts::PI;

/// Lanczos approximation coefficients for g=7.
///
/// These provide approximately 10^-10 precision for positive real arguments.
const LANCZOS_G: f64 = 7.0;
const LANCZOS_COEFFICIENTS: [f64; 9] = [
    0.999_999_999_999_809_9,
    676.5203681218851,
    -1259.1392167224028,
    771.323_428_777_653_1,
    -176.615_029_162_140_6,
    12.507343278686905,
    -0.13857109526572012,
    9.984_369_578_019_572e-6,
    1.5056327351493116e-7,
];

/// Compute the natural logarithm of the gamma function.
///
/// Uses the Lanczos approximation with g=7 and 9 coefficients,
/// providing approximately 10^-10 precision for positive real arguments.
///
/// # Arguments
///
/// * `x` - The argument to the gamma function (must be positive)
///
/// # Returns
///
/// The natural logarithm of Γ(x). Returns `f64::INFINITY` for x ≤ 0.
///
/// # Examples
///
/// ```
/// use bayesian_core::math::ln_gamma;
///
/// // ln(Γ(1)) = ln(1) = 0
/// assert!((ln_gamma(1.0) - 0.0).abs() < 1e-10);
///
/// // ln(Γ(2)) = ln(1!) = 0
/// assert!((ln_gamma(2.0) - 0.0).abs() < 1e-10);
///
/// // ln(Γ(3)) = ln(2!) = ln(2)
/// assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < 1e-10);
///
/// // ln(Γ(0.5)) = ln(√π)
/// assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-10);
/// ```
pub fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    let x = x - 1.0;
    let mut sum = LANCZOS_COEFFICIENTS[0];
    for (i, &coeff) in LANCZOS_COEFFICIENTS.iter().enumerate().skip(1) {
        sum += coeff / (x + i as f64);
    }

    let t = x + LANCZOS_G + 0.5;
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Compute the natural logarithm of the beta function.
///
/// The beta function is defined as:
/// B(a, b) = Γ(a) * Γ(b) / Γ(a + b)
///
/// So ln(B(a, b)) = ln(Γ(a)) + ln(Γ(b)) - ln(Γ(a + b))
///
/// # Arguments
///
/// * `a` - First parameter (must be positive)
/// * `b` - Second parameter (must be positive)
///
/// # Returns
///
/// The natural logarithm of B(a, b). Returns `f64::INFINITY` if either argument is ≤ 0.
///
/// # Examples
///
/// ```
/// use bayesian_core::math::ln_beta;
///
/// // B(1, 1) = 1, so ln(B(1,1)) = 0
/// assert!((ln_beta(1.0, 1.0) - 0.0).abs() < 1e-10);
///
/// // B(0.5, 0.5) = π, so ln(B(0.5, 0.5)) = ln(π)
/// assert!((ln_beta(0.5, 0.5) - std::f64::consts::PI.ln()).abs() < 1e-10);
/// ```
pub fn ln_beta(a: f64, b: f64) -> f64 {
    if a <= 0.0 || b <= 0.0 {
        return f64::INFINITY;
    }
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Compute the digamma function (derivative of ln(Γ(x))).
///
/// Uses the recurrence relation ψ(x+1) = ψ(x) + 1/x and an asymptotic
/// expansion for large arguments.
///
/// # Arguments
///
/// * `x` - The argument (must be positive)
///
/// # Returns
///
/// The digamma function ψ(x) = d/dx ln(Γ(x)). Returns `f64::NEG_INFINITY` for x ≤ 0.
///
/// # Examples
///
/// ```
/// use bayesian_core::math::digamma;
///
/// // ψ(1) = -γ (Euler-Mascheroni constant ≈ -0.5772)
/// assert!((digamma(1.0) - (-0.5772156649015329)).abs() < 1e-6);
/// ```
pub fn digamma(mut x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use recurrence to shift x to a larger value where asymptotic expansion is accurate
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion for large x
    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;

    result += x.ln() - 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x4) - 1.0 / (252.0 * x6);

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_ln_gamma_integers() {
        // Γ(n) = (n-1)! for positive integers
        assert!((ln_gamma(1.0) - 0.0).abs() < EPSILON); // 0! = 1
        assert!((ln_gamma(2.0) - 0.0).abs() < EPSILON); // 1! = 1
        assert!((ln_gamma(3.0) - 2.0_f64.ln()).abs() < EPSILON); // 2! = 2
        assert!((ln_gamma(4.0) - 6.0_f64.ln()).abs() < EPSILON); // 3! = 6
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < EPSILON); // 4! = 24
        assert!((ln_gamma(6.0) - 120.0_f64.ln()).abs() < EPSILON); // 5! = 120
    }

    #[test]
    fn test_ln_gamma_half_integers() {
        // Γ(0.5) = √π
        let expected = 0.5 * PI.ln();
        assert!((ln_gamma(0.5) - expected).abs() < EPSILON);

        // Γ(1.5) = 0.5 * √π
        let expected = (0.5_f64).ln() + 0.5 * PI.ln();
        assert!((ln_gamma(1.5) - expected).abs() < EPSILON);

        // Γ(2.5) = 0.75 * √π
        let expected = (0.75_f64).ln() + 0.5 * PI.ln();
        assert!((ln_gamma(2.5) - expected).abs() < EPSILON);
    }

    #[test]
    fn test_ln_gamma_negative_returns_infinity() {
        assert!(ln_gamma(0.0).is_infinite());
        assert!(ln_gamma(-1.0).is_infinite());
        assert!(ln_gamma(-0.5).is_infinite());
    }

    #[test]
    fn test_ln_beta_special_values() {
        // B(1, 1) = 1
        assert!((ln_beta(1.0, 1.0) - 0.0).abs() < EPSILON);

        // B(0.5, 0.5) = π
        assert!((ln_beta(0.5, 0.5) - PI.ln()).abs() < EPSILON);

        // B(1, n) = 1/n
        for n in 1..=10 {
            let expected = -(n as f64).ln();
            assert!(
                (ln_beta(1.0, n as f64) - expected).abs() < EPSILON,
                "B(1, {}) failed: expected {}, got {}",
                n,
                expected,
                ln_beta(1.0, n as f64)
            );
        }
    }

    #[test]
    fn test_ln_beta_symmetry() {
        // B(a, b) = B(b, a)
        let test_cases = [(2.0, 3.0), (0.5, 1.5), (5.0, 2.0), (0.1, 10.0)];
        for (a, b) in test_cases {
            assert!(
                (ln_beta(a, b) - ln_beta(b, a)).abs() < EPSILON,
                "Symmetry failed for ({}, {})",
                a,
                b
            );
        }
    }

    #[test]
    fn test_ln_beta_negative_returns_infinity() {
        assert!(ln_beta(0.0, 1.0).is_infinite());
        assert!(ln_beta(1.0, 0.0).is_infinite());
        assert!(ln_beta(-1.0, 1.0).is_infinite());
        assert!(ln_beta(1.0, -1.0).is_infinite());
    }

    #[test]
    fn test_digamma_known_values() {
        // ψ(1) = -γ (Euler-Mascheroni constant)
        let euler_mascheroni = 0.5772156649015329;
        assert!(
            (digamma(1.0) - (-euler_mascheroni)).abs() < 1e-6,
            "digamma(1) = {}, expected {}",
            digamma(1.0),
            -euler_mascheroni
        );

        // ψ(2) = 1 - γ
        assert!(
            (digamma(2.0) - (1.0 - euler_mascheroni)).abs() < 1e-6,
            "digamma(2) = {}, expected {}",
            digamma(2.0),
            1.0 - euler_mascheroni
        );

        // ψ(0.5) = -γ - 2*ln(2)
        let expected = -euler_mascheroni - 2.0 * 2.0_f64.ln();
        assert!(
            (digamma(0.5) - expected).abs() < 1e-6,
            "digamma(0.5) = {}, expected {}",
            digamma(0.5),
            expected
        );
    }

    #[test]
    fn test_digamma_recurrence() {
        // ψ(x+1) = ψ(x) + 1/x
        let x = 3.5;
        let lhs = digamma(x + 1.0);
        let rhs = digamma(x) + 1.0 / x;
        assert!(
            (lhs - rhs).abs() < 1e-6,
            "Recurrence failed: digamma({}) = {}, digamma({}) + 1/{} = {}",
            x + 1.0,
            lhs,
            x,
            x,
            rhs
        );
    }

    #[test]
    fn test_digamma_negative_returns_neg_infinity() {
        assert!(digamma(0.0).is_infinite() && digamma(0.0) < 0.0);
        assert!(digamma(-1.0).is_infinite() && digamma(-1.0) < 0.0);
    }
}
