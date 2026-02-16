"""
Streamlit app for Bayesian inference with BayesianGPU

Run with: streamlit run streamlit_app.py

Requirements: pip install streamlit bayesiangpu
"""

import streamlit as st
from bayesiangpu import Model, Beta, Binomial, Normal, HalfNormal, sample

st.set_page_config(page_title="BayesianGPU Demo", layout="wide")

st.title("BayesianGPU - Bayesian Inference Demo")

# Sidebar for model selection
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Beta-Binomial (Proportion)", "Normal (Mean Estimation)"]
)

# Sampling parameters
st.sidebar.markdown("### Sampling Parameters")
num_samples = st.sidebar.slider("Samples per chain", 100, 2000, 500)
num_chains = st.sidebar.slider("Number of chains", 1, 8, 4)
seed = st.sidebar.number_input("Random seed", value=42)

if model_type == "Beta-Binomial (Proportion)":
    st.markdown("## Beta-Binomial Model")
    st.markdown("""
    Estimate the probability of success from observed data.

    **Model:**
    - Prior: theta ~ Beta(alpha, beta)
    - Likelihood: data ~ Binomial(n, theta)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prior Parameters")
        alpha = st.slider("Alpha (prior successes + 1)", 0.1, 10.0, 1.0)
        beta_param = st.slider("Beta (prior failures + 1)", 0.1, 10.0, 1.0)

    with col2:
        st.markdown("### Observed Data")
        n_trials = st.number_input("Number of trials", min_value=1, max_value=1000, value=100)
        successes = st.number_input("Number of successes", min_value=0, max_value=n_trials, value=65)

    if st.button("Run Inference", type="primary"):
        with st.spinner("Running MCMC sampling..."):
            model = Model()
            model.param("theta", Beta(alpha, beta_param))
            model.observe(Binomial(n_trials, "theta"), [float(successes)])

            result = sample(model, num_samples=num_samples, num_chains=num_chains, seed=seed)

        st.success("Sampling complete!")

        # Results
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Posterior Summary")
            summary = result.summarize("theta")
            st.metric("Posterior Mean", f"{summary.mean:.4f}")
            st.metric("95% Credible Interval", f"[{summary.q025:.4f}, {summary.q975:.4f}]")
            st.metric("Standard Deviation", f"{summary.std:.4f}")

        with col2:
            st.markdown("### Diagnostics")
            st.metric("R-hat", f"{summary.rhat:.4f}", delta="Good" if summary.rhat < 1.01 else "Warning")
            st.metric("ESS", f"{summary.ess:.0f}", delta="Good" if summary.ess > 400 else "Low")
            st.metric("Divergences", result.diagnostics.divergences)

        # Full summary table
        st.markdown("### Full Summary Table")
        st.code(result.format_summary())

        # Warnings
        warnings = result.warnings()
        if warnings:
            st.warning("Warnings:\n" + "\n".join(f"- {w}" for w in warnings))

else:  # Normal model
    st.markdown("## Normal Model - Mean Estimation")
    st.markdown("""
    Estimate the mean and standard deviation of normally distributed data.

    **Model:**
    - Prior: mu ~ Normal(prior_mean, prior_std)
    - Prior: sigma ~ HalfNormal(scale)
    - Likelihood: data ~ Normal(mu, sigma)
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Prior Parameters")
        prior_mean = st.slider("Prior mean (mu)", -10.0, 10.0, 0.0)
        prior_std = st.slider("Prior std for mu", 0.1, 20.0, 10.0)
        sigma_scale = st.slider("HalfNormal scale for sigma", 0.1, 10.0, 5.0)

    with col2:
        st.markdown("### Observed Data")
        data_str = st.text_area(
            "Enter data (comma-separated)",
            value="2.3, 2.1, 2.5, 2.4, 2.2, 2.6, 2.3, 2.4, 2.5, 2.2"
        )
        try:
            data = [float(x.strip()) for x in data_str.split(",") if x.strip()]
            st.info(f"Parsed {len(data)} data points. Sample mean: {sum(data)/len(data):.3f}")
        except ValueError:
            st.error("Invalid data format. Please use comma-separated numbers.")
            data = []

    if data and st.button("Run Inference", type="primary"):
        with st.spinner("Running MCMC sampling..."):
            model = Model()
            model.param("mu", Normal(prior_mean, prior_std))
            model.param("sigma", HalfNormal(sigma_scale))
            model.observe(Normal("mu", "sigma"), data)

            result = sample(model, num_samples=num_samples, num_chains=num_chains, seed=seed)

        st.success("Sampling complete!")

        # Results for mu
        st.markdown("### Posterior Summary - Mean (mu)")
        col1, col2 = st.columns(2)

        mu_summary = result.summarize("mu")
        with col1:
            st.metric("Posterior Mean", f"{mu_summary.mean:.4f}")
            st.metric("95% CI", f"[{mu_summary.q025:.4f}, {mu_summary.q975:.4f}]")
        with col2:
            st.metric("R-hat", f"{mu_summary.rhat:.4f}")
            st.metric("ESS", f"{mu_summary.ess:.0f}")

        # Results for sigma
        st.markdown("### Posterior Summary - Std Dev (sigma)")
        col1, col2 = st.columns(2)

        sigma_summary = result.summarize("sigma")
        with col1:
            st.metric("Posterior Mean", f"{sigma_summary.mean:.4f}")
            st.metric("95% CI", f"[{sigma_summary.q025:.4f}, {sigma_summary.q975:.4f}]")
        with col2:
            st.metric("R-hat", f"{sigma_summary.rhat:.4f}")
            st.metric("ESS", f"{sigma_summary.ess:.0f}")

        # Full summary
        st.markdown("### Full Summary Table")
        st.code(result.format_summary())

st.sidebar.markdown("---")
st.sidebar.markdown("Built with [BayesianGPU](https://github.com/mojavedataops/bayesiangpu-core)")
