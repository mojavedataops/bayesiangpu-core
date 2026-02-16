"""Basic tests for bayesiangpu Python package"""

import pytest


def test_import():
    """Test that the package can be imported"""
    import bayesiangpu
    assert hasattr(bayesiangpu, "__version__")
    assert hasattr(bayesiangpu, "Model")
    assert hasattr(bayesiangpu, "sample")


def test_distributions():
    """Test distribution factory functions"""
    from bayesiangpu import Normal, HalfNormal, Beta, Gamma, Uniform

    # Create distributions
    normal = Normal(0, 1)
    assert normal.dist_type == "Normal"

    half_normal = HalfNormal(1)
    assert half_normal.dist_type == "HalfNormal"

    beta = Beta(1, 1)
    assert beta.dist_type == "Beta"

    gamma = Gamma(2, 1)
    assert gamma.dist_type == "Gamma"

    uniform = Uniform(0, 10)
    assert uniform.dist_type == "Uniform"


def test_model_building():
    """Test model building API"""
    from bayesiangpu import Model, Beta, Binomial

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    assert model.num_params == 1
    assert model.param_names == ["theta"]
    assert model.has_likelihood


def test_model_no_likelihood():
    """Test model without likelihood"""
    from bayesiangpu import Model, Normal

    model = Model()
    model.param("mu", Normal(0, 10))

    assert model.num_params == 1
    assert not model.has_likelihood


def test_beta_binomial_inference():
    """Test Beta-Binomial model inference"""
    from bayesiangpu import Model, Beta, Binomial, sample

    # Define model
    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    # Run quick inference
    result = sample(model, num_samples=500, num_warmup=500, num_chains=2, seed=42)

    # Check result structure
    assert "theta" in result.param_names
    assert result.num_samples == 500
    assert result.num_chains == 2

    # Check samples
    samples = result.get_samples("theta")
    assert len(samples) == 1000  # 500 * 2 chains

    # Check summary
    summary = result.summarize("theta")
    assert 0 < summary.mean < 1  # Should be around 0.65
    assert summary.std > 0


def test_diagnostics():
    """Test diagnostic output"""
    from bayesiangpu import Model, Beta, Binomial, sample

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    result = sample(model, num_samples=500, num_warmup=500, num_chains=2, seed=42)

    # Check diagnostics
    diag = result.diagnostics
    assert "theta" in diag.rhat
    assert "theta" in diag.ess
    assert diag.divergences >= 0


def test_chain_samples():
    """Test getting samples by chain"""
    from bayesiangpu import Model, Beta, Binomial, sample

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    result = sample(model, num_samples=200, num_warmup=200, num_chains=3, seed=42)

    chain_samples = result.get_chain_samples("theta")
    assert len(chain_samples) == 3
    assert all(len(chain) == 200 for chain in chain_samples)


def test_summary_table():
    """Test summary table formatting"""
    from bayesiangpu import Model, Beta, Binomial, sample

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    result = sample(model, num_samples=200, num_warmup=200, num_chains=2, seed=42)

    # Get formatted summary
    summary_str = result.format_summary()
    assert "theta" in summary_str
    assert "Mean" in summary_str
    assert "R-hat" in summary_str


def test_warnings():
    """Test warning generation"""
    from bayesiangpu import Model, Beta, Binomial, sample

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    result = sample(model, num_samples=100, num_warmup=100, num_chains=2, seed=42)

    warnings = result.warnings()
    # With few samples, we might have warnings about low ESS
    assert isinstance(warnings, list)


def test_normal_likelihood():
    """Test model with Normal likelihood"""
    from bayesiangpu import Model, Normal, HalfNormal, sample

    data = [2.3, 2.1, 2.5, 2.4, 2.2]

    model = Model()
    model.param("mu", Normal(0, 10))
    model.param("sigma", HalfNormal(5))
    model.observe(Normal("mu", "sigma"), data)

    result = sample(model, num_samples=200, num_warmup=200, num_chains=2, seed=42)

    assert "mu" in result.param_names
    assert "sigma" in result.param_names

    mu_summary = result.summarize("mu")
    # Mean should be close to data mean (~2.3)
    assert 1.0 < mu_summary.mean < 4.0


def test_parameter_reference():
    """Test parameter references in distributions"""
    from bayesiangpu import Model, Beta, Binomial

    model = Model()
    model.param("p", Beta(2, 2))
    model.observe(Binomial(50, "p"), [25])  # 'p' references the parameter

    assert model.has_likelihood


@pytest.mark.parametrize("seed", [1, 42, 123, 999])
def test_reproducibility(seed):
    """Test that same seed gives same results"""
    from bayesiangpu import Model, Beta, Binomial, sample

    model = Model()
    model.param("theta", Beta(1, 1))
    model.observe(Binomial(100, "theta"), [65])

    result1 = sample(model, num_samples=100, num_warmup=100, num_chains=1, seed=seed)
    result2 = sample(model, num_samples=100, num_warmup=100, num_chains=1, seed=seed)

    samples1 = result1.get_samples("theta")
    samples2 = result2.get_samples("theta")

    assert samples1 == samples2
