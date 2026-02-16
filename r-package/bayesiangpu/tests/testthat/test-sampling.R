test_that("bg_sample runs successfully on simple model", {
  skip_on_cran()

  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  result <- bg_sample(model, num_samples = 100, num_warmup = 100, num_chains = 2, seed = 42)

  expect_s3_class(result, "InferenceResult")
  expect_equal(result$param_names, "theta")
  expect_equal(length(result$samples$theta), 200)  # 2 chains * 100 samples
})

test_that("quick_sample works", {
  skip_on_cran()

  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  result <- quick_sample(model, seed = 42)

  expect_s3_class(result, "InferenceResult")
})

test_that("result summarize works", {
  skip_on_cran()

  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  result <- bg_sample(model, num_samples = 100, num_warmup = 100, num_chains = 2, seed = 42)
  summary <- result$summarize("theta")

  expect_true("mean" %in% names(summary))
  expect_true("std" %in% names(summary))
  expect_true("rhat" %in% names(summary))
  expect_true("ess" %in% names(summary))

  # Posterior mean should be near 65/100 = 0.65
  expect_true(summary$mean > 0.5 && summary$mean < 0.8)
})

test_that("result converts to tibble", {
  skip_on_cran()

  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  result <- quick_sample(model, seed = 42)
  tbl <- result$as_tibble()

  expect_s3_class(tbl, "tbl_df")
  expect_true("theta" %in% names(tbl))
})

test_that("multi-parameter model works", {
  skip_on_cran()

  model <- Model() |>
    param("mu", Normal(0, 10)) |>
    param("sigma", HalfNormal(1)) |>
    observe(Normal("mu", "sigma"), c(2.3, 2.1, 2.5, 2.4, 2.2))

  result <- bg_sample(model, num_samples = 100, num_warmup = 100, num_chains = 2, seed = 42)

  expect_equal(length(result$param_names), 2)
  expect_true("mu" %in% result$param_names)
  expect_true("sigma" %in% result$param_names)
})
