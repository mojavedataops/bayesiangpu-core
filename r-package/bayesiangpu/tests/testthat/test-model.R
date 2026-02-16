test_that("Model can be created and parameters added", {
  model <- Model()
  expect_s3_class(model, "BayesianModel")

  model$param("theta", Beta(1, 1))
  expect_equal(model$param_names(), "theta")
  expect_equal(model$num_params(), 1)
  expect_false(model$has_likelihood())
})

test_that("Model supports chaining with pipes", {
  model <- Model() |>
    param("mu", Normal(0, 10)) |>
    param("sigma", HalfNormal(1))

  expect_equal(model$num_params(), 2)
  expect_equal(model$param_names(), c("mu", "sigma"))
})

test_that("Model can have observations", {
  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  expect_true(model$has_likelihood())
})

test_that("Model converts to JSON correctly", {
  model <- Model() |>
    param("theta", Beta(1, 1)) |>
    observe(Binomial(100, "theta"), 65)

  json <- model$to_json()
  parsed <- jsonlite::fromJSON(json)

  expect_equal(length(parsed$priors), 1)
  expect_equal(parsed$priors[[1]]$name, "theta")
  expect_equal(parsed$likelihood$observed, 65)
})
