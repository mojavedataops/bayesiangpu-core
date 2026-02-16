test_that("distribution factories create valid specifications", {
  # Normal
  dist <- Normal(0, 1)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Normal", dist))

  # HalfNormal
  dist <- HalfNormal(1)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("HalfNormal", dist))

  # Beta
  dist <- Beta(1, 1)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Beta", dist))

  # Gamma
  dist <- Gamma(2, 1)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Gamma", dist))

  # Uniform
  dist <- Uniform(0, 10)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Uniform", dist))

  # Binomial
  dist <- Binomial(100, 0.5)
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Binomial", dist))

  # Binomial with parameter reference
  dist <- Binomial(100, "theta")
  expect_true(grepl("theta", dist))
})

test_that("distribution parameters are correctly encoded", {
  dist <- Normal(2.5, 1.5)
  parsed <- jsonlite::fromJSON(dist)
  expect_equal(parsed$dist_type, "Normal")
  expect_equal(parsed$params$loc, 2.5)
  expect_equal(parsed$params$scale, 1.5)
})

test_that("Dirichlet distribution creates valid specification", {
  # Uniform Dirichlet
  dist <- Dirichlet(c(1, 1, 1))
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Dirichlet", dist))

  # Concentrated Dirichlet
  dist <- Dirichlet(c(10, 10, 10))
  parsed <- jsonlite::fromJSON(dist)
  expect_equal(parsed$dist_type, "Dirichlet")
  expect_equal(parsed$params$dim, 3)
})

test_that("Multinomial distribution creates valid specification", {
  # With fixed probabilities
  dist <- Multinomial(10, c(0.2, 0.3, 0.5))
  expect_s3_class(dist, "bg_distribution")
  expect_true(grepl("Multinomial", dist))

  parsed <- jsonlite::fromJSON(dist)
  expect_equal(parsed$dist_type, "Multinomial")
  expect_equal(parsed$params$n, 10)
  expect_equal(parsed$params$dim, 3)

  # With parameter reference (for hierarchical models)
  dist <- Multinomial(100, "theta")
  expect_true(grepl("theta", dist))
})
