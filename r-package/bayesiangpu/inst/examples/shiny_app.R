#' Shiny App for Bayesian Inference with BayesianGPU
#'
#' Run with: shiny::runApp("shiny_app.R")
#'
#' Requirements: install.packages(c("shiny", "bayesiangpu", "ggplot2"))

library(shiny)
library(bayesiangpu)
library(ggplot2)

ui <- fluidPage(
  titlePanel("BayesianGPU - Bayesian Inference Demo"),

  sidebarLayout(
    sidebarPanel(
      selectInput("model_type", "Select Model",
                  choices = c("Beta-Binomial (Proportion)",
                              "Normal (Mean Estimation)")),

      hr(),
      h4("Sampling Parameters"),
      sliderInput("num_samples", "Samples per chain",
                  min = 100, max = 2000, value = 500, step = 100),
      sliderInput("num_chains", "Number of chains",
                  min = 1, max = 8, value = 4, step = 1),
      numericInput("seed", "Random seed", value = 42, min = 1),

      hr(),
      conditionalPanel(
        condition = "input.model_type == 'Beta-Binomial (Proportion)'",
        h4("Prior Parameters"),
        sliderInput("alpha", "Alpha (prior successes + 1)",
                    min = 0.1, max = 10, value = 1, step = 0.1),
        sliderInput("beta_param", "Beta (prior failures + 1)",
                    min = 0.1, max = 10, value = 1, step = 0.1),

        h4("Observed Data"),
        numericInput("n_trials", "Number of trials", value = 100, min = 1, max = 1000),
        numericInput("successes", "Number of successes", value = 65, min = 0, max = 1000)
      ),

      conditionalPanel(
        condition = "input.model_type == 'Normal (Mean Estimation)'",
        h4("Prior Parameters"),
        sliderInput("prior_mean", "Prior mean (mu)", min = -10, max = 10, value = 0),
        sliderInput("prior_std", "Prior std for mu", min = 0.1, max = 20, value = 10),
        sliderInput("sigma_scale", "HalfNormal scale for sigma",
                    min = 0.1, max = 10, value = 5),

        h4("Observed Data"),
        textAreaInput("data_input", "Enter data (comma-separated)",
                      value = "2.3, 2.1, 2.5, 2.4, 2.2, 2.6, 2.3, 2.4, 2.5, 2.2")
      ),

      hr(),
      actionButton("run_btn", "Run Inference", class = "btn-primary btn-block")
    ),

    mainPanel(
      tabsetPanel(
        tabPanel("Summary",
                 h3("Posterior Summary"),
                 verbatimTextOutput("summary_text"),
                 hr(),
                 h4("Diagnostics"),
                 verbatimTextOutput("diagnostics_text"),
                 uiOutput("warnings_ui")
        ),
        tabPanel("Posterior Plot",
                 plotOutput("posterior_plot", height = "400px")
        ),
        tabPanel("Trace Plot",
                 plotOutput("trace_plot", height = "400px")
        )
      )
    )
  )
)

server <- function(input, output, session) {
  result <- reactiveVal(NULL)

  observeEvent(input$run_btn, {
    if (input$model_type == "Beta-Binomial (Proportion)") {
      if (input$successes > input$n_trials) {
        showNotification("Successes cannot exceed number of trials!",
                         type = "error")
        return()
      }

      model <- Model() |>
        param("theta", Beta(input$alpha, input$beta_param)) |>
        observe(Binomial(input$n_trials, "theta"), input$successes)
    } else {
      # Parse data
      data <- tryCatch({
        as.numeric(strsplit(input$data_input, ",")[[1]])
      }, error = function(e) NULL)

      if (is.null(data) || any(is.na(data))) {
        showNotification("Invalid data format. Please use comma-separated numbers.",
                         type = "error")
        return()
      }

      model <- Model() |>
        param("mu", Normal(input$prior_mean, input$prior_std)) |>
        param("sigma", HalfNormal(input$sigma_scale)) |>
        observe(Normal("mu", "sigma"), data)
    }

    withProgress(message = "Running MCMC sampling...", {
      res <- bg_sample(model,
                       num_samples = input$num_samples,
                       num_chains = input$num_chains,
                       seed = input$seed)
      result(res)
    })
  })

  output$summary_text <- renderPrint({
    req(result())
    result()$summary()
  })

  output$diagnostics_text <- renderPrint({
    req(result())
    res <- result()
    cat(sprintf("Divergences: %d\n", res$divergences))
    cat(sprintf("Final step size: %.4f\n", res$step_size))
    cat(sprintf("Converged: %s\n", if (res$is_converged()) "Yes" else "No"))
  })

  output$warnings_ui <- renderUI({
    req(result())
    warns <- result()$warnings()
    if (length(warns) > 0) {
      div(
        class = "alert alert-warning",
        h5("Warnings:"),
        tags$ul(lapply(warns, tags$li))
      )
    }
  })

  output$posterior_plot <- renderPlot({
    req(result())
    res <- result()

    # Get first parameter for plotting
    param_name <- res$param_names[1]
    samples <- res$samples[[param_name]]

    df <- data.frame(x = samples)
    ggplot(df, aes(x = x)) +
      geom_density(fill = "steelblue", alpha = 0.7) +
      labs(title = paste("Posterior Distribution:", param_name),
           x = param_name, y = "Density") +
      theme_minimal() +
      theme(text = element_text(size = 14))
  })

  output$trace_plot <- renderPlot({
    req(result())
    res <- result()

    # Get first parameter for plotting
    param_name <- res$param_names[1]
    chains <- res$chains[[param_name]]

    # Create data frame with chain labels
    chain_dfs <- lapply(seq_along(chains), function(i) {
      data.frame(
        iteration = seq_along(chains[[i]]),
        value = chains[[i]],
        chain = factor(i)
      )
    })
    df <- do.call(rbind, chain_dfs)

    ggplot(df, aes(x = iteration, y = value, color = chain)) +
      geom_line(alpha = 0.7) +
      labs(title = paste("Trace Plot:", param_name),
           x = "Iteration", y = param_name) +
      theme_minimal() +
      theme(text = element_text(size = 14),
            legend.position = "bottom")
  })
}

shinyApp(ui, server)
