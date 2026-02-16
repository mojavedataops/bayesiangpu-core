"""
Dash app for Bayesian inference with BayesianGPU

Run with: python dash_app.py

Requirements: pip install dash bayesiangpu
"""

from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from bayesiangpu import Model, Beta, Binomial, sample

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("BayesianGPU - Beta-Binomial Demo", className="text-center my-4"),
            html.P(
                "Estimate the probability of success using Bayesian inference",
                className="text-center text-muted"
            ),
        ])
    ]),

    dbc.Row([
        # Left column - inputs
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prior Parameters"),
                dbc.CardBody([
                    html.Label("Alpha (prior successes + 1)"),
                    dcc.Slider(id="alpha", min=0.1, max=10, value=1, step=0.1,
                              marks={i: str(i) for i in range(0, 11, 2)}),
                    html.Br(),
                    html.Label("Beta (prior failures + 1)"),
                    dcc.Slider(id="beta", min=0.1, max=10, value=1, step=0.1,
                              marks={i: str(i) for i in range(0, 11, 2)}),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Observed Data"),
                dbc.CardBody([
                    html.Label("Number of trials"),
                    dbc.Input(id="n-trials", type="number", value=100, min=1, max=1000),
                    html.Br(),
                    html.Label("Number of successes"),
                    dbc.Input(id="successes", type="number", value=65, min=0, max=1000),
                ])
            ], className="mb-3"),

            dbc.Card([
                dbc.CardHeader("Sampling Parameters"),
                dbc.CardBody([
                    html.Label("Samples per chain"),
                    dcc.Slider(id="num-samples", min=100, max=2000, value=500, step=100,
                              marks={i: str(i) for i in [100, 500, 1000, 1500, 2000]}),
                    html.Br(),
                    html.Label("Number of chains"),
                    dcc.Slider(id="num-chains", min=1, max=8, value=4, step=1,
                              marks={i: str(i) for i in range(1, 9)}),
                ])
            ], className="mb-3"),

            dbc.Button("Run Inference", id="run-btn", color="primary", className="w-100"),
        ], width=4),

        # Right column - results
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Results"),
                dbc.CardBody([
                    dcc.Loading(
                        id="loading",
                        children=[html.Div(id="results")],
                        type="default",
                    )
                ])
            ]),
        ], width=8),
    ]),

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P(
                "Built with BayesianGPU - GPU-accelerated Bayesian inference",
                className="text-center text-muted"
            ),
        ])
    ]),
], fluid=True)


@callback(
    Output("results", "children"),
    Input("run-btn", "n_clicks"),
    State("alpha", "value"),
    State("beta", "value"),
    State("n-trials", "value"),
    State("successes", "value"),
    State("num-samples", "value"),
    State("num-chains", "value"),
    prevent_initial_call=True,
)
def run_inference(n_clicks, alpha, beta_param, n_trials, successes, num_samples, num_chains):
    if n_clicks is None:
        return html.P("Click 'Run Inference' to start")

    # Validate inputs
    if successes > n_trials:
        return dbc.Alert("Successes cannot exceed number of trials!", color="danger")

    # Build and run model
    model = Model()
    model.param("theta", Beta(alpha, beta_param))
    model.observe(Binomial(n_trials, "theta"), [float(successes)])

    result = sample(model, num_samples=num_samples, num_chains=num_chains, seed=42)

    # Get summary
    summary = result.summarize("theta")
    diagnostics = result.diagnostics

    # Build result cards
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Posterior Mean", className="card-title"),
                        html.H2(f"{summary.mean:.4f}", className="text-primary"),
                    ])
                ], className="text-center"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("95% Credible Interval", className="card-title"),
                        html.H2(f"[{summary.q025:.4f}, {summary.q975:.4f}]", className="text-success"),
                    ])
                ], className="text-center"),
            ], width=4),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Standard Deviation", className="card-title"),
                        html.H2(f"{summary.std:.4f}", className="text-info"),
                    ])
                ], className="text-center"),
            ], width=4),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Diagnostics"),
                        html.P([
                            html.Strong("R-hat: "),
                            html.Span(
                                f"{summary.rhat:.4f}",
                                className="text-success" if summary.rhat < 1.01 else "text-warning"
                            ),
                            " ",
                            dbc.Badge("Good" if summary.rhat < 1.01 else "Warning",
                                     color="success" if summary.rhat < 1.01 else "warning"),
                        ]),
                        html.P([
                            html.Strong("ESS: "),
                            html.Span(
                                f"{summary.ess:.0f}",
                                className="text-success" if summary.ess > 400 else "text-warning"
                            ),
                            " ",
                            dbc.Badge("Good" if summary.ess > 400 else "Low",
                                     color="success" if summary.ess > 400 else "warning"),
                        ]),
                        html.P([
                            html.Strong("Divergences: "),
                            html.Span(str(diagnostics.divergences)),
                        ]),
                    ])
                ]),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Full Summary"),
                        html.Pre(result.format_summary(), style={"fontSize": "12px"}),
                    ])
                ]),
            ], width=6),
        ]),

        # Warnings
        html.Div([
            dbc.Alert(
                [html.Strong("Warnings: ")] + [html.Li(w) for w in result.warnings()],
                color="warning",
                className="mt-3"
            )
        ] if result.warnings() else []),
    ])


if __name__ == "__main__":
    app.run(debug=True)
