"""
APP GUIDE PAGE — AI Trading Professional
==========================================
Interactive guide explaining platform features, AI models, 
example prediction analysis, and usage instructions.
"""

from dash import html, dcc
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


def build_app_guide_page():
    """Build the complete app guide page."""
    return html.Div([
        _section("🚀", "Welcome to AI Trading Professional",
            html.Div([
                html.P(
                    "AI Trading Professional is a production-grade platform combining "
                    "8 advanced deep learning models with real-time market data to generate "
                    "actionable trading signals across stocks, forex, crypto, commodities, and indices.",
                    style={"color": "#cbd5e1", "fontSize": "15px", "lineHeight": "1.8", "marginBottom": "16px"},
                ),
                html.Div([
                    _badge("📊", "36 Assets", "Stocks, Forex, Crypto, Commodities"),
                    _badge("🤖", "8 AI Models", "Deep Learning Ensemble"),
                    _badge("⏱️", "4 Timeframes", "15m, 1H, 4H, Daily"),
                    _badge("🧪", "Walk-Forward", "Out-of-sample validated"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))", "gap": "12px"}),
            ]),
        ),

        _section("⚙️", "How It Works",
            html.Div([
                _step("1", "Select Asset & Timeframe",
                    "Choose from 36 assets across 5 categories (Crypto, Forex, Commodities, Indices, Stocks). "
                    "Pick your timeframe and optionally enable multi-timeframe consensus."),
                _step("2", "AI Models Analyze",
                    "8 deep learning models each independently analyze 150+ engineered features from live FMP market data. "
                    "Each model brings a different architectural strength."),
                _step("3", "Ensemble Prediction",
                    "Individual predictions are combined via inverse-MSE weighted averaging. "
                    "Models with lower historical error get higher weight — institutional-grade methodology."),
                _step("4", "Signal & Strategy",
                    "You receive: predicted price, BUY/HOLD/SELL signal, confidence score, "
                    "stop loss/take profit levels, Kelly Criterion position sizing, 5-day forecast, and risk assessment."),
            ]),
        ),

        _section("🧠", "The 8 AI Models",
            html.Div([
                _model("Advanced Transformer", "Self-attention for long-range temporal dependencies.", "#6366f1"),
                _model("CNN-LSTM Hybrid", "Spatial pattern extraction + sequential memory.", "#06b6d4"),
                _model("Temporal Conv. Network", "Dilated causal convolutions across timescales.", "#8b5cf6"),
                _model("Informer", "Sparse attention optimized for long sequence forecasting.", "#10b981"),
                _model("N-BEATS", "Neural basis expansion — trend + seasonal decomposition.", "#f59e0b"),
                _model("LSTM-GRU Ensemble", "Dual recurrent architecture for regime transitions.", "#ef4444"),
                _model("XGBoost", "Gradient-boosted trees on 150+ engineered features.", "#14b8a6"),
                _model("Sklearn Ensemble", "Meta-ensemble: Random Forest + Extra Trees + Ridge.", "#a855f7"),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))", "gap": "12px"}),
        ),

        _section("📈", "Example: BTC Prediction Analysis",
            html.Div([
                html.P("Here's what a typical prediction analysis looks like:",
                       style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "16px"}),
                dcc.Graph(figure=_example_chart(), config={"displayModeBar": False}),
                html.Div([
                    _metric("Signal", "BUY", "#10b981"),
                    _metric("Confidence", "87.3%", "#6366f1"),
                    _metric("Current", "$68,450", "#e2e8f0"),
                    _metric("Predicted", "$71,200", "#10b981"),
                    _metric("Stop Loss", "$66,100", "#ef4444"),
                    _metric("Take Profit", "$73,800", "#10b981"),
                ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(140px, 1fr))",
                          "gap": "10px", "marginTop": "16px"}),
                html.Div([
                    html.P("Model Consensus:", style={"color": "#94a3b8", "fontSize": "12px",
                                                       "fontWeight": "600", "marginBottom": "8px"}),
                    _bar("Transformer", 72100, "#6366f1"),
                    _bar("CNN-LSTM", 71800, "#06b6d4"),
                    _bar("TCN", 70500, "#8b5cf6"),
                    _bar("XGBoost", 71400, "#14b8a6"),
                    _bar("N-BEATS", 70900, "#f59e0b"),
                ], style={"marginTop": "20px", "padding": "16px",
                          "background": "rgba(15,23,42,0.4)", "borderRadius": "10px",
                          "border": "1px solid rgba(99,102,241,0.08)"}),
            ]),
        ),

        _section("💎", "Platform Features",
            html.Div([
                _feat("🤖 AI Prediction Engine",
                    "Real-time predictions with 8-model ensemble, confidence scoring, and BUY/HOLD/SELL signals."),
                _feat("📊 Advanced Analytics",
                    "Market regime detection (GMM), model drift monitoring (PSI/KS), SHAP explainability, "
                    "and alternative data (economic indicators, sentiment)."),
                _feat("📈 Backtesting Engine",
                    "Walk-forward validation with quarter-Kelly position sizing. "
                    "Tests strategy on historical data before risking real capital."),
                _feat("💼 Portfolio Optimization",
                    "Black-Litterman portfolio construction with AI views. Monte Carlo simulation and efficient frontier."),
                _feat("🏦 FTMO Dashboard",
                    "FTMO challenge tracker — daily drawdown, max loss limits, profit targets in real-time."),
                _feat("🧠 Model Training",
                    "Train and fine-tune AI models on latest data. Models persisted to Google Cloud Storage."),
                _feat("⏱️ Multi-Timeframe Analysis",
                    "Consensus across 15m, 1H, 4H, Daily. Higher conviction when multiple timeframes agree."),
                _feat("⚠️ Risk Assessment",
                    "VaR (95%), Sharpe/Sortino ratios, max drawdown, beta, and custom risk scoring by asset class."),
            ]),
        ),

        _section("💰", "Subscription Plans",
            html.Div([
                _plan("Free", "$0", ["3 tickers", "2 preds/day", "Daily TF", "3 models"], "#64748b"),
                _plan("Starter", "$49/mo", ["5 tickers", "5 preds/day", "Daily TF", "3 models"], "#06b6d4"),
                _plan("Professional", "$129/mo", ["All tickers", "50 preds/day", "All TFs", "8 models",
                                                    "Backtesting", "SHAP", "FTMO"], "#8b5cf6"),
                _plan("Institutional", "$349/mo", ["Everything in Pro", "Unlimited preds", "Portfolio optimizer",
                                                    "API access", "Dedicated support"], "#f59e0b"),
            ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "12px"}),
        ),

        _section("💡", "Pro Tips",
            html.Div([
                _tip("Use multi-timeframe analysis — when 15m, 1H, and 1D all agree, the signal is much stronger."),
                _tip("Signals below 60% confidence are low conviction — reduce position size or skip."),
                _tip("Run backtests before live trading any strategy. Walk-forward validation prevents overfitting."),
                _tip("Check the Risk Assessment tab before every trade. High VaR means smaller positions."),
                _tip("The position sizing calculator uses 2% fixed risk per trade. Adjust based on your tolerance."),
            ]),
        ),
    ], style={"maxWidth": "900px"})


# ── HELPERS ──────────────────────────────────────────────────────

def _section(icon, title, content):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "20px"}),
            html.H2(title, style={"color": "#e2e8f0", "fontSize": "1.2rem", "fontWeight": "700", "margin": "0"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "16px"}),
        content,
    ], style={"padding": "24px", "marginBottom": "20px",
              "background": "rgba(15,23,42,0.5)", "borderRadius": "14px",
              "border": "1px solid rgba(99,102,241,0.08)"})

def _badge(icon, title, sub):
    return html.Div([
        html.Div(icon, style={"fontSize": "22px", "marginBottom": "6px"}),
        html.Div(title, style={"color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700"}),
        html.Div(sub, style={"color": "#64748b", "fontSize": "11px"}),
    ], style={"padding": "16px", "textAlign": "center", "borderRadius": "10px",
              "background": "rgba(99,102,241,0.06)", "border": "1px solid rgba(99,102,241,0.1)"})

def _step(num, title, desc):
    return html.Div([
        html.Div([
            html.Div(num, style={"width": "32px", "height": "32px", "borderRadius": "50%",
                                  "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                                  "display": "flex", "alignItems": "center", "justifyContent": "center",
                                  "color": "#fff", "fontSize": "14px", "fontWeight": "700", "flexShrink": "0"}),
            html.Div([
                html.Div(title, style={"color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700"}),
                html.Div(desc, style={"color": "#94a3b8", "fontSize": "13px", "lineHeight": "1.6", "marginTop": "4px"}),
            ]),
        ], style={"display": "flex", "gap": "14px", "alignItems": "flex-start"}),
    ], style={"padding": "16px", "marginBottom": "10px",
              "background": "rgba(99,102,241,0.04)", "borderRadius": "10px",
              "border": "1px solid rgba(99,102,241,0.06)"})

def _model(name, desc, color):
    return html.Div([
        html.Div(name, style={"color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700", "marginBottom": "6px"}),
        html.Div(desc, style={"color": "#94a3b8", "fontSize": "12px", "lineHeight": "1.5"}),
    ], style={"padding": "16px", "borderRadius": "10px", "background": "rgba(15,23,42,0.4)",
              "border": f"1px solid {color}22", "borderLeft": f"3px solid {color}"})

def _metric(label, value, color):
    return html.Div([
        html.Div(label, style={"color": "#64748b", "fontSize": "10px", "textTransform": "uppercase",
                                "letterSpacing": "1px", "marginBottom": "4px"}),
        html.Div(value, style={"color": color, "fontSize": "18px", "fontWeight": "700"}),
    ], style={"padding": "12px", "textAlign": "center", "borderRadius": "8px",
              "background": "rgba(15,23,42,0.4)", "border": "1px solid rgba(99,102,241,0.08)"})

def _bar(model_name, price, color):
    w = max(20, min(100, ((price - 68000) / 6000) * 100))
    return html.Div([
        html.Span(model_name, style={"color": "#94a3b8", "fontSize": "11px", "width": "120px", "flexShrink": "0"}),
        html.Div([
            html.Div(style={"width": f"{w}%", "height": "6px", "borderRadius": "3px", "background": color}),
        ], style={"flex": "1", "height": "6px", "background": "rgba(99,102,241,0.08)", "borderRadius": "3px"}),
        html.Span(f"${price:,.0f}", style={"color": color, "fontSize": "11px", "fontWeight": "600",
                                              "width": "70px", "textAlign": "right", "flexShrink": "0"}),
    ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "6px"})

def _feat(title, desc):
    return html.Div([
        html.Div(title, style={"color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700", "marginBottom": "4px"}),
        html.Div(desc, style={"color": "#94a3b8", "fontSize": "13px", "lineHeight": "1.6"}),
    ], style={"padding": "14px 16px", "marginBottom": "8px", "borderRadius": "8px",
              "background": "rgba(99,102,241,0.04)", "borderLeft": "3px solid rgba(99,102,241,0.2)"})

def _plan(name, price, features, color):
    return html.Div([
        html.Div(name, style={"color": color, "fontSize": "16px", "fontWeight": "700", "marginBottom": "4px"}),
        html.Div(price, style={"color": "#e2e8f0", "fontSize": "22px", "fontWeight": "800", "marginBottom": "12px"}),
        *[html.Div([
            html.Span("✓ ", style={"color": color}),
            html.Span(f, style={"color": "#94a3b8", "fontSize": "12px"}),
        ], style={"marginBottom": "4px"}) for f in features],
    ], style={"padding": "20px", "borderRadius": "10px", "background": "rgba(15,23,42,0.4)",
              "border": f"1px solid {color}33"})

def _tip(text):
    return html.Div([
        html.Span("💡 ", style={"flexShrink": "0"}),
        html.Span(text, style={"color": "#94a3b8", "fontSize": "13px", "lineHeight": "1.6"}),
    ], style={"display": "flex", "gap": "8px", "padding": "10px 14px", "marginBottom": "6px",
              "background": "rgba(245,158,11,0.04)", "borderRadius": "8px",
              "border": "1px solid rgba(245,158,11,0.08)"})

def _example_chart():
    np.random.seed(42)
    dates = [datetime(2026, 3, 1) + timedelta(days=i) for i in range(30)]
    prices = [64000]
    for _ in range(29):
        prices.append(prices[-1] * (1 + np.random.normal(0.003, 0.02)))
    pred_dates = [dates[-1] + timedelta(days=i) for i in range(1, 6)]
    pred_prices = [prices[-1] * (1 + np.random.normal(0.005, 0.008)) for _ in range(5)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Historical",
                              line=dict(color="#6366f1", width=2)))
    fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode="lines+markers", name="AI Forecast",
                              line=dict(color="#10b981", width=2, dash="dot"),
                              marker=dict(size=6, color="#10b981")))
    fig.add_hline(y=prices[-1], line_dash="dash", line_color="#f59e0b", opacity=0.5,
                  annotation_text="Current Price")
    fig.update_layout(
        template="plotly_dark", height=320,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="rgba(99,102,241,0.06)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.06)", title="Price ($)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
