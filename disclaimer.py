"""
DISCLAIMER MODAL — MarketLens AI
============================================
Shows a legal disclaimer overlay after user signs in.
User must accept to proceed. Acceptance is stored in Firestore user doc.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


DISCLAIMER_TEXT = """
<h3 style="color:#e2e8f0; margin:0 0 16px; font-weight:700;">Terms of Use & Disclaimer</h3>

<p style="margin-bottom:12px;"><strong style="color:#f59e0b;">⚠️ NOT FINANCIAL ADVICE</strong></p>

<p>By using MarketLens AI ("the Platform"), you acknowledge and agree to the following:</p>

<p><strong style="color:#94a3b8;">1. No Investment Advice.</strong> The Platform provides AI-generated market analysis and predictions for <em>educational and informational purposes only</em>. Nothing on this Platform constitutes investment advice, financial advice, trading advice, or any other sort of advice. You should not treat any of the Platform's content as such.</p>

<p><strong style="color:#94a3b8;">2. No Guarantees.</strong> AI predictions, signals, and market analyses are based on historical data and machine learning models. Past performance does not guarantee future results. Markets are inherently unpredictable and all trading involves substantial risk of loss.</p>

<p><strong style="color:#94a3b8;">3. Risk of Loss.</strong> You could lose some or all of your invested capital. Never trade with money you cannot afford to lose. The use of leverage or margin can amplify both gains and losses.</p>

<p><strong style="color:#94a3b8;">4. Your Responsibility.</strong> You are solely responsible for your own trading and investment decisions. You should conduct your own research, consult qualified financial professionals, and consider your own financial situation and risk tolerance before making any trading decisions.</p>

<p><strong style="color:#94a3b8;">5. Data Accuracy.</strong> While we strive to provide accurate market data and AI analysis, we do not guarantee the accuracy, completeness, or timeliness of any information provided. Data may be delayed, incomplete, or contain errors.</p>

<p><strong style="color:#94a3b8;">6. Limitation of Liability.</strong> The Platform, its creators, affiliates, and service providers shall not be liable for any losses, damages, or costs arising from your use of the Platform or reliance on its content.</p>

<p><strong style="color:#94a3b8;">7. Privacy.</strong> Your email, account data, and usage metrics are stored securely. We do not sell your personal data to third parties. Payment processing is handled securely by Stripe.</p>

<p><strong style="color:#94a3b8;">8. Subscription Terms.</strong> Paid plans are billed according to your chosen billing cycle. You may cancel at any time. Refund eligibility is subject to our 14-day money-back guarantee.</p>

<p style="color:#64748b; font-size:12px; margin-top:16px;">By clicking "I Accept", you confirm that you have read, understood, and agree to these terms. Last updated: April 2026.</p>
"""


def build_disclaimer_overlay():
    """
    Build the disclaimer modal overlay.
    This should be inserted into the main dashboard layout.
    It will be shown/hidden via the disclaimer-accepted store.
    """
    return html.Div(
        id="disclaimer-overlay",
        children=[
            html.Div([
                # Logo
                html.Div([
                    html.Div("⚡", style={"fontSize": "28px"}),
                ], style={
                    "background": "linear-gradient(135deg, #06b6d4, #8b5cf6)",
                    "width": "56px", "height": "56px", "borderRadius": "14px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "margin": "0 auto 16px",
                }),

                # Scrollable content
                html.Div(
                    id="disclaimer-content",
                    children=dcc.Markdown(
                        DISCLAIMER_TEXT,
                        dangerously_allow_html=True,
                        style={"color": "#cbd5e1", "fontSize": "13px", "lineHeight": "1.7"},
                    ),
                    style={
                        "maxHeight": "55vh", "overflowY": "auto", "padding": "0 8px",
                        "marginBottom": "24px",
                        # Custom scrollbar
                        "scrollbarWidth": "thin",
                        "scrollbarColor": "#334155 transparent",
                    },
                ),

                # Accept / Decline buttons
                html.Div([
                    html.Button(
                        "✅ I Accept — Continue to Platform",
                        id="disclaimer-accept-btn",
                        n_clicks=0,
                        style={
                            "flex": "1", "padding": "14px", "borderRadius": "10px",
                            "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                            "color": "#fff", "border": "none", "fontSize": "14px",
                            "fontWeight": "700", "cursor": "pointer",
                            "boxShadow": "0 4px 20px rgba(99,102,241,0.3)",
                        },
                    ),
                    html.Button(
                        "Decline",
                        id="disclaimer-decline-btn",
                        n_clicks=0,
                        style={
                            "padding": "14px 24px", "borderRadius": "10px",
                            "background": "rgba(239,68,68,0.08)", "color": "#f87171",
                            "border": "1px solid rgba(239,68,68,0.2)", "fontSize": "14px",
                            "fontWeight": "600", "cursor": "pointer",
                        },
                    ),
                ], style={"display": "flex", "gap": "12px"}),

            ], style={
                "maxWidth": "640px", "width": "90%", "padding": "36px 32px",
                "background": "rgba(15,23,42,0.95)", "backdropFilter": "blur(20px)",
                "border": "1px solid rgba(99,102,241,0.15)", "borderRadius": "20px",
                "boxShadow": "0 25px 80px rgba(0,0,0,0.6)",
                "maxHeight": "90vh", "overflow": "hidden",
            }),
        ],
        style={
            "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
            "background": "rgba(0,0,0,0.7)", "backdropFilter": "blur(8px)",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "zIndex": "9999",
        },
    )
