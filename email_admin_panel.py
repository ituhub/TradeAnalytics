"""
=============================================================================
EMAIL ADMIN PANEL — Dash Integration for app.py
=============================================================================
Add this to your app.py to get an admin email panel in the sidebar/settings.

INTEGRATION STEPS:
  1. Copy email_service.py into your project root
  2. In app.py, add the import and init:
       from email_service import init_email_service, send_test_email
       init_email_service(firestore_db=db)  # after Firestore init
  3. Add the layout component (email_admin_layout) to your sidebar or settings tab
  4. Register the callbacks with: register_email_callbacks(app)
  5. Add env vars to your Cloud Run deploy command:
       SMTP_EMAIL=your-email@gmail.com
       SMTP_APP_PASSWORD=your-16-char-app-password

=============================================================================
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from email_service import (
    send_announcement,
    send_market_alert,
    send_custom_email,
    send_test_email,
    get_all_users,
    SMTP_EMAIL,
)


# =============================================================================
# LAYOUT — Admin Email Panel
# =============================================================================

def email_admin_layout():
    """Returns the email admin panel component. Add to your layout."""
    return dbc.Card(
        [
            dbc.CardHeader(
                html.Div([
                    html.Span("📧", style={"marginRight": "8px", "fontSize": "18px"}),
                    html.Span("Email Center", style={
                        "fontWeight": "600", "fontSize": "15px", "color": "#f1f5f9"
                    }),
                ], style={"display": "flex", "alignItems": "center"}),
                style={
                    "background": "linear-gradient(135deg, #1e293b, #0f172a)",
                    "borderBottom": "1px solid #334155",
                },
            ),
            dbc.CardBody([
                # --- Email Type Selector ---
                dbc.Label("Email Type", style={"color": "#94a3b8", "fontSize": "12px",
                                                "textTransform": "uppercase",
                                                "letterSpacing": "1px", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="email-type-dropdown",
                    options=[
                        {"label": "📢 Announcement / Update", "value": "announcement"},
                        {"label": "🚨 Market Alert", "value": "market_alert"},
                        {"label": "📝 Custom Email", "value": "custom"},
                        {"label": "🧪 Send Test Email", "value": "test"},
                    ],
                    value="announcement",
                    clearable=False,
                    style={"marginBottom": "16px"},
                    className="dash-dropdown-dark",
                ),

                # --- Audience Filter ---
                dbc.Label("Send To", style={"color": "#94a3b8", "fontSize": "12px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "1px", "marginBottom": "4px"}),
                dcc.Dropdown(
                    id="email-audience-dropdown",
                    options=[
                        {"label": "All Users", "value": "all"},
                        {"label": "Pro Users Only", "value": "pro"},
                        {"label": "Free Users Only", "value": "free"},
                    ],
                    value="all",
                    clearable=False,
                    style={"marginBottom": "16px"},
                    className="dash-dropdown-dark",
                ),

                # --- Subject ---
                dbc.Label("Subject", style={"color": "#94a3b8", "fontSize": "12px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "1px", "marginBottom": "4px"}),
                dbc.Input(
                    id="email-subject-input",
                    placeholder="e.g., New Feature: Walk-Forward Backtesting",
                    type="text",
                    style={
                        "backgroundColor": "#0f172a", "border": "1px solid #334155",
                        "color": "#f1f5f9", "marginBottom": "16px",
                    },
                ),

                # --- Body ---
                dbc.Label("Message", style={"color": "#94a3b8", "fontSize": "12px",
                                             "textTransform": "uppercase",
                                             "letterSpacing": "1px", "marginBottom": "4px"}),
                dbc.Textarea(
                    id="email-body-input",
                    placeholder=(
                        "Write your message here...\n\n"
                        "You can use basic HTML: <p>, <strong>, <a href='...'>, <ul>/<li>"
                    ),
                    style={
                        "backgroundColor": "#0f172a", "border": "1px solid #334155",
                        "color": "#f1f5f9", "minHeight": "140px", "marginBottom": "16px",
                        "fontFamily": "monospace", "fontSize": "13px",
                    },
                ),

                # --- Market Alert Fields (shown conditionally) ---
                html.Div(
                    id="market-alert-fields",
                    children=[
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Ticker", style={"color": "#94a3b8", "fontSize": "11px"}),
                                dbc.Input(id="alert-ticker", placeholder="AAPL",
                                          style={"backgroundColor": "#0f172a",
                                                 "border": "1px solid #334155",
                                                 "color": "#f1f5f9"}),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Signal", style={"color": "#94a3b8", "fontSize": "11px"}),
                                dcc.Dropdown(id="alert-signal",
                                             options=[{"label": "BUY", "value": "BUY"},
                                                      {"label": "SELL", "value": "SELL"},
                                                      {"label": "HOLD", "value": "HOLD"}],
                                             value="BUY", clearable=False),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Confidence %", style={"color": "#94a3b8", "fontSize": "11px"}),
                                dbc.Input(id="alert-confidence", type="number",
                                          placeholder="85.0",
                                          style={"backgroundColor": "#0f172a",
                                                 "border": "1px solid #334155",
                                                 "color": "#f1f5f9"}),
                            ], width=3),
                            dbc.Col([
                                dbc.Label("Current $", style={"color": "#94a3b8", "fontSize": "11px"}),
                                dbc.Input(id="alert-current-price", type="number",
                                          placeholder="192.30",
                                          style={"backgroundColor": "#0f172a",
                                                 "border": "1px solid #334155",
                                                 "color": "#f1f5f9"}),
                            ], width=3),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Predicted $", style={"color": "#94a3b8", "fontSize": "11px"}),
                                dbc.Input(id="alert-predicted-price", type="number",
                                          placeholder="198.50",
                                          style={"backgroundColor": "#0f172a",
                                                 "border": "1px solid #334155",
                                                 "color": "#f1f5f9"}),
                            ], width=6),
                        ], className="mb-3"),
                    ],
                    style={"display": "none"},
                ),

                # --- Test Email Field ---
                html.Div(
                    id="test-email-field",
                    children=[
                        dbc.Label("Send test to", style={"color": "#94a3b8", "fontSize": "12px"}),
                        dbc.Input(
                            id="test-email-address",
                            placeholder=SMTP_EMAIL or "your-email@gmail.com",
                            type="email",
                            style={
                                "backgroundColor": "#0f172a", "border": "1px solid #334155",
                                "color": "#f1f5f9", "marginBottom": "16px",
                            },
                        ),
                    ],
                    style={"display": "none"},
                ),

                # --- User Count Preview ---
                html.Div(id="email-user-count", style={
                    "fontSize": "13px", "color": "#64748b", "marginBottom": "16px",
                }),

                # --- Send Button ---
                dbc.Button(
                    [html.Span("📤", style={"marginRight": "6px"}), "Send Email"],
                    id="send-email-btn",
                    color="primary",
                    size="md",
                    style={
                        "width": "100%",
                        "background": "linear-gradient(135deg, #2563eb, #7c3aed)",
                        "border": "none",
                        "fontWeight": "600",
                    },
                ),

                # --- Result feedback ---
                html.Div(id="email-send-result", style={"marginTop": "12px"}),

            ], style={"backgroundColor": "#111827", "padding": "20px"}),
        ],
        style={
            "backgroundColor": "#111827",
            "border": "1px solid #1e293b",
            "borderRadius": "12px",
            "marginBottom": "16px",
        },
    )


# =============================================================================
# CALLBACKS
# =============================================================================

def register_email_callbacks(app):
    """Register all email admin callbacks. Call once in app.py."""

    @app.callback(
        [Output("market-alert-fields", "style"),
         Output("test-email-field", "style"),
         Output("email-subject-input", "style"),
         Output("email-body-input", "style"),
         Output("email-audience-dropdown", "disabled")],
        Input("email-type-dropdown", "value"),
    )
    def toggle_email_fields(email_type):
        base_input = {
            "backgroundColor": "#0f172a", "border": "1px solid #334155",
            "color": "#f1f5f9", "marginBottom": "16px",
        }
        base_textarea = {
            "backgroundColor": "#0f172a", "border": "1px solid #334155",
            "color": "#f1f5f9", "minHeight": "140px", "marginBottom": "16px",
            "fontFamily": "monospace", "fontSize": "13px",
        }
        hidden = {"display": "none"}
        shown = {"display": "block", "marginBottom": "16px"}

        if email_type == "market_alert":
            return shown, hidden, base_input, base_textarea, False
        elif email_type == "test":
            return hidden, shown, {**base_input, "display": "none"}, \
                   {**base_textarea, "display": "none"}, True
        else:
            return hidden, hidden, base_input, base_textarea, False

    @app.callback(
        Output("email-user-count", "children"),
        [Input("email-audience-dropdown", "value"),
         Input("email-type-dropdown", "value")],
    )
    def update_user_count(audience, email_type):
        if email_type == "test":
            return "Will send a test email to the specified address."
        try:
            users = get_all_users(audience)
            count = len(users)
            return f"📋 {count} user{'s' if count != 1 else ''} will receive this email."
        except Exception:
            return "Could not fetch user count."

    @app.callback(
        Output("email-send-result", "children"),
        Input("send-email-btn", "n_clicks"),
        [
            State("email-type-dropdown", "value"),
            State("email-audience-dropdown", "value"),
            State("email-subject-input", "value"),
            State("email-body-input", "value"),
            State("alert-ticker", "value"),
            State("alert-signal", "value"),
            State("alert-confidence", "value"),
            State("alert-current-price", "value"),
            State("alert-predicted-price", "value"),
            State("test-email-address", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_send_email(n_clicks, email_type, audience, subject, body,
                          ticker, signal, confidence, current_price,
                          predicted_price, test_email):
        if not n_clicks:
            return ""

        try:
            if email_type == "test":
                addr = test_email or SMTP_EMAIL
                if not addr:
                    return dbc.Alert("Enter a test email address.", color="warning")
                result = send_test_email(addr)
                if result["success"]:
                    return dbc.Alert(f"✅ Test email sent to {addr}", color="success")
                return dbc.Alert(f"❌ Failed: {result.get('error', 'Unknown')}", color="danger")

            elif email_type == "announcement":
                if not subject or not body:
                    return dbc.Alert("Subject and message are required.", color="warning")
                result = send_announcement(
                    title=subject,
                    message=body.replace("\n", "<br>"),
                    plan_filter=audience,
                )

            elif email_type == "market_alert":
                if not all([ticker, signal, confidence, current_price, predicted_price]):
                    return dbc.Alert("All market alert fields are required.", color="warning")
                result = send_market_alert(
                    ticker=ticker,
                    signal=signal,
                    confidence=float(confidence),
                    predicted_price=float(predicted_price),
                    current_price=float(current_price),
                    details=body.replace("\n", "<br>") if body else "",
                    plan_filter=audience,
                )

            elif email_type == "custom":
                if not subject or not body:
                    return dbc.Alert("Subject and body are required.", color="warning")
                result = send_custom_email(
                    subject=subject,
                    body_html=body.replace("\n", "<br>"),
                    plan_filter=audience,
                )
            else:
                return dbc.Alert("Unknown email type.", color="warning")

            # Show result
            sent = result.get("sent", 0)
            failed = result.get("failed", 0)
            total = result.get("total", 0)

            if failed == 0 and sent > 0:
                return dbc.Alert(f"✅ Sent to {sent}/{total} users.", color="success")
            elif sent > 0:
                return dbc.Alert(
                    f"⚠️ Sent {sent}/{total}, {failed} failed.",
                    color="warning",
                )
            else:
                errors = result.get("errors", [])
                err_msg = errors[0].get("error", "Unknown") if errors else "No users found"
                return dbc.Alert(f"❌ Failed: {err_msg}", color="danger")

        except Exception as e:
            return dbc.Alert(f"❌ Error: {str(e)}", color="danger")
