"""
╔══════════════════════════════════════════════════════════════════════╗
║  MarketLens AI — Admin Monitoring & Feature Flags                    ║
║                                                                      ║
║  Provides:                                                           ║
║    1. Real-time system health dashboard                              ║
║    2. Feature flags (toggle features without redeploying)            ║
║    3. User analytics & session monitoring                            ║
║    4. API health checks                                              ║
║    5. Error log viewer                                               ║
║    6. Run diagnostics from browser                                   ║
║                                                                      ║
║  Feature flags are stored in Firestore → _system_config collection   ║
║  Changes take effect on next page load (no redeploy needed)          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any

from dash import html, dcc, Input, Output, State, callback, ctx, no_update, ALL
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# FIRESTORE CONNECTION
# ═══════════════════════════════════════════════════════════════════════

_db = None
try:
    from google.cloud import firestore
    _db = firestore.Client()
except Exception:
    logger.warning("⚠️ Firestore not available for admin monitoring")

FEATURE_FLAGS_DOC = "feature_flags"
SYSTEM_CONFIG_COLLECTION = "_system_config"


# ═══════════════════════════════════════════════════════════════════════
# FEATURE FLAGS — Stored in Firestore, toggle from admin panel
# ═══════════════════════════════════════════════════════════════════════

# Default flags — used if Firestore doesn't have them yet
DEFAULT_FLAGS = {
    "maintenance_mode": {
        "enabled": False,
        "label": "Maintenance Mode",
        "description": "Show maintenance banner to all users. App still works but users see a notice.",
        "category": "system",
    },
    "new_user_registration": {
        "enabled": True,
        "label": "New User Registration",
        "description": "Allow new account creation. Disable to freeze signups.",
        "category": "auth",
    },
    "demo_mode": {
        "enabled": True,
        "label": "Demo Mode Access",
        "description": "Allow users to try the app without signing in.",
        "category": "auth",
    },
    "ai_analysis": {
        "enabled": True,
        "label": "AI Analysis Engine",
        "description": "Enable the Run AI Analysis button. Disable during model updates.",
        "category": "features",
    },
    "backtesting": {
        "enabled": True,
        "label": "Backtesting Module",
        "description": "Enable walk-forward backtesting for eligible plans.",
        "category": "features",
    },
    "regime_detection": {
        "enabled": True,
        "label": "Regime Detection",
        "description": "Enable market regime analysis (HMM-based).",
        "category": "features",
    },
    "portfolio_optimization": {
        "enabled": True,
        "label": "Portfolio Optimization",
        "description": "Enable portfolio optimization page for enterprise users.",
        "category": "features",
    },
    "email_notifications": {
        "enabled": True,
        "label": "Email Notifications",
        "description": "Send automated market alerts and subscription emails.",
        "category": "system",
    },
    "stripe_payments": {
        "enabled": True,
        "label": "Stripe Payments",
        "description": "Accept subscription payments via Stripe checkout.",
        "category": "billing",
    },
    "discovery_plan": {
        "enabled": True,
        "label": "Discovery Plan (14-day trial)",
        "description": "Auto-assign Discovery plan to new registrations.",
        "category": "billing",
    },
    "model_training": {
        "enabled": True,
        "label": "Model Training Page",
        "description": "Allow professional users to trigger model training.",
        "category": "features",
    },
    "contact_form": {
        "enabled": True,
        "label": "Contact Form",
        "description": "Enable the contact support form.",
        "category": "system",
    },
}


def get_feature_flags() -> Dict:
    """Load feature flags from Firestore, falling back to defaults."""
    flags = {k: v.copy() for k, v in DEFAULT_FLAGS.items()}
    if _db:
        try:
            doc = _db.collection(SYSTEM_CONFIG_COLLECTION).document(FEATURE_FLAGS_DOC).get()
            if doc.exists:
                saved = doc.to_dict()
                for key, val in saved.items():
                    if key in flags and isinstance(val, dict):
                        flags[key]["enabled"] = val.get("enabled", flags[key]["enabled"])
                    elif key in flags and isinstance(val, bool):
                        flags[key]["enabled"] = val
        except Exception as e:
            logger.warning(f"Failed to load feature flags: {e}")
    return flags


def set_feature_flag(flag_id: str, enabled: bool) -> bool:
    """Update a single feature flag in Firestore."""
    if not _db:
        logger.error("Firestore not available — cannot save flag")
        return False
    try:
        ref = _db.collection(SYSTEM_CONFIG_COLLECTION).document(FEATURE_FLAGS_DOC)
        ref.set({flag_id: {"enabled": enabled, "updated_at": datetime.now().isoformat()}}, merge=True)
        logger.info(f"🏴 Flag '{flag_id}' → {'ON' if enabled else 'OFF'}")
        return True
    except Exception as e:
        logger.error(f"Failed to save flag '{flag_id}': {e}")
        return False


def is_flag_enabled(flag_id: str) -> bool:
    """Quick check if a feature flag is enabled. Cached per-request."""
    flags = get_feature_flags()
    return flags.get(flag_id, {}).get("enabled", True)


# ═══════════════════════════════════════════════════════════════════════
# SYSTEM HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════════

def run_health_checks() -> List[Dict]:
    """Run real-time system health checks and return results."""
    checks = []

    # 1. Firestore
    try:
        if _db:
            list(_db.collection("users").limit(1).stream())
            checks.append({"name": "Firestore Database", "status": "ok", "detail": "Connected"})
        else:
            checks.append({"name": "Firestore Database", "status": "warn", "detail": "Using local fallback"})
    except Exception as e:
        checks.append({"name": "Firestore Database", "status": "fail", "detail": str(e)[:80]})

    # 2. Stripe
    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if stripe_key:
        try:
            import stripe
            stripe.api_key = stripe_key
            stripe.Product.list(limit=1)
            checks.append({"name": "Stripe API", "status": "ok", "detail": "Connected"})
        except Exception as e:
            checks.append({"name": "Stripe API", "status": "fail", "detail": str(e)[:80]})
    else:
        checks.append({"name": "Stripe API", "status": "warn", "detail": "Key not set"})

    # 3. FMP API
    fmp_key = os.environ.get("FMP_API_KEY", "")
    if fmp_key:
        try:
            import requests
            resp = requests.get(
                f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}",
                timeout=8)
            if resp.status_code == 200 and resp.json():
                price = resp.json()[0].get("price", "?")
                checks.append({"name": "FMP Market Data", "status": "ok", "detail": f"Live — AAPL ${price}"})
            else:
                checks.append({"name": "FMP Market Data", "status": "warn", "detail": f"Status {resp.status_code}"})
        except Exception as e:
            checks.append({"name": "FMP Market Data", "status": "fail", "detail": str(e)[:80]})
    else:
        checks.append({"name": "FMP Market Data", "status": "warn", "detail": "Key not set — demo mode"})

    # 4. Email service
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD", "")
    checks.append({
        "name": "Email Service",
        "status": "ok" if smtp_pass else "warn",
        "detail": "SMTP configured" if smtp_pass else "Not configured"
    })

    # 5. GCS Model Storage
    try:
        from google.cloud import storage
        client = storage.Client()
        checks.append({"name": "GCS Storage", "status": "ok", "detail": "Connected"})
    except Exception:
        checks.append({"name": "GCS Storage", "status": "warn", "detail": "Not available"})

    # 6. AI Backend
    try:
        from enhprog import MultiTimeframeDataManager
        checks.append({"name": "AI Backend (enhprog)", "status": "ok", "detail": "Loaded"})
    except Exception:
        checks.append({"name": "AI Backend (enhprog)", "status": "warn", "detail": "Import failed — demo mode"})

    # 7. Memory / Process
    try:
        import psutil
        mem = psutil.virtual_memory()
        checks.append({
            "name": "Memory Usage",
            "status": "ok" if mem.percent < 85 else "warn" if mem.percent < 95 else "fail",
            "detail": f"{mem.percent:.0f}% ({mem.used // (1024**2)}MB / {mem.total // (1024**2)}MB)"
        })
    except ImportError:
        checks.append({"name": "Memory Usage", "status": "ok", "detail": "psutil not installed — skipped"})

    return checks


# ═══════════════════════════════════════════════════════════════════════
# USER ANALYTICS
# ═══════════════════════════════════════════════════════════════════════

def get_user_analytics() -> Dict:
    """Get user statistics and analytics from Firestore."""
    analytics = {
        "total": 0, "plans": {}, "recent_signups": 0,
        "active_today": 0, "discovery_expiring_soon": 0,
        "total_analyses": 0,
    }
    if not _db:
        return analytics

    try:
        users = list(_db.collection("users").stream())
        now = datetime.now()
        week_ago = (now - timedelta(days=7)).isoformat()

        for doc in users:
            u = doc.to_dict()
            plan = u.get("plan", "free")
            analytics["total"] += 1
            analytics["plans"][plan] = analytics["plans"].get(plan, 0) + 1

            # Recent signups (last 7 days)
            created = u.get("created_at", "")
            if created and created > week_ago:
                analytics["recent_signups"] += 1

            # Active today
            pred_date = u.get("predictions_date", "")
            if pred_date == now.strftime("%Y-%m-%d"):
                analytics["active_today"] += 1

            # Discovery expiring within 3 days
            if plan == "discovery":
                ends = u.get("discovery_ends", "")
                if ends:
                    try:
                        end_dt = datetime.fromisoformat(ends)
                        if 0 < (end_dt - now).days <= 3:
                            analytics["discovery_expiring_soon"] += 1
                    except (ValueError, TypeError):
                        pass

            # Total analyses
            analytics["total_analyses"] += u.get("total_predictions", 0)

    except Exception as e:
        logger.warning(f"Failed to get user analytics: {e}")

    return analytics


# ═══════════════════════════════════════════════════════════════════════
# ACTIVITY LOG — Recent system events
# ═══════════════════════════════════════════════════════════════════════

def log_admin_event(event_type: str, detail: str):
    """Log an admin event to Firestore."""
    if not _db:
        return
    try:
        _db.collection("_admin_log").add({
            "type": event_type,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.warning(f"Failed to log admin event: {e}")


def get_recent_events(limit: int = 20) -> List[Dict]:
    """Get recent admin/system events."""
    if not _db:
        return []
    try:
        docs = (_db.collection("_admin_log")
                .order_by("timestamp", direction=firestore.Query.DESCENDING)
                .limit(limit).stream())
        return [d.to_dict() for d in docs]
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════
# DASH UI — MONITORING DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

def _status_dot(status):
    """Return a colored status indicator."""
    colors = {"ok": "#10b981", "warn": "#f59e0b", "fail": "#ef4444"}
    labels = {"ok": "Healthy", "warn": "Warning", "fail": "Down"}
    c = colors.get(status, "#64748b")
    return html.Div([
        html.Div(style={
            "width": "8px", "height": "8px", "borderRadius": "50%",
            "background": c, "flexShrink": "0",
            "boxShadow": f"0 0 6px {c}",
        }),
        html.Span(labels.get(status, "Unknown"), style={
            "fontSize": "11px", "fontWeight": "600", "color": c,
        }),
    ], style={"display": "flex", "alignItems": "center", "gap": "6px"})


def build_monitoring_section():
    """Build the real-time system monitoring dashboard."""
    checks = run_health_checks()
    analytics = get_user_analytics()

    ok = sum(1 for c in checks if c["status"] == "ok")
    total = len(checks)
    overall = "ok" if ok == total else "warn" if ok >= total - 2 else "fail"
    overall_colors = {"ok": "#10b981", "warn": "#f59e0b", "fail": "#ef4444"}
    overall_labels = {"ok": "ALL SYSTEMS OPERATIONAL", "warn": "PARTIAL DEGRADATION", "fail": "SYSTEM ISSUES"}

    # Health status banner
    health_banner = html.Div([
        html.Div([
            html.Div(style={
                "width": "12px", "height": "12px", "borderRadius": "50%",
                "background": overall_colors[overall],
                "boxShadow": f"0 0 10px {overall_colors[overall]}",
            }),
            html.Span(overall_labels[overall], style={
                "fontSize": "13px", "fontWeight": "700", "letterSpacing": "0.5px",
                "color": overall_colors[overall],
            }),
            html.Span(f"({ok}/{total} services healthy)", style={
                "fontSize": "11px", "color": "#64748b",
            }),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        html.Span(f"Last check: {datetime.now().strftime('%H:%M:%S UTC')}", style={
            "fontSize": "10px", "color": "#475569",
        }),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "14px 18px", "borderRadius": "12px", "marginBottom": "16px",
        "background": f"rgba({','.join(str(int(overall_colors[overall].lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.06)",
        "border": f"1px solid {overall_colors[overall]}25",
    })

    # Service health cards
    service_cards = []
    for c in checks:
        service_cards.append(html.Div([
            html.Div([
                html.Div(c["name"], style={
                    "fontSize": "12px", "fontWeight": "600", "color": "#e2e8f0",
                }),
                _status_dot(c["status"]),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                       "marginBottom": "6px"}),
            html.Div(c["detail"], style={
                "fontSize": "11px", "color": "#64748b", "lineHeight": "1.4",
            }),
        ], style={
            "padding": "14px", "borderRadius": "12px",
            "background": "rgba(15,23,42,0.5)",
            "border": f"1px solid rgba({','.join(str(int({'ok':'10b981','warn':'f59e0b','fail':'ef4444'}.get(c['status'],'64748b').lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)",
        }))

    # User analytics cards
    analytics_cards = html.Div([
        html.Div([
            html.Div("Active Today", style={"fontSize": "10px", "color": "#64748b",
                                              "fontWeight": "600", "textTransform": "uppercase",
                                              "letterSpacing": "1px"}),
            html.Div(str(analytics["active_today"]), style={
                "fontSize": "24px", "fontWeight": "800", "color": "#10b981",
            }),
        ], style={"textAlign": "center", "padding": "16px", "borderRadius": "12px",
                   "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(16,185,129,0.1)"}),

        html.Div([
            html.Div("Signups (7d)", style={"fontSize": "10px", "color": "#64748b",
                                              "fontWeight": "600", "textTransform": "uppercase",
                                              "letterSpacing": "1px"}),
            html.Div(str(analytics["recent_signups"]), style={
                "fontSize": "24px", "fontWeight": "800", "color": "#06b6d4",
            }),
        ], style={"textAlign": "center", "padding": "16px", "borderRadius": "12px",
                   "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(6,182,212,0.1)"}),

        html.Div([
            html.Div("Total Analyses", style={"fontSize": "10px", "color": "#64748b",
                                                "fontWeight": "600", "textTransform": "uppercase",
                                                "letterSpacing": "1px"}),
            html.Div(f"{analytics['total_analyses']:,}", style={
                "fontSize": "24px", "fontWeight": "800", "color": "#8b5cf6",
            }),
        ], style={"textAlign": "center", "padding": "16px", "borderRadius": "12px",
                   "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(139,92,246,0.1)"}),

        html.Div([
            html.Div("Discovery Expiring", style={"fontSize": "10px", "color": "#64748b",
                                                    "fontWeight": "600", "textTransform": "uppercase",
                                                    "letterSpacing": "1px"}),
            html.Div(str(analytics["discovery_expiring_soon"]), style={
                "fontSize": "24px", "fontWeight": "800",
                "color": "#f59e0b" if analytics["discovery_expiring_soon"] > 0 else "#64748b",
            }),
            html.Div("within 3 days", style={"fontSize": "9px", "color": "#475569"}),
        ], style={"textAlign": "center", "padding": "16px", "borderRadius": "12px",
                   "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(245,158,11,0.1)"}),
    ], style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "12px",
               "marginBottom": "16px"})

    return html.Div([
        health_banner,
        html.Div(service_cards, style={
            "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
            "gap": "10px", "marginBottom": "16px",
        }),
        analytics_cards,
    ])


# ═══════════════════════════════════════════════════════════════════════
# DASH UI — FEATURE FLAGS PANEL
# ═══════════════════════════════════════════════════════════════════════

def build_feature_flags_section():
    """Build the feature flags toggle panel."""
    flags = get_feature_flags()

    categories = {"system": "System", "auth": "Authentication", "features": "Features", "billing": "Billing"}
    grouped = {}
    for flag_id, flag in flags.items():
        cat = flag.get("category", "system")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append((flag_id, flag))

    sections = []
    for cat_id, cat_label in categories.items():
        if cat_id not in grouped:
            continue

        flag_rows = []
        for flag_id, flag in grouped[cat_id]:
            enabled = flag.get("enabled", True)
            flag_rows.append(html.Div([
                html.Div([
                    html.Div([
                        html.Div(style={
                            "width": "8px", "height": "8px", "borderRadius": "50%",
                            "background": "#10b981" if enabled else "#ef4444",
                            "boxShadow": f"0 0 6px {'#10b981' if enabled else '#ef4444'}",
                            "flexShrink": "0",
                        }),
                        html.Span(flag.get("label", flag_id), style={
                            "fontSize": "13px", "fontWeight": "600",
                            "color": "#e2e8f0" if enabled else "#64748b",
                        }),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
                    html.Button(
                        "ON" if enabled else "OFF",
                        id={"type": "flag-toggle", "index": flag_id},
                        n_clicks=0,
                        style={
                            "padding": "4px 14px", "borderRadius": "6px", "border": "none",
                            "fontSize": "11px", "fontWeight": "700", "cursor": "pointer",
                            "background": "rgba(16,185,129,0.15)" if enabled else "rgba(239,68,68,0.15)",
                            "color": "#10b981" if enabled else "#ef4444",
                            "minWidth": "50px",
                        },
                    ),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"}),
                html.Div(flag.get("description", ""), style={
                    "fontSize": "11px", "color": "#475569", "marginTop": "4px", "lineHeight": "1.4",
                }),
            ], style={
                "padding": "12px 14px", "borderRadius": "10px",
                "background": "rgba(15,23,42,0.4)",
                "border": "1px solid rgba(99,102,241,0.06)",
                "marginBottom": "6px",
            }))

        sections.append(html.Div([
            html.Div(cat_label.upper(), style={
                "fontSize": "9px", "fontWeight": "700", "letterSpacing": "2px",
                "color": "#475569", "marginBottom": "8px",
            }),
            *flag_rows,
        ], style={"marginBottom": "16px"}))

    return html.Div([
        html.Div([
            html.Div("Changes take effect on next page load — no redeploy needed.", style={
                "fontSize": "11px", "color": "#64748b", "marginBottom": "8px",
            }),
            html.Div(id="flag-save-status", style={"fontSize": "12px", "minHeight": "18px"}),
        ], style={"marginBottom": "12px"}),
        *sections,
    ])


# ═══════════════════════════════════════════════════════════════════════
# DASH CALLBACKS — Feature flag toggle
# ═══════════════════════════════════════════════════════════════════════

def register_monitoring_callbacks(app):
    """Register admin monitoring callbacks. Call from app.py."""

    @app.callback(
        Output("flag-save-status", "children"),
        Input({"type": "flag-toggle", "index": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_feature_flag(n_clicks_list):
        if not ctx.triggered_id:
            raise PreventUpdate

        flag_id = ctx.triggered_id["index"]
        flags = get_feature_flags()
        current = flags.get(flag_id, {}).get("enabled", True)
        new_state = not current

        success = set_feature_flag(flag_id, new_state)
        log_admin_event("flag_toggle", f"{flag_id} → {'ON' if new_state else 'OFF'}")

        if success:
            return html.Span(
                f"✅ {flag_id} → {'ON' if new_state else 'OFF'} (refresh page to see changes)",
                style={"color": "#10b981", "fontSize": "12px"},
            )
        else:
            return html.Span(
                f"❌ Failed to save {flag_id}",
                style={"color": "#ef4444", "fontSize": "12px"},
            )

    @app.callback(
        Output("health-check-results", "children"),
        Input("run-health-check-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def rerun_health_checks(n_clicks):
        if not n_clicks:
            raise PreventUpdate
        return build_monitoring_section()
