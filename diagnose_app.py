#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║          MarketLens AI — Full Diagnostic & Health Check             ║
║                                                                      ║
║  Run on GCP Cloud Shell:                                             ║
║    cd ~/TradeAnalytics && python3 diagnose_app.py                    ║
║                                                                      ║
║  Checks: imports, env vars, Firestore, Stripe, APIs, models,        ║
║  plan gating, auth, pages, callbacks, deployment, and more.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import importlib
import traceback
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"
INFO = "ℹ️"
SKIP = "⏭️"

results = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}


def check(label, condition, detail="", warn_only=False):
    if condition:
        results["pass"] += 1
        print(f"  {PASS} {label}")
    elif warn_only:
        results["warn"] += 1
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    else:
        results["fail"] += 1
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))


def skip(label, reason=""):
    results["skip"] += 1
    print(f"  {SKIP} {label}" + (f" — {reason}" if reason else ""))


def section(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ═══════════════════════════════════════════════════════════════════════
# 1. PYTHON ENVIRONMENT
# ═══════════════════════════════════════════════════════════════════════

section("1. PYTHON ENVIRONMENT")

check("Python version ≥ 3.10", sys.version_info >= (3, 10), f"Got {sys.version}")
print(f"  {INFO} Python: {sys.version.split()[0]}")
print(f"  {INFO} Working dir: {os.getcwd()}")

# ═══════════════════════════════════════════════════════════════════════
# 2. CORE DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════

section("2. CORE PYTHON PACKAGES")

REQUIRED_PACKAGES = [
    ("dash", "Dash web framework"),
    ("dash_bootstrap_components", "Dash Bootstrap"),
    ("plotly", "Plotly charts"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("stripe", "Stripe payments"),
    ("flask", "Flask server"),
]

OPTIONAL_PACKAGES = [
    ("torch", "PyTorch (model training)"),
    ("tensorflow", "TensorFlow (models)"),
    ("xgboost", "XGBoost model"),
    ("sklearn", "Scikit-learn ensemble"),
    ("shap", "SHAP explanations"),
    ("google.cloud.firestore", "Firestore DB"),
    ("google.cloud.storage", "GCS model storage"),
]

for pkg, desc in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        check(f"{desc} ({pkg} {ver})", True)
    except ImportError:
        check(f"{desc} ({pkg})", False, "NOT INSTALLED")

print()
for pkg, desc in OPTIONAL_PACKAGES:
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "?")
        check(f"{desc} ({ver})", True)
    except ImportError:
        check(f"{desc}", False, "Not available — feature will be degraded", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 3. ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════

section("3. ENVIRONMENT VARIABLES")

CRITICAL_ENVS = [
    ("STRIPE_SECRET_KEY", "Stripe API key"),
    ("STRIPE_PUBLISHABLE_KEY", "Stripe publishable key"),
    ("STRIPE_WEBHOOK_SECRET", "Stripe webhook secret"),
]

IMPORTANT_ENVS = [
    ("FMP_API_KEY", "Financial Modeling Prep API"),
    ("APP_DOMAIN", "App domain for Stripe redirects"),
    ("ADMIN_EMAILS", "Admin email list"),
    ("STRIPE_STARTER_MONTHLY", "Starter monthly price ID"),
    ("STRIPE_STARTER_YEARLY", "Starter yearly price ID"),
    ("STRIPE_PRO_MONTHLY", "Professional monthly price ID"),
    ("STRIPE_PRO_YEARLY", "Professional yearly price ID"),
]

for var, desc in CRITICAL_ENVS:
    val = os.environ.get(var, "")
    if val:
        masked = val[:8] + "..." + val[-4:] if len(val) > 16 else "***set***"
        check(f"{desc}: {masked}", True)
    else:
        check(f"{desc} ({var})", False, "NOT SET — feature disabled")

print()
for var, desc in IMPORTANT_ENVS:
    val = os.environ.get(var, "")
    if val:
        masked = val[:12] + "..." if len(val) > 16 else val
        check(f"{desc}: {masked}", True)
    else:
        check(f"{desc} ({var})", False, "Not set", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 4. FILE STRUCTURE
# ═══════════════════════════════════════════════════════════════════════

section("4. FILE STRUCTURE")

REQUIRED_FILES = [
    "app.py",
    "saas_auth.py",
    "pages.py",
    "requirements.txt",
]

OPTIONAL_FILES = [
    "setup_stripe.py",
    "Dockerfile",
    "Procfile",
    ".gcloudignore",
    "disclaimer.py",
    "email_service.py",
    "app_guide.py",
    "enhprog.py",
    "gcs_model_loader.py",
    "ai_backtest_engine.py",
    "ai_portfolio_system.py",
    "email_admin_panel.py",
]

for f in REQUIRED_FILES:
    exists = os.path.isfile(f)
    size = os.path.getsize(f) if exists else 0
    check(f"{f} ({size:,} bytes)" if exists else f, exists, "MISSING — critical file")

print()
for f in OPTIONAL_FILES:
    exists = os.path.isfile(f)
    if exists:
        size = os.path.getsize(f)
        check(f"{f} ({size:,} bytes)", True)
    else:
        check(f"{f}", False, "Not present — optional module", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 5. SYNTAX VALIDATION
# ═══════════════════════════════════════════════════════════════════════

section("5. SYNTAX VALIDATION")

import ast

for f in ["app.py", "saas_auth.py", "pages.py"]:
    if os.path.isfile(f):
        try:
            with open(f) as fh:
                ast.parse(fh.read())
            check(f"{f} — valid Python syntax", True)
        except SyntaxError as e:
            check(f"{f} — syntax error", False, f"Line {e.lineno}: {e.msg}")
    else:
        skip(f"{f} — syntax check", "file missing")


# ═══════════════════════════════════════════════════════════════════════
# 6. APP MODULE IMPORTS
# ═══════════════════════════════════════════════════════════════════════

section("6. MODULE IMPORT CHAIN")

# Test saas_auth imports
try:
    import saas_auth
    check("saas_auth module imports successfully", True)

    # Verify key functions exist
    critical_funcs = [
        "build_login_page", "build_pricing_page", "build_user_badge",
        "create_user", "authenticate_user", "get_user_by_token",
        "check_prediction_limit", "record_prediction", "get_user_plan",
        "get_allowed_tickers", "get_allowed_timeframes", "get_models_limit",
        "can_access_feature", "is_admin", "create_checkout_session",
        "register_webhook_route", "reset_user_password",
        "get_discovery_days_remaining", "is_discovery_active",
        "build_upgrade_prompt", "build_limit_reached_prompt",
    ]
    missing_funcs = [f for f in critical_funcs if not hasattr(saas_auth, f)]
    check(f"saas_auth — all {len(critical_funcs)} critical functions present",
          len(missing_funcs) == 0, f"Missing: {', '.join(missing_funcs)}")

except Exception as e:
    check("saas_auth module", False, str(e))

# Test pages imports
try:
    from pages import (
        build_analytics_page, build_backtest_page, build_portfolio_page,
        build_ftmo_page, build_model_training_page, build_admin_page,
        run_backtest, run_regime_analysis, run_drift_detection,
        run_portfolio_optimization,
    )
    check("pages module — all page builders import OK", True)
except ImportError as e:
    check("pages module", False, str(e))
except Exception as e:
    check("pages module", False, f"Unexpected: {e}", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 7. PLAN DEFINITIONS & GATING
# ═══════════════════════════════════════════════════════════════════════

section("7. SUBSCRIPTION PLANS & FEATURE GATING")

try:
    from saas_auth import PLANS, DISCOVERY_DURATION_DAYS, ALL_TICKERS

    EXPECTED_PLANS = ["free", "discovery", "starter", "professional", "enterprise"]
    for plan_id in EXPECTED_PLANS:
        present = plan_id in PLANS
        check(f"Plan '{plan_id}' defined", present)
        if present:
            p = PLANS[plan_id]
            # Verify required keys
            required_keys = ["name", "predictions_per_day", "tickers", "timeframes", "features", "models_limit"]
            missing_keys = [k for k in required_keys if k not in p]
            if missing_keys:
                check(f"  Plan '{plan_id}' missing keys", False, ', '.join(missing_keys))

    print()
    check(f"Discovery duration: {DISCOVERY_DURATION_DAYS} days", DISCOVERY_DURATION_DAYS > 0)
    check(f"Tickers defined: {len(ALL_TICKERS)} tickers", len(ALL_TICKERS) >= 10,
          f"Only {len(ALL_TICKERS)} tickers found")

    # Verify plan hierarchy makes sense
    free_preds = PLANS["free"]["predictions_per_day"]
    disc_preds = PLANS["discovery"]["predictions_per_day"]
    starter_preds = PLANS["starter"]["predictions_per_day"]
    pro_preds = PLANS["professional"]["predictions_per_day"]
    ent_preds = PLANS["enterprise"]["predictions_per_day"]

    check(f"Plan hierarchy: free({free_preds}) < discovery({disc_preds}) < starter({starter_preds}) < pro({pro_preds}) < enterprise({ent_preds})",
          free_preds < disc_preds < starter_preds < pro_preds < ent_preds)

    # Check feature escalation
    print()
    GATED_FEATURES = ["backtesting", "shap_explanations", "regime_detection",
                      "drift_alerts", "mtf_analysis", "ftmo_dashboard", "model_training"]
    for feat in GATED_FEATURES:
        free_has = PLANS["free"]["features"].get(feat, False)
        pro_has = PLANS["professional"]["features"].get(feat, False)
        ent_has = PLANS["enterprise"]["features"].get(feat, False)
        check(f"Feature '{feat}': free={free_has}, pro={pro_has}, enterprise={ent_has}",
              ent_has, f"Enterprise should have all features", warn_only=not ent_has)

    # Stripe price IDs
    print()
    for plan_id in ["starter", "professional"]:
        p = PLANS[plan_id]
        monthly = p.get("stripe_price_monthly", "")
        yearly = p.get("stripe_price_yearly", "")
        check(f"{plan_id} Stripe monthly price ID", bool(monthly),
              "Not configured — upgrades won't work", warn_only=True)
        check(f"{plan_id} Stripe yearly price ID", bool(yearly),
              "Not configured", warn_only=True)

    # Enterprise should be contact-only
    check("Enterprise is contact-only (no Stripe price)",
          PLANS["enterprise"].get("is_contact_only", False) or
          PLANS["enterprise"].get("stripe_price_monthly") is None)

except Exception as e:
    check("Plan definitions", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 8. AUTHENTICATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════

section("8. AUTHENTICATION SYSTEM")

try:
    from saas_auth import (
        create_user, authenticate_user, get_user_by_token,
        logout_user, _hash_password, FIRESTORE_AVAILABLE,
    )

    check(f"Firestore available: {FIRESTORE_AVAILABLE}", True)
    if not FIRESTORE_AVAILABLE:
        check("Firestore connection", False,
              "Using local fallback — data won't persist across deploys", warn_only=True)

    # Test password hashing
    hashed, salt = _hash_password("test_password_123")
    check("Password hashing works", len(hashed) == 64 and len(salt) == 32,
          "Hash function returned unexpected format")

    # Test user creation (dry run — we won't actually create in prod)
    check("create_user function callable", callable(create_user))
    check("authenticate_user function callable", callable(authenticate_user))
    check("get_user_by_token function callable", callable(get_user_by_token))

    # Check admin config
    from saas_auth import ADMIN_EMAILS, is_admin
    check(f"Admin emails configured: {ADMIN_EMAILS}", len(ADMIN_EMAILS) > 0)

except Exception as e:
    check("Auth system", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 9. STRIPE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════

section("9. STRIPE INTEGRATION")

try:
    import stripe as stripe_mod
    check(f"Stripe SDK version: {stripe_mod.__version__}", True)

    stripe_key = os.environ.get("STRIPE_SECRET_KEY", "")
    if stripe_key:
        # Test Stripe API connectivity
        try:
            stripe_mod.api_key = stripe_key
            products = stripe_mod.Product.list(limit=1)
            check("Stripe API connection — live", True)
            product_count = len(products.data)
            check(f"Stripe products found: {product_count}", product_count > 0,
                  "No products — run setup_stripe.py", warn_only=True)

            # Check for price IDs
            prices = stripe_mod.Price.list(limit=10, active=True)
            active_prices = len(prices.data)
            check(f"Active Stripe prices: {active_prices}", active_prices >= 4,
                  f"Expected 4+ (2 plans x 2 intervals), got {active_prices}", warn_only=True)

        except stripe_mod.error.AuthenticationError:
            check("Stripe API authentication", False, "Invalid API key")
        except Exception as e:
            check("Stripe API connection", False, str(e), warn_only=True)
    else:
        skip("Stripe API test", "STRIPE_SECRET_KEY not set")

    # Test webhook route registration
    from saas_auth import register_webhook_route
    check("Webhook route function available", callable(register_webhook_route))

    # Test checkout session creator
    from saas_auth import create_checkout_session, create_customer_portal_session
    check("Checkout session creator available", callable(create_checkout_session))
    check("Customer portal session creator available", callable(create_customer_portal_session))

except Exception as e:
    check("Stripe integration", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 10. FIRESTORE DATABASE
# ═══════════════════════════════════════════════════════════════════════

section("10. FIRESTORE DATABASE")

try:
    from google.cloud import firestore
    db = firestore.Client()
    check("Firestore client created", True)

    # Test read access
    try:
        test_ref = db.collection("users").limit(1).stream()
        user_count = sum(1 for _ in test_ref)
        check(f"Firestore 'users' collection accessible ({user_count}+ docs)", True)
    except Exception as e:
        check("Firestore read access", False, str(e))

    # Check collections exist
    for coll_name in ["users", "trading_results", "contact_submissions"]:
        try:
            docs = list(db.collection(coll_name).limit(1).stream())
            check(f"Collection '{coll_name}' exists ({len(docs)} sample docs)", True)
        except Exception as e:
            check(f"Collection '{coll_name}'", False, str(e), warn_only=True)

    # Count total users
    try:
        all_users = list(db.collection("users").stream())
        plan_counts = {}
        for doc in all_users:
            data = doc.to_dict()
            plan = data.get("plan", "free")
            plan_counts[plan] = plan_counts.get(plan, 0) + 1
        check(f"Total users: {len(all_users)}", True)
        for plan, count in sorted(plan_counts.items()):
            print(f"    {INFO} {plan}: {count} users")
    except Exception as e:
        check("User count", False, str(e), warn_only=True)

except ImportError:
    skip("Firestore checks", "google-cloud-firestore not installed")
except Exception as e:
    check("Firestore connection", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 11. FMP API (Financial Data)
# ═══════════════════════════════════════════════════════════════════════

section("11. FMP API (Financial Data)")

fmp_key = os.environ.get("FMP_API_KEY", "")
if fmp_key:
    try:
        import requests
        url = f"https://financialmodelingprep.com/api/v3/quote/AAPL?apikey={fmp_key}"
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200 and resp.json():
            data = resp.json()[0]
            check(f"FMP API live — AAPL: ${data.get('price', '?')}", True)
        else:
            check("FMP API response", False, f"Status {resp.status_code}")
    except Exception as e:
        check("FMP API connection", False, str(e))

    # Test a few tickers
    test_tickers = ["BTCUSD", "ETHUSD", "^GSPC"]
    for ticker in test_tickers:
        try:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries=5&apikey={fmp_key}"
            resp = requests.get(url, timeout=10)
            has_data = resp.status_code == 200 and "historical" in resp.text
            check(f"FMP data for {ticker}", has_data,
                  f"No historical data returned", warn_only=True)
        except Exception as e:
            check(f"FMP {ticker}", False, str(e), warn_only=True)
else:
    skip("FMP API tests", "FMP_API_KEY not set — app runs in demo/simulation mode")


# ═══════════════════════════════════════════════════════════════════════
# 12. AI MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════

section("12. AI MODEL ARCHITECTURES")

MODEL_CLASSES = [
    ("enhprog", "XGBoostTimeSeriesModel", "XGBoost"),
    ("enhprog", "SklearnEnsemble", "Sklearn Ensemble"),
    ("enhprog", "AdvancedTransformer", "Advanced Transformer"),
    ("enhprog", "CNNLSTMAttention", "CNN-LSTM Attention"),
    ("enhprog", "EnhancedTCN", "Temporal Conv. Network"),
    ("enhprog", "EnhancedInformer", "Informer"),
    ("enhprog", "EnhancedNBeats", "N-BEATS"),
    ("enhprog", "LSTMGRUEnsemble", "LSTM-GRU Ensemble"),
]

for module_name, class_name, display_name in MODEL_CLASSES:
    try:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name, None)
        if cls:
            check(f"{display_name} ({class_name})", True)
        else:
            check(f"{display_name} ({class_name})", False, f"Class not found in {module_name}", warn_only=True)
    except ImportError:
        check(f"{display_name} ({module_name}.{class_name})", False,
              f"Module '{module_name}' not importable", warn_only=True)
    except Exception as e:
        check(f"{display_name}", False, str(e), warn_only=True)

# Check for saved models on disk / GCS
print()
MODEL_DIRS = ["models", "saved_models", "/tmp/models"]
for d in MODEL_DIRS:
    if os.path.isdir(d):
        files = os.listdir(d)
        model_files = [f for f in files if f.endswith((".pt", ".pth", ".pkl", ".joblib", ".h5"))]
        check(f"Model directory '{d}': {len(model_files)} model files", len(model_files) > 0,
              "No saved model files found", warn_only=True)
    else:
        skip(f"Model directory '{d}'", "does not exist")

# GCS model loader
try:
    from gcs_model_loader import ensure_models_available
    check("GCS model loader available", True)
except ImportError:
    check("GCS model loader", False, "gcs_model_loader.py not found — models load from disk only", warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 13. MULTI-TIMEFRAME SYSTEM
# ═══════════════════════════════════════════════════════════════════════

section("13. MULTI-TIMEFRAME ANALYSIS")

try:
    from enhprog import MultiTimeframeDataManager, MultiTimeframeAnalyzer
    check("MultiTimeframeDataManager importable", True)
    check("MultiTimeframeAnalyzer importable", True)

    TIMEFRAMES = ["15min", "1hour", "4hour", "1day"]
    print(f"  {INFO} Supported timeframes: {', '.join(TIMEFRAMES)}")

except ImportError as e:
    check("Multi-timeframe modules", False, f"enhprog not available: {e}", warn_only=True)

# Verify MTF section builder in app
try:
    # Check that the function exists by reading the source
    with open("app.py") as f:
        source = f.read()
    check("_build_mtf_section function defined in app.py", "_build_mtf_section" in source)
    check("MTF timeframe selector in dashboard", "mtf-timeframes" in source or "mtf_timeframes" in source)
except Exception as e:
    check("MTF in app.py", False, str(e), warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 14. ANALYSIS & RESEARCH FEATURES
# ═══════════════════════════════════════════════════════════════════════

section("14. ANALYSIS & RESEARCH FEATURES")

try:
    with open("app.py") as f:
        app_source = f.read()
    with open("pages.py") as f:
        pages_source = f.read()

    features_to_check = [
        ("PredictionEngine", "AI analysis engine", app_source),
        ("run_regime_analysis", "Regime detection (HMM)", pages_source),
        ("run_drift_detection", "Drift detection", pages_source),
        ("run_backtest", "Backtesting engine", pages_source),
        ("run_portfolio_optimization", "Portfolio optimization", pages_source),
        ("generate_risk_metrics", "Risk metrics generator", app_source),
        ("build_price_trajectory_chart", "Price trajectory chart", app_source),
        ("build_risk_gauge", "Risk gauge visualization", app_source),
        ("build_prediction_results", "Analysis results builder", app_source),
        ("build_trading_strategy_tab", "Trading strategy tab", app_source),
        ("build_forecast_tab", "Forecast tab", app_source),
        ("build_risk_tab", "Risk analysis tab", app_source),
        ("build_analytics_page", "Analytics page", pages_source),
        ("build_backtest_page", "Backtest page", pages_source),
        ("build_portfolio_page", "Portfolio page", pages_source),
        ("build_model_training_page", "Model training page", pages_source),
        ("build_admin_page", "Admin panel", pages_source),
    ]

    for func_name, desc, source in features_to_check:
        check(f"{desc} ({func_name})", f"def {func_name}" in source or f"class {func_name}" in source)

except Exception as e:
    check("Feature check", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 15. DASH CALLBACKS & ROUTING
# ═══════════════════════════════════════════════════════════════════════

section("15. DASH CALLBACKS & ROUTING")

try:
    with open("app.py") as f:
        app_source = f.read()

    CRITICAL_CALLBACKS = [
        ("route_auth", "Auth routing (login/register/pricing)"),
        ("handle_login", "Login handler"),
        ("handle_register", "Registration handler"),
        ("handle_demo_mode", "Demo mode handler"),
        ("handle_logout", "Logout handler"),
        ("handle_upgrade_click", "Upgrade button handler"),
        ("handle_pricing_upgrade", "Pricing page upgrade"),
        ("toggle_auth_tabs", "Auth tab toggle (login/register/forgot)"),
        ("handle_password_reset", "Password reset handler"),
        ("route_page", "Main page router"),
        ("run_prediction", "AI analysis execution"),
        ("nav_button_click", "Navigation click handler"),
        ("highlight_active_nav", "Active nav highlighting"),
        ("quick_select_group", "Ticker group quick select"),
        ("rerun_backtest", "Backtest rerun"),
        ("toggle_contact_modal", "Contact modal toggle"),
        ("send_contact_form", "Contact form submission"),
        ("handle_disclaimer", "Disclaimer handler"),
    ]

    for func_name, desc in CRITICAL_CALLBACKS:
        check(f"{desc} ({func_name})", f"def {func_name}" in app_source)

    # Check clientside callback for Stripe redirect
    check("Stripe redirect clientside callback",
          "clientside_callback" in app_source and "redirect-store" in app_source)

except Exception as e:
    check("Callback check", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 16. UI COMPONENT IDs (Callback Dependencies)
# ═══════════════════════════════════════════════════════════════════════

section("16. UI COMPONENT IDs")

try:
    with open("app.py") as f:
        app_source = f.read()

    # Combine with saas_auth source for login page IDs
    with open("saas_auth.py") as f:
        auth_source = f.read()

    combined = app_source + auth_source

    CRITICAL_IDS = [
        "app-container", "url", "user-session", "redirect-store",
        "login-btn", "login-email", "login-password",
        "register-btn", "register-name", "register-email", "register-password",
        "auth-tab-login", "auth-tab-register",
        "forgot-password-link", "back-to-login-link",
        "reset-email", "reset-password", "reset-password-confirm", "reset-password-btn",
        "demo-mode-link",
        "login-form", "register-form", "forgot-form",
        "auth-error", "auth-success",
    ]

    missing_ids = [cid for cid in CRITICAL_IDS if f'"{cid}"' not in combined and f"'{cid}'" not in combined]
    check(f"All {len(CRITICAL_IDS)} critical component IDs present",
          len(missing_ids) == 0,
          f"Missing: {', '.join(missing_ids)}")

except Exception as e:
    check("Component ID check", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 17. LANDING PAGE (NEW SPLIT LAYOUT)
# ═══════════════════════════════════════════════════════════════════════

section("17. LANDING PAGE LAYOUT")

try:
    with open("saas_auth.py") as f:
        auth_source = f.read()

    check("Split layout (flexDirection: row)", "flexDirection" in auth_source and "row" in auth_source)
    check("Landing side content", "landing_side" in auth_source or "INSTITUTIONAL-GRADE" in auth_source)
    check("Feature cards helper (_build_feature_card)", "def _build_feature_card" in auth_source)
    check("Stat pills helper (_build_stat_pill)", "def _build_stat_pill" in auth_source)
    check("Model tags helper (_build_model_tag)", "def _build_model_tag" in auth_source)
    check("Research language (not 'predictions')",
          "Market Intelligence" in auth_source or "market intelligence" in auth_source,
          "Still using 'predictions' in hero", warn_only=True)
    check("FTMO removed from landing",
          "FTMO Challenge Dashboard" not in auth_source,
          "FTMO still referenced in landing page", warn_only=True)
    check("MSE score removed from landing",
          "MSE Score" not in auth_source,
          "MSE still shown on landing", warn_only=True)

except Exception as e:
    check("Landing page check", False, str(e))


# ═══════════════════════════════════════════════════════════════════════
# 18. CLOUD RUN DEPLOYMENT
# ═══════════════════════════════════════════════════════════════════════

section("18. CLOUD RUN DEPLOYMENT STATUS")

try:
    import subprocess

    # Check active Cloud Run service
    result = subprocess.run(
        ["gcloud", "run", "services", "describe", "tradeanalytics",
         "--region", "us-central1", "--format",
         "value(status.url,status.conditions[0].status,spec.template.spec.containers[0].resources.limits.memory,spec.template.spec.containers[0].resources.limits.cpu)"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("\n")
        url = parts[0] if len(parts) > 0 else "?"
        print(f"  {INFO} Service URL: {url}")
        check("Cloud Run service active", True)
    else:
        check("Cloud Run service", False, "Could not describe service", warn_only=True)

    # Last build
    result = subprocess.run(
        ["gcloud", "builds", "list", "--limit=1", "--format=value(id,status,createTime)"],
        capture_output=True, text=True, timeout=15
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split("\t")
        build_id = parts[0][:12] if parts else "?"
        status = parts[1] if len(parts) > 1 else "?"
        created = parts[2] if len(parts) > 2 else "?"
        check(f"Last build: {status} ({created})", status == "SUCCESS",
              f"Last build status: {status}")
    else:
        skip("Build history", "Could not fetch")

    # Git status
    result = subprocess.run(["git", "status", "--short"], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        uncommitted = result.stdout.strip()
        if uncommitted:
            lines = uncommitted.count("\n") + 1
            check(f"Git — {lines} uncommitted changes", False,
                  "Uncommitted files present", warn_only=True)
            for line in uncommitted.split("\n")[:5]:
                print(f"    {INFO} {line}")
            if lines > 5:
                print(f"    {INFO} ... and {lines - 5} more")
        else:
            check("Git — working tree clean", True)

except FileNotFoundError:
    skip("Cloud Run checks", "gcloud CLI not available")
except subprocess.TimeoutExpired:
    skip("Cloud Run checks", "Command timed out")
except Exception as e:
    check("Deployment check", False, str(e), warn_only=True)


# ═══════════════════════════════════════════════════════════════════════
# 19. LIVE HTTP HEALTH CHECK
# ═══════════════════════════════════════════════════════════════════════

section("19. LIVE HTTP HEALTH CHECK")

app_domain = os.environ.get("APP_DOMAIN", "")
if app_domain:
    try:
        import requests
        resp = requests.get(app_domain, timeout=15, allow_redirects=True)
        check(f"HTTP GET {app_domain} → {resp.status_code}", resp.status_code == 200,
              f"Status {resp.status_code}")
        check(f"Response size: {len(resp.content):,} bytes",
              len(resp.content) > 1000, "Response too small — may be an error page")

        # Check that it returns Dash HTML
        is_dash = "dash" in resp.text.lower() or "_dash-" in resp.text or "react" in resp.text.lower()
        check("Response contains Dash app markup", is_dash,
              "Doesn't look like a Dash app response", warn_only=True)

    except requests.exceptions.Timeout:
        check("HTTP health check", False, "Request timed out (15s)")
    except requests.exceptions.ConnectionError as e:
        check("HTTP health check", False, f"Connection failed: {e}")
    except Exception as e:
        check("HTTP health check", False, str(e))
else:
    skip("HTTP health check", "APP_DOMAIN not set")


# ═══════════════════════════════════════════════════════════════════════
# 20. REQUIREMENTS.TXT AUDIT
# ═══════════════════════════════════════════════════════════════════════

section("20. REQUIREMENTS.TXT AUDIT")

if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f:
        req_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    check(f"requirements.txt has {len(req_lines)} entries", len(req_lines) > 0)

    MUST_HAVE = ["dash", "plotly", "numpy", "pandas", "stripe", "gunicorn", "flask"]
    for pkg in MUST_HAVE:
        found = any(pkg in line.lower() for line in req_lines)
        check(f"'{pkg}' in requirements.txt", found,
              f"Missing from requirements — deploy may fail", warn_only=(pkg == "flask"))

else:
    check("requirements.txt", False, "MISSING — deployment will fail")


# ═══════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ═══════════════════════════════════════════════════════════════════════

section("DIAGNOSTIC SUMMARY")

total = results["pass"] + results["fail"] + results["warn"] + results["skip"]
score = results["pass"] / max(total - results["skip"], 1) * 100

print(f"""
  {PASS} Passed:  {results['pass']}
  {FAIL} Failed:  {results['fail']}
  {WARN} Warnings: {results['warn']}
  {SKIP} Skipped: {results['skip']}
  ──────────────────
  Total checks: {total}
  Health score: {score:.0f}%
""")

if results["fail"] == 0 and results["warn"] <= 3:
    print(f"  🟢 SYSTEM HEALTHY — Ready for production")
elif results["fail"] == 0:
    print(f"  🟡 SYSTEM FUNCTIONAL — {results['warn']} warnings to review")
elif results["fail"] <= 3:
    print(f"  🟠 SYSTEM DEGRADED — {results['fail']} failures need attention")
else:
    print(f"  🔴 SYSTEM CRITICAL — {results['fail']} failures must be fixed")

print(f"\n  {INFO} Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"{'═'*60}\n")

sys.exit(0 if results["fail"] == 0 else 1)
