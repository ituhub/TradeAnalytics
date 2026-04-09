"""
AUTH, PAYMENTS & PLAN GATING — MarketLens AI
========================================================================
Handles:
  - User authentication (Firestore-backed, session cookies)
  - Stripe subscription billing (3 tiers + annual)
  - Discovery plan: 14-day auto-assigned welcome plan on registration
  - Plan-based feature gating (tickers, timeframes, predictions/day)
  - Login / Register / Pricing page layouts for Dash
  - Usage tracking and rate limiting
========================================================================
"""

import os
import json
import hashlib
import secrets
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, Any
from functools import wraps

import stripe
from dash import html, dcc, Input, Output, State, callback, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

# =============================================================================
# STRIPE CONFIGURATION
# =============================================================================

STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
APP_DOMAIN = os.environ.get("APP_DOMAIN", "http://localhost:8050")

# =============================================================================
# ADMIN CONFIGURATION
# =============================================================================
# Comma-separated list of admin emails. Only these users can access Admin Panel.
# Set via environment variable or hardcode your email below.

ADMIN_EMAILS = [
    e.strip().lower()
    for e in os.environ.get("ADMIN_EMAILS", "itubusinesshub@gmail.com").split(",")
    if e.strip()
]


def is_admin(user: Optional[Dict]) -> bool:
    """Check if user is an admin."""
    if not user:
        return False
    return user.get("email", "").strip().lower() in ADMIN_EMAILS

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe API configured")
else:
    logger.warning("⚠️ STRIPE_SECRET_KEY not set — payments disabled")

# =============================================================================
# PLAN DEFINITIONS
# =============================================================================

DISCOVERY_DURATION_DAYS = 14  # How long the Discovery plan lasts after registration

PLANS = {
    "free": {
        "name": "Free Demo",
        "predictions_per_day": 2,
        "tickers": ["ETHUSD", "BTCUSD", "SOLUSD"],
        "timeframes": ["1day"],
        "features": {
            "backtesting": False,
            "portfolio": False,
            "shap_explanations": False,
            "regime_detection": False,
            "drift_alerts": False,
            "mtf_analysis": False,
            "api_access": False,
            "ftmo_dashboard": False,
            "model_training": False,
        },
        "models_limit": 2,  # downgraded from 3 to make Discovery feel like an upgrade
        "stripe_price_monthly": None,
        "stripe_price_yearly": None,
    },
    "discovery": {
        "name": "Discovery",
        "predictions_per_day": 5,
        "tickers": ["ETHUSD", "BTCUSD", "SOLUSD", "USDJPY", "GC=F", "NG=F", "EURUSD"],
        "timeframes": ["1hour", "4hour", "1day"],
        "features": {
            "backtesting": True,
            "portfolio": False,
            "shap_explanations": True,
            "regime_detection": True,
            "drift_alerts": False,
            "mtf_analysis": False,
            "api_access": False,
            "ftmo_dashboard": False,
            "model_training": False,
        },
        "models_limit": 4,
        "stripe_price_monthly": None,
        "stripe_price_yearly": None,
        "is_promo": True,  # marks this as a non-purchasable promotional plan
        "duration_days": DISCOVERY_DURATION_DAYS,
    },
    "starter": {
        "name": "Starter",
        "predictions_per_day": 10,
        "tickers": ["BTCUSD", "ETHUSD", "SOLUSD", "SPY", "AAPL", "EURUSD", "GC=F"],
        "timeframes": ["1hour", "4hour", "1day"],
        "features": {
            "backtesting": True,
            "portfolio": False,
            "shap_explanations": True,
            "regime_detection": True,
            "drift_alerts": True,
            "mtf_analysis": True,
            "api_access": False,
            "ftmo_dashboard": False,
            "model_training": False,
        },
        "models_limit": 4,
        "stripe_price_monthly": os.environ.get("STRIPE_STARTER_MONTHLY", ""),
        "stripe_price_yearly": os.environ.get("STRIPE_STARTER_YEARLY", ""),
        "price_display": {"monthly": 39, "yearly": 29},
        "trial_days": 7,
    },
    "professional": {
        "name": "Professional",
        "predictions_per_day": 25,
        "tickers": "all",  # special value = all tickers
        "timeframes": ["15min", "1hour", "4hour", "1day"],
        "features": {
            "backtesting": True,
            "portfolio": False,
            "shap_explanations": True,
            "regime_detection": True,
            "drift_alerts": True,
            "mtf_analysis": True,
            "api_access": False,
            "ftmo_dashboard": True,
            "model_training": True,
            "monte_carlo": False,
            "custom_tickers": False,
        },
        "models_limit": 8,
        "stripe_price_monthly": os.environ.get("STRIPE_PRO_MONTHLY", ""),
        "stripe_price_yearly": os.environ.get("STRIPE_PRO_YEARLY", ""),
        "price_display": {"monthly": 89, "yearly": 69},
        "trial_days": 14,
    },
    "enterprise": {
        "name": "Enterprise",
        "predictions_per_day": 9999,  # effectively unlimited
        "tickers": "all",
        "timeframes": ["15min", "1hour", "4hour", "1day"],
        "features": {
            "backtesting": True,
            "portfolio": True,
            "shap_explanations": True,
            "regime_detection": True,
            "drift_alerts": True,
            "mtf_analysis": True,
            "api_access": True,
            "ftmo_dashboard": True,
            "model_training": True,
            "monte_carlo": True,
            "custom_tickers": True,
        },
        "models_limit": 8,
        "stripe_price_monthly": None,  # custom pricing only
        "stripe_price_yearly": None,
        "price_display": None,  # no fixed price — contact support
        "is_contact_only": True,  # marks this as non-purchasable
    },
}

# Full ticker list (imported from app.py at runtime)
ALL_TICKERS = [
    "ETHUSD", "SOLUSD", "BTCUSD",
    "USDJPY",
    "^GDAXI", "^GSPC", "^HSI",
    "CC=F", "NG=F", "GC=F", "KC=F", "SI=F", "HG=F",
]


# =============================================================================
# USER DATABASE (Firestore-backed with local fallback)
# =============================================================================

_firestore_db = None
FIRESTORE_AVAILABLE = False

try:
    from google.cloud import firestore as _firestore_mod
    _firestore_db = _firestore_mod.Client()
    FIRESTORE_AVAILABLE = True
    logger.info("✅ Firestore connected for user management")
except Exception:
    logger.info("ℹ️ Firestore not available — using local user storage")

# Local fallback storage
_LOCAL_USERS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "users.json"
)
os.makedirs(os.path.dirname(_LOCAL_USERS_FILE), exist_ok=True)


def _load_local_users() -> Dict:
    try:
        if os.path.exists(_LOCAL_USERS_FILE):
            with open(_LOCAL_USERS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_local_users(users: Dict):
    try:
        with open(_LOCAL_USERS_FILE, "w") as f:
            json.dump(users, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save local users: {e}")


def _hash_password(password: str, salt: str = None) -> Tuple[str, str]:
    """Hash password with salt using SHA-256."""
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    return hashed, salt


# =============================================================================
# USER MANAGEMENT
# =============================================================================

def create_user(email: str, password: str, name: str = "") -> Dict:
    """Create a new user account."""
    email = email.strip().lower()

    # Check if user exists
    existing = get_user_by_email(email)
    if existing:
        return {"error": "Email already registered"}

    hashed_pw, salt = _hash_password(password)
    user_id = f"user_{secrets.token_hex(8)}"

    user = {
        "user_id": user_id,
        "email": email,
        "name": name or email.split("@")[0],
        "password_hash": hashed_pw,
        "password_salt": salt,
        "plan": "discovery",
        "stripe_customer_id": None,
        "stripe_subscription_id": None,
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat(),
        "predictions_today": 0,
        "predictions_date": datetime.now().strftime("%Y-%m-%d"),
        "total_predictions": 0,
        "session_token": None,
        "trial_started": None,
        "trial_ends": None,
        # Discovery plan auto-assignment
        "discovery_started": datetime.now().isoformat(),
        "discovery_ends": (datetime.now() + timedelta(days=DISCOVERY_DURATION_DAYS)).isoformat(),
        "discovery_used": True,  # prevents re-activation
    }

    # Create Stripe customer
    if STRIPE_SECRET_KEY:
        try:
            customer = stripe.Customer.create(
                email=email,
                name=user["name"],
                metadata={"user_id": user_id},
            )
            user["stripe_customer_id"] = customer.id
            logger.info(f"✅ Stripe customer created: {customer.id}")
        except Exception as e:
            logger.warning(f"Stripe customer creation failed: {e}")

    # Save user
    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            _firestore_db.collection("users").document(user_id).set(user)
            logger.info(f"✅ User created in Firestore: {email}")
        except Exception as e:
            logger.warning(f"Firestore save failed: {e}")
            _save_user_local(user)
    else:
        _save_user_local(user)

    return user


def _save_user_local(user: Dict):
    users = _load_local_users()
    users[user["user_id"]] = user
    _save_local_users(users)


def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """Authenticate user and return user dict or None."""
    email = email.strip().lower()
    user = get_user_by_email(email)
    if not user:
        return None

    hashed, _ = _hash_password(password, user.get("password_salt", ""))
    if hashed != user.get("password_hash", ""):
        return None

    # Generate session token
    token = secrets.token_hex(32)
    user["session_token"] = token
    user["last_login"] = datetime.now().isoformat()
    _update_user(user)

    return user


def get_user_by_email(email: str) -> Optional[Dict]:
    """Look up user by email."""
    email = email.strip().lower()

    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            docs = _firestore_db.collection("users").where("email", "==", email).limit(1).stream()
            for doc in docs:
                return doc.to_dict()
        except Exception:
            pass

    # Local fallback
    users = _load_local_users()
    for u in users.values():
        if u.get("email", "").lower() == email:
            return u

    return None


def get_user_by_token(token: str) -> Optional[Dict]:
    """Look up user by session token."""
    if not token:
        return None

    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            docs = _firestore_db.collection("users").where("session_token", "==", token).limit(1).stream()
            for doc in docs:
                return doc.to_dict()
        except Exception:
            pass

    users = _load_local_users()
    for u in users.values():
        if u.get("session_token") == token:
            return u

    return None


def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Look up user by ID."""
    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            doc = _firestore_db.collection("users").document(user_id).get()
            if doc.exists:
                return doc.to_dict()
        except Exception:
            pass

    users = _load_local_users()
    return users.get(user_id)


def _update_user(user: Dict):
    """Update user in storage."""
    uid = user.get("user_id")
    if not uid:
        return

    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            _firestore_db.collection("users").document(uid).set(user)
            return
        except Exception:
            pass

    _save_user_local(user)


def logout_user(user_id: str):
    """Clear session token."""
    user = get_user_by_id(user_id)
    if user:
        user["session_token"] = None
        _update_user(user)


# =============================================================================
# PLAN GATING — Check what a user can access
# =============================================================================

def get_user_plan(user: Optional[Dict]) -> Dict:
    """Get the plan config for a user. Auto-expires Discovery plan."""
    if not user:
        return PLANS["free"]
    plan_id = user.get("plan", "free")

    # Auto-expire Discovery plan
    if plan_id == "discovery":
        discovery_ends = user.get("discovery_ends")
        if discovery_ends:
            try:
                end_dt = datetime.fromisoformat(discovery_ends)
                if datetime.now() > end_dt:
                    # Discovery expired — downgrade to free
                    user["plan"] = "free"
                    _update_user(user)
                    logger.info(f"🔻 Discovery expired for {user.get('email')} → free")
                    return PLANS["free"]
            except (ValueError, TypeError):
                pass

    return PLANS.get(plan_id, PLANS["free"])


def get_discovery_days_remaining(user: Optional[Dict]) -> int:
    """Get number of days left on Discovery plan. Returns 0 if not on Discovery or expired."""
    if not user or user.get("plan") != "discovery":
        return 0
    discovery_ends = user.get("discovery_ends")
    if not discovery_ends:
        return 0
    try:
        end_dt = datetime.fromisoformat(discovery_ends)
        remaining = (end_dt - datetime.now()).days
        return max(0, remaining)
    except (ValueError, TypeError):
        return 0


def is_discovery_active(user: Optional[Dict]) -> bool:
    """Check if user is currently on an active Discovery plan."""
    if not user or user.get("plan") != "discovery":
        return False
    return get_discovery_days_remaining(user) > 0


def has_used_discovery(user: Optional[Dict]) -> bool:
    """Check if user has already used their Discovery period (prevents re-activation)."""
    if not user:
        return False
    return bool(user.get("discovery_used", False))


def get_allowed_tickers(user: Optional[Dict]) -> list:
    """Get tickers this user can access."""
    plan = get_user_plan(user)
    tickers = plan.get("tickers", [])
    if tickers == "all":
        return ALL_TICKERS
    return tickers


def get_allowed_timeframes(user: Optional[Dict]) -> list:
    """Get timeframes this user can access."""
    plan = get_user_plan(user)
    return plan.get("timeframes", ["1day"])


def get_models_limit(user: Optional[Dict]) -> int:
    """Max models in ensemble for this user."""
    plan = get_user_plan(user)
    return plan.get("models_limit", 3)


def can_access_feature(user: Optional[Dict], feature: str) -> bool:
    """Check if user's plan includes a specific feature."""
    plan = get_user_plan(user)
    return plan.get("features", {}).get(feature, False)


def check_prediction_limit(user: Optional[Dict]) -> Tuple[bool, int, int]:
    """
    Check if user can make another prediction today.
    Returns: (allowed, used_today, daily_limit)
    """
    if not user:
        plan = PLANS["free"]
        return True, 0, plan["predictions_per_day"]  # Demo allows a few

    plan = get_user_plan(user)
    daily_limit = plan["predictions_per_day"]

    # Reset counter if new day
    today = datetime.now().strftime("%Y-%m-%d")
    if user.get("predictions_date") != today:
        user["predictions_today"] = 0
        user["predictions_date"] = today
        _update_user(user)

    used = user.get("predictions_today", 0)
    allowed = used < daily_limit
    return allowed, used, daily_limit


def record_prediction(user: Optional[Dict]):
    """Increment the user's prediction counter."""
    if not user:
        return

    today = datetime.now().strftime("%Y-%m-%d")
    if user.get("predictions_date") != today:
        user["predictions_today"] = 0
        user["predictions_date"] = today

    user["predictions_today"] = user.get("predictions_today", 0) + 1
    user["total_predictions"] = user.get("total_predictions", 0) + 1
    _update_user(user)


def get_plan_badge_info(user: Optional[Dict]) -> Dict:
    """Get display info for the plan badge."""
    plan = get_user_plan(user)
    plan_id = user.get("plan", "free") if user else "free"
    colors = {
        "free": {"bg": "rgba(100,116,139,0.12)", "border": "#64748b", "text": "#94a3b8"},
        "discovery": {"bg": "rgba(16,185,129,0.12)", "border": "#10b981", "text": "#34d399"},
        "starter": {"bg": "rgba(6,182,212,0.12)", "border": "#06b6d4", "text": "#22d3ee"},
        "professional": {"bg": "rgba(139,92,246,0.12)", "border": "#8b5cf6", "text": "#a78bfa"},
        "enterprise": {"bg": "rgba(245,158,11,0.12)", "border": "#f59e0b", "text": "#fcd34d"},
    }
    c = colors.get(plan_id, colors["free"])
    return {
        "plan_id": plan_id,
        "plan_name": plan["name"],
        "colors": c,
    }


# =============================================================================
# STRIPE CHECKOUT & BILLING
# =============================================================================

def create_checkout_session(user: Dict, plan_id: str, billing: str = "monthly") -> Optional[str]:
    """
    Create a Stripe Checkout session for subscription.
    Returns the checkout URL or None.
    """
    if not STRIPE_SECRET_KEY:
        logger.warning("Stripe not configured")
        return None

    plan = PLANS.get(plan_id)
    if not plan:
        return None

    price_key = "stripe_price_monthly" if billing == "monthly" else "stripe_price_yearly"
    price_id = plan.get(price_key)
    if not price_id:
        logger.warning(f"No Stripe price ID for {plan_id}/{billing}")
        return None

    try:
        session_params = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": f"{APP_DOMAIN}/payment-success?session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{APP_DOMAIN}/pricing",
            "customer_email": user.get("email"),
            "metadata": {
                "user_id": user.get("user_id"),
                "plan_id": plan_id,
            },
        }

        # Attach to existing Stripe customer if available
        if user.get("stripe_customer_id"):
            session_params["customer"] = user["stripe_customer_id"]
            del session_params["customer_email"]

        # Add trial period
        trial_days = plan.get("trial_days")
        if trial_days and not user.get("trial_started"):
            session_params["subscription_data"] = {
                "trial_period_days": trial_days,
            }

        session = stripe.checkout.Session.create(**session_params)
        logger.info(f"✅ Checkout session created for {user.get('email')} → {plan_id}")
        return session.url

    except Exception as e:
        logger.error(f"Stripe checkout error: {e}")
        return None


def create_customer_portal_session(user: Dict) -> Optional[str]:
    """Create a Stripe Customer Portal session for plan management."""
    if not STRIPE_SECRET_KEY or not user.get("stripe_customer_id"):
        return None

    try:
        session = stripe.billing_portal.Session.create(
            customer=user["stripe_customer_id"],
            return_url=f"{APP_DOMAIN}/",
        )
        return session.url
    except Exception as e:
        logger.error(f"Portal session error: {e}")
        return None


def handle_stripe_webhook(payload: bytes, sig_header: str) -> Dict:
    """
    Handle incoming Stripe webhook events.
    Call this from a Flask/Dash route.
    Returns {"status": "ok"} or {"error": "..."}.
    """
    if not STRIPE_WEBHOOK_SECRET:
        return {"error": "Webhook secret not configured"}

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
    except Exception as e:
        return {"error": f"Webhook verification failed: {e}"}

    event_type = event["type"]
    data = event["data"]["object"]

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(data)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data)
    elif event_type == "invoice.payment_failed":
        _handle_payment_failed(data)

    return {"status": "ok"}


def _handle_checkout_completed(session_data: Dict):
    """Handle successful checkout."""
    user_id = session_data.get("metadata", {}).get("user_id")
    plan_id = session_data.get("metadata", {}).get("plan_id")
    subscription_id = session_data.get("subscription")
    customer_id = session_data.get("customer")

    if user_id:
        user = get_user_by_id(user_id)
        if user:
            user["plan"] = plan_id or "starter"
            user["stripe_subscription_id"] = subscription_id
            user["stripe_customer_id"] = customer_id
            user["trial_started"] = datetime.now().isoformat()
            trial_days = PLANS.get(plan_id, {}).get("trial_days", 0)
            user["trial_ends"] = (datetime.now() + timedelta(days=trial_days)).isoformat() if trial_days else None
            _update_user(user)
            logger.info(f"✅ User {user['email']} upgraded to {plan_id}")


def _handle_subscription_updated(sub_data: Dict):
    """Handle subscription changes (upgrade/downgrade)."""
    customer_id = sub_data.get("customer")
    status = sub_data.get("status")

    # Find user by Stripe customer ID
    user = _find_user_by_stripe_customer(customer_id)
    if not user:
        return

    if status == "active":
        # Determine plan from price
        items = sub_data.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id")
            plan_id = _price_id_to_plan(price_id)
            if plan_id:
                user["plan"] = plan_id
                user["stripe_subscription_id"] = sub_data.get("id")
                _update_user(user)
                logger.info(f"✅ Subscription updated: {user['email']} → {plan_id}")
    elif status in ("past_due", "unpaid"):
        logger.warning(f"⚠️ Payment issue for {user['email']}: {status}")
    elif status == "canceled":
        user["plan"] = "free"
        user["stripe_subscription_id"] = None
        _update_user(user)
        logger.info(f"🔻 Subscription canceled: {user['email']} → free")


def _handle_subscription_deleted(sub_data: Dict):
    """Handle subscription cancellation."""
    customer_id = sub_data.get("customer")
    user = _find_user_by_stripe_customer(customer_id)
    if user:
        user["plan"] = "free"
        user["stripe_subscription_id"] = None
        _update_user(user)
        logger.info(f"🔻 Subscription deleted: {user['email']} → free")


def _handle_payment_failed(invoice_data: Dict):
    """Handle failed payment."""
    customer_id = invoice_data.get("customer")
    user = _find_user_by_stripe_customer(customer_id)
    if user:
        logger.warning(f"⚠️ Payment failed for {user['email']}")


def _find_user_by_stripe_customer(customer_id: str) -> Optional[Dict]:
    """Find user by Stripe customer ID."""
    if not customer_id:
        return None

    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            docs = _firestore_db.collection("users").where(
                "stripe_customer_id", "==", customer_id
            ).limit(1).stream()
            for doc in docs:
                return doc.to_dict()
        except Exception:
            pass

    users = _load_local_users()
    for u in users.values():
        if u.get("stripe_customer_id") == customer_id:
            return u

    return None


def _price_id_to_plan(price_id: str) -> Optional[str]:
    """Map a Stripe price ID back to a plan name."""
    for plan_id, plan in PLANS.items():
        if price_id in (plan.get("stripe_price_monthly"), plan.get("stripe_price_yearly")):
            return plan_id
    return None


# =============================================================================
# STRIPE WEBHOOK ROUTE (Flask-level, added to Dash server)
# =============================================================================

def register_webhook_route(server):
    """Register the /webhook/stripe route on the Flask server."""
    from flask import request, jsonify

    @server.route("/webhook/stripe", methods=["POST"])
    def stripe_webhook():
        payload = request.get_data()
        sig = request.headers.get("Stripe-Signature", "")
        result = handle_stripe_webhook(payload, sig)
        status_code = 200 if result.get("status") == "ok" else 400
        return jsonify(result), status_code

    logger.info("✅ Stripe webhook route registered at /webhook/stripe")


def reset_user_password(email: str, new_password: str) -> Dict:
    """Reset a user's password. Returns success/error dict."""
    email = email.strip().lower()

    if len(new_password) < 8:
        return {"error": "Password must be at least 8 characters"}

    user = get_user_by_email(email)
    if not user:
        # Don't reveal whether email exists for security
        return {"error": "If an account exists with this email, the password has been reset"}

    hashed_pw, salt = _hash_password(new_password)
    user["password_hash"] = hashed_pw
    user["password_salt"] = salt
    _update_user(user)

    logger.info(f"🔑 Password reset for {email}")
    return {"success": True, "message": "Password reset successfully. You can now sign in."}


# =============================================================================
# DASH UI — LOGIN PAGE
# =============================================================================

def _build_feature_card(icon, title, description, color="#06b6d4"):
    """Build a single feature card for the landing page."""
    return html.Div([
        html.Div(icon, style={
            "fontSize": "28px", "marginBottom": "12px",
            "width": "48px", "height": "48px", "borderRadius": "12px",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "background": f"linear-gradient(135deg, {color}15, {color}08)",
            "border": f"1px solid {color}20",
        }),
        html.H4(title, style={
            "color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700",
            "margin": "0 0 6px 0", "letterSpacing": "0.01em",
        }),
        html.P(description, style={
            "color": "#64748b", "fontSize": "12px", "margin": "0",
            "lineHeight": "1.5",
        }),
    ], style={
        "padding": "18px 16px", "borderRadius": "14px",
        "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(99,102,241,0.08)",
        "transition": "border-color 0.2s, transform 0.2s",
    })


def _build_stat_pill(value, label, color="#06b6d4"):
    """Build a stat pill for the landing hero section."""
    return html.Div([
        html.Div(value, style={
            "fontSize": "20px", "fontWeight": "800", "color": color,
            "letterSpacing": "-0.02em", "lineHeight": "1",
        }),
        html.Div(label, style={
            "fontSize": "10px", "color": "#64748b", "fontWeight": "600",
            "letterSpacing": "0.05em", "textTransform": "uppercase", "marginTop": "2px",
        }),
    ], style={
        "textAlign": "center", "padding": "12px 16px", "borderRadius": "12px",
        "background": "rgba(15,23,42,0.6)", "border": "1px solid rgba(99,102,241,0.08)",
        "flex": "1", "minWidth": "0",
    })


def _build_model_tag(name, color="#6366f1"):
    """Build a model architecture tag."""
    return html.Span(name, style={
        "fontSize": "10px", "fontWeight": "600", "letterSpacing": "0.03em",
        "color": color, "background": f"{color}12",
        "border": f"1px solid {color}25",
        "padding": "4px 10px", "borderRadius": "6px", "whiteSpace": "nowrap",
    })


def build_login_page(error_msg: str = "", success_msg: str = ""):
    """Build the landing page (left) + login/register panel (right) layout."""

    # ── LEFT SIDE: LANDING PAGE ─────────────────────────────────────────────
    landing_side = html.Div([
        # Scrollable inner container
        html.Div([

            # ── Brand header ──
            html.Div([
                html.Div([
                    html.Div("⚡", style={"fontSize": "28px"}),
                ], style={
                    "background": "linear-gradient(135deg, #06b6d4, #8b5cf6)",
                    "width": "48px", "height": "48px", "borderRadius": "14px",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "boxShadow": "0 4px 20px rgba(99,102,241,0.3)",
                }),
                html.Div([
                    html.Span("MarketLens", style={
                        "color": "#e2e8f0", "fontSize": "20px", "fontWeight": "800",
                        "letterSpacing": "-0.02em",
                    }),
                    html.Span(" AI", style={
                        "color": "#06b6d4", "fontSize": "20px", "fontWeight": "800",
                    }),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "14px",
                       "marginBottom": "32px"}),

            # ── Hero section ──
            html.Div([
                html.Div("📊 AI-POWERED TRADING RESEARCH & ANALYSIS PLATFORM", style={
                    "fontSize": "10px", "fontWeight": "700", "letterSpacing": "0.1em",
                    "color": "#06b6d4", "marginBottom": "12px",
                    "padding": "5px 14px", "borderRadius": "20px", "display": "inline-block",
                    "background": "rgba(6,182,212,0.08)", "border": "1px solid rgba(6,182,212,0.15)",
                }),
                html.P(
                    "Transform complex market data into clear, data-driven insights.",
                    style={
                        "color": "#e2e8f0", "fontSize": "clamp(1.1rem, 2.5vw, 1.5rem)",
                        "fontWeight": "700", "lineHeight": "1.3",
                        "margin": "0 0 12px 0", "maxWidth": "500px",
                    },
                ),
                html.P(
                    "MarketLens AI is designed for traders, analysts, and learners who want to "
                    "explore financial markets using advanced analytical tools — without relying on guesswork.",
                    style={
                        "color": "#94a3b8", "fontSize": "13px", "lineHeight": "1.6",
                        "margin": "0 0 24px 0", "maxWidth": "500px",
                    },
                ),
            ]),

            # ── Stats row ──
            html.Div([
                _build_stat_pill("8", "AI Models", "#8b5cf6"),
                _build_stat_pill("13+", "Tickers", "#06b6d4"),
                _build_stat_pill("24/7", "Analysis", "#10b981"),
                _build_stat_pill("99.9%", "Uptime", "#f59e0b"),
            ], style={
                "display": "flex", "gap": "10px", "marginBottom": "28px",
                "flexWrap": "wrap",
            }),

            # ── Model architectures ──
            html.Div([
                html.Div("Neural Network Ensemble", style={
                    "color": "#94a3b8", "fontSize": "11px", "fontWeight": "600",
                    "letterSpacing": "0.05em", "textTransform": "uppercase",
                    "marginBottom": "10px",
                }),
                html.Div([
                    _build_model_tag("Transformer", "#8b5cf6"),
                    _build_model_tag("CNN-LSTM", "#06b6d4"),
                    _build_model_tag("TCN", "#10b981"),
                    _build_model_tag("Informer", "#ec4899"),
                    _build_model_tag("N-BEATS", "#f59e0b"),
                    _build_model_tag("LSTM-GRU", "#6366f1"),
                    _build_model_tag("XGBoost", "#ef4444"),
                    _build_model_tag("Sklearn", "#14b8a6"),
                ], style={
                    "display": "flex", "flexWrap": "wrap", "gap": "6px",
                }),
            ], style={"marginBottom": "28px"}),

            # ── What You Can Do ──
            html.Div([
                html.Div([
                    html.Span("🧠 ", style={"fontSize": "14px"}),
                    html.Span("What You Can Do", style={
                        "color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700",
                    }),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "14px"}),
                html.Div([
                    _build_feature_card(
                        "📊", "Analyze Market Behavior",
                        "Use advanced AI models to study price action, momentum, and volatility across multiple asset classes.",
                        "#8b5cf6",
                    ),
                    _build_feature_card(
                        "🔄", "Multi-Model Perspectives",
                        "Explore insights from 8 different model architectures — each capturing different market dynamics and patterns.",
                        "#06b6d4",
                    ),
                    _build_feature_card(
                        "📈", "Backtest Strategies",
                        "Validate trading ideas with walk-forward backtesting — Sharpe ratio, drawdown analysis, and equity curves.",
                        "#10b981",
                    ),
                    _build_feature_card(
                        "🔍", "Interpretable Insights",
                        "Understand model outputs with SHAP analysis — see exactly which features drive each analytical result.",
                        "#f59e0b",
                    ),
                    _build_feature_card(
                        "🌊", "Market Regime Analysis",
                        "Study structural market shifts — identify bull, bear, and sideways regimes using Hidden Markov Models.",
                        "#ec4899",
                    ),
                    _build_feature_card(
                        "⏱️", "Multi-Timeframe Study",
                        "Analyze 15min to daily timeframes simultaneously and observe how signals align across different horizons.",
                        "#6366f1",
                    ),
                ], style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(200px, 1fr))",
                    "gap": "12px",
                }),
            ], style={"marginBottom": "28px"}),

            # ── Built for Research & Understanding ──
            html.Div([
                html.Div([
                    html.Span("🔬 ", style={"fontSize": "14px"}),
                    html.Span("Built for Research & Understanding", style={
                        "color": "#e2e8f0", "fontSize": "14px", "fontWeight": "700",
                    }),
                ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "10px"}),
                html.P(
                    "Powered by a combination of modern machine learning techniques, this platform "
                    "enables users to evaluate strategies objectively and gain deeper insights into "
                    "financial data — all in a transparent, research-driven environment.",
                    style={
                        "color": "#94a3b8", "fontSize": "12px", "lineHeight": "1.6",
                        "margin": "0 0 12px 0",
                    },
                ),
                html.Div([
                    html.Div([
                        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
                        html.Span("Evaluate strategies with real data — not guesswork", style={
                            "color": "#94a3b8", "fontSize": "12px",
                        }),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
                        html.Span("Gain deeper insights into market structure and dynamics", style={
                            "color": "#94a3b8", "fontSize": "12px",
                        }),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
                        html.Span("Understand why models produce their outputs — full transparency", style={
                            "color": "#94a3b8", "fontSize": "12px",
                        }),
                    ], style={"marginBottom": "6px"}),
                    html.Div([
                        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
                        html.Span("13+ global instruments: crypto, forex, indices, and commodities", style={
                            "color": "#94a3b8", "fontSize": "12px",
                        }),
                    ]),
                ], style={
                    "padding": "14px 16px", "borderRadius": "12px",
                    "background": "rgba(16,185,129,0.04)", "border": "1px solid rgba(16,185,129,0.08)",
                }),
            ], style={"marginBottom": "28px"}),

            # ── Plan highlights ──
            html.Div([
                html.Div("Plans For Every Trader", style={
                    "color": "#94a3b8", "fontSize": "11px", "fontWeight": "600",
                    "letterSpacing": "0.05em", "textTransform": "uppercase",
                    "marginBottom": "14px",
                }),
                html.Div([
                    # Discovery
                    html.Div([
                        html.Div([
                            html.Span("🎁", style={"fontSize": "18px"}),
                            html.Span("Discovery", style={
                                "color": "#34d399", "fontSize": "14px", "fontWeight": "700",
                            }),
                            html.Span("FREE", style={
                                "fontSize": "9px", "fontWeight": "700", "letterSpacing": "0.08em",
                                "color": "#34d399", "background": "rgba(16,185,129,0.12)",
                                "padding": "2px 8px", "borderRadius": "4px",
                            }),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                        html.P("14 days free on signup — 4 models, 5 analyses/day, backtesting & SHAP", style={
                            "color": "#64748b", "fontSize": "12px", "margin": "0", "lineHeight": "1.4",
                        }),
                    ], style={
                        "padding": "14px 16px", "borderRadius": "12px",
                        "background": "rgba(16,185,129,0.04)", "border": "1px solid rgba(16,185,129,0.1)",
                    }),
                    # Starter
                    html.Div([
                        html.Div([
                            html.Span("⚡", style={"fontSize": "18px"}),
                            html.Span("Starter", style={
                                "color": "#06b6d4", "fontSize": "14px", "fontWeight": "700",
                            }),
                            html.Span("€39/mo", style={
                                "fontSize": "12px", "fontWeight": "700", "color": "#06b6d4",
                            }),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                        html.P("10 analyses/day, drift alerts, multi-TF analysis, 7 tickers", style={
                            "color": "#64748b", "fontSize": "12px", "margin": "0", "lineHeight": "1.4",
                        }),
                    ], style={
                        "padding": "14px 16px", "borderRadius": "12px",
                        "background": "rgba(6,182,212,0.04)", "border": "1px solid rgba(6,182,212,0.1)",
                    }),
                    # Professional
                    html.Div([
                        html.Div([
                            html.Span("🔥", style={"fontSize": "18px"}),
                            html.Span("Professional", style={
                                "color": "#8b5cf6", "fontSize": "14px", "fontWeight": "700",
                            }),
                            html.Span("€89/mo", style={
                                "fontSize": "12px", "fontWeight": "700", "color": "#8b5cf6",
                            }),
                            html.Span("POPULAR", style={
                                "fontSize": "8px", "fontWeight": "700", "letterSpacing": "0.08em",
                                "color": "#fff", "background": "linear-gradient(135deg, #8b5cf6, #6366f1)",
                                "padding": "2px 8px", "borderRadius": "4px",
                            }),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                        html.P("Full 8-model ensemble, 25 analyses/day, all tickers, risk dashboard, model training", style={
                            "color": "#64748b", "fontSize": "12px", "margin": "0", "lineHeight": "1.4",
                        }),
                    ], style={
                        "padding": "14px 16px", "borderRadius": "12px",
                        "background": "rgba(139,92,246,0.04)", "border": "1px solid rgba(139,92,246,0.12)",
                    }),
                    # Enterprise
                    html.Div([
                        html.Div([
                            html.Span("🏛️", style={"fontSize": "18px"}),
                            html.Span("Enterprise", style={
                                "color": "#f59e0b", "fontSize": "14px", "fontWeight": "700",
                            }),
                            html.Span("Custom", style={
                                "fontSize": "12px", "fontWeight": "700", "color": "#f59e0b",
                            }),
                        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "6px"}),
                        html.P("Unlimited analyses, API access, Monte Carlo, portfolio optimization, white-label", style={
                            "color": "#64748b", "fontSize": "12px", "margin": "0", "lineHeight": "1.4",
                        }),
                    ], style={
                        "padding": "14px 16px", "borderRadius": "12px",
                        "background": "rgba(245,158,11,0.04)", "border": "1px solid rgba(245,158,11,0.1)",
                    }),
                ], style={"display": "flex", "flexDirection": "column", "gap": "8px"}),
            ], style={"marginBottom": "28px"}),

            # ── Trust footer ──
            html.Div([
                html.Div(style={
                    "height": "1px", "margin": "0 0 16px 0",
                    "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.12), transparent)",
                }),
                html.Div([
                    html.Span("🔒 Secure payments via Stripe", style={"color": "#475569", "fontSize": "11px"}),
                    html.Span("  •  ", style={"color": "#334155"}),
                    html.Span("Cancel anytime", style={"color": "#475569", "fontSize": "11px"}),
                    html.Span("  •  ", style={"color": "#334155"}),
                    html.Span("14-day money-back guarantee", style={"color": "#475569", "fontSize": "11px"}),
                ], style={"textAlign": "center"}),
                html.Div([
                    html.Span("Built with ", style={"color": "#334155", "fontSize": "11px"}),
                    html.Span("PyTorch", style={"color": "#475569", "fontSize": "11px", "fontWeight": "600"}),
                    html.Span(" • ", style={"color": "#334155", "fontSize": "11px"}),
                    html.Span("TensorFlow", style={"color": "#475569", "fontSize": "11px", "fontWeight": "600"}),
                    html.Span(" • ", style={"color": "#334155", "fontSize": "11px"}),
                    html.Span("Plotly Dash", style={"color": "#475569", "fontSize": "11px", "fontWeight": "600"}),
                    html.Span(" • ", style={"color": "#334155", "fontSize": "11px"}),
                    html.Span("Google Cloud", style={"color": "#475569", "fontSize": "11px", "fontWeight": "600"}),
                ], style={"textAlign": "center", "marginTop": "8px"}),
            ], style={"paddingBottom": "20px"}),

        ], style={
            "padding": "40px 36px",
            "overflowY": "auto", "maxHeight": "100vh",
            "scrollbarWidth": "thin", "scrollbarColor": "rgba(99,102,241,0.2) transparent",
        }),
    ], style={
        "flex": "1", "minWidth": "0",
        "display": "flex", "flexDirection": "column",
    }, className="landing-content-side")

    # ── RIGHT SIDE: LOGIN / REGISTER PANEL ──────────────────────────────────
    login_side = html.Div([
        html.Div([
            # Logo (compact for right panel)
            html.Div([
                html.Div("⚡", style={"fontSize": "30px"}),
            ], style={
                "background": "linear-gradient(135deg, #06b6d4, #8b5cf6)",
                "width": "56px", "height": "56px", "borderRadius": "16px",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "margin": "0 auto 16px", "boxShadow": "0 8px 30px rgba(99,102,241,0.3)",
            }),
            html.H2([
                html.Span("MarketLens", style={"color": "#e2e8f0"}),
                html.Span(" AI", style={"color": "#06b6d4"}),
            ], style={
                "textAlign": "center", "fontSize": "1.3rem", "fontWeight": "800",
                "margin": "0 0 4px 0",
            }),
            html.P([
                html.Span("AI-Powered", style={"color": "#a78bfa", "fontWeight": "600"}),
                html.Span(" Market Intelligence", style={"color": "#64748b"}),
            ], style={
                "textAlign": "center", "fontSize": "12px", "margin": "0 0 24px 0",
            }),

            # Error message
            html.Div(error_msg, id="auth-error", style={
                "color": "#ef4444", "fontSize": "13px", "textAlign": "center",
                "marginBottom": "12px", "display": "block" if error_msg else "none",
                "background": "rgba(239,68,68,0.08)", "padding": "8px 12px",
                "borderRadius": "8px", "border": "1px solid rgba(239,68,68,0.2)",
            }),

            # Success message
            html.Div(success_msg, id="auth-success", style={
                "color": "#10b981", "fontSize": "13px", "textAlign": "center",
                "marginBottom": "12px", "display": "block" if success_msg else "none",
                "background": "rgba(16,185,129,0.08)", "padding": "8px 12px",
                "borderRadius": "8px", "border": "1px solid rgba(16,185,129,0.2)",
            }),

            # Tab toggle: Login / Register
            html.Div([
                html.Button("Sign In", id="auth-tab-login", n_clicks=0, style={
                    "flex": "1", "padding": "10px", "border": "none", "borderRadius": "8px 0 0 8px",
                    "background": "rgba(99,102,241,0.15)", "color": "#a78bfa",
                    "fontSize": "13px", "fontWeight": "700", "cursor": "pointer",
                }),
                html.Button("Create Account", id="auth-tab-register", n_clicks=0, style={
                    "flex": "1", "padding": "10px", "border": "none", "borderRadius": "0 8px 8px 0",
                    "background": "rgba(30,41,59,0.5)", "color": "#64748b",
                    "fontSize": "13px", "fontWeight": "600", "cursor": "pointer",
                }),
            ], style={"display": "flex", "marginBottom": "24px", "borderRadius": "8px",
                       "border": "1px solid rgba(99,102,241,0.15)"}),

            # Login form
            html.Div([
                html.Div([
                    html.Label("Email", style={"color": "#94a3b8", "fontSize": "12px",
                                                "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="login-email", type="email", placeholder="you@example.com",
                              style=_input_style()),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("Password", style={"color": "#94a3b8", "fontSize": "12px",
                                                   "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="login-password", type="password", placeholder="••••••••",
                              style=_input_style()),
                ], style={"marginBottom": "8px"}),
                # Forgot password link
                html.Div([
                    html.A("Forgot your password?", id="forgot-password-link", href="#",
                           n_clicks=0, style={
                        "color": "#6366f1", "fontSize": "12px", "textDecoration": "none",
                        "fontWeight": "500",
                    }),
                ], style={"textAlign": "right", "marginBottom": "20px"}),
                html.Button("Sign In →", id="login-btn", n_clicks=0, style=_primary_btn_style()),
            ], id="login-form"),

            # Register form (hidden by default)
            html.Div([
                html.Div([
                    html.Label("Full Name", style={"color": "#94a3b8", "fontSize": "12px",
                                                    "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="register-name", type="text", placeholder="Your name",
                              style=_input_style()),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("Email", style={"color": "#94a3b8", "fontSize": "12px",
                                                "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="register-email", type="email", placeholder="you@example.com",
                              style=_input_style()),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("Password", style={"color": "#94a3b8", "fontSize": "12px",
                                                   "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="register-password", type="password", placeholder="Min 8 characters",
                              style=_input_style()),
                ], style={"marginBottom": "24px"}),
                html.Button("Create Account →", id="register-btn", n_clicks=0, style=_primary_btn_style()),
            ], id="register-form", style={"display": "none"}),

            # Forgot password form (hidden by default)
            html.Div([
                html.Div([
                    html.Div("🔐", style={"fontSize": "28px", "marginBottom": "8px"}),
                    html.H3("Reset Password", style={"color": "#e2e8f0", "margin": "0 0 4px 0",
                                                       "fontSize": "1.1rem", "fontWeight": "700"}),
                    html.P("Enter your email and new password below", style={
                        "color": "#64748b", "fontSize": "12px", "margin": "0 0 20px 0",
                    }),
                ], style={"textAlign": "center"}),
                html.Div([
                    html.Label("Email", style={"color": "#94a3b8", "fontSize": "12px",
                                                "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="reset-email", type="email", placeholder="you@example.com",
                              style=_input_style()),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("New Password", style={"color": "#94a3b8", "fontSize": "12px",
                                                       "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="reset-password", type="password", placeholder="Min 8 characters",
                              style=_input_style()),
                ], style={"marginBottom": "16px"}),
                html.Div([
                    html.Label("Confirm New Password", style={"color": "#94a3b8", "fontSize": "12px",
                                                               "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="reset-password-confirm", type="password", placeholder="Confirm password",
                              style=_input_style()),
                ], style={"marginBottom": "24px"}),
                html.Button("Reset Password →", id="reset-password-btn", n_clicks=0, style=_primary_btn_style()),
                html.Div([
                    html.A("← Back to Sign In", id="back-to-login-link", href="#",
                           n_clicks=0, style={
                        "color": "#6366f1", "fontSize": "12px", "textDecoration": "none",
                        "fontWeight": "600",
                    }),
                ], style={"textAlign": "center", "marginTop": "16px"}),
            ], id="forgot-form", style={"display": "none"}),

            # Divider
            html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                            "margin": "24px 0"}),

            # Demo mode link
            html.Div([
                html.Span("Or ", style={"color": "#475569", "fontSize": "13px"}),
                html.A("try the free demo →", id="demo-mode-link", href="#", style={
                    "color": "#6366f1", "fontSize": "13px", "fontWeight": "600",
                    "textDecoration": "none",
                }),
            ], style={"textAlign": "center", "marginBottom": "10px"}),

            # Discovery promo badge
            html.Div([
                html.Div([
                    html.Span("🎁 ", style={"fontSize": "16px"}),
                    html.Span("Create account → 14 days Discovery FREE", style={
                        "color": "#34d399", "fontSize": "12px", "fontWeight": "600",
                    }),
                ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "4px"}),
                html.Div("4 AI models • 5 analyses/day • Backtesting • SHAP",
                         style={"color": "#64748b", "fontSize": "11px", "textAlign": "center", "marginTop": "4px"}),
            ], style={
                "padding": "10px 16px", "borderRadius": "10px", "marginBottom": "12px",
                "background": "rgba(16,185,129,0.06)", "border": "1px solid rgba(16,185,129,0.12)",
            }),

            # Footer links
            html.Div([
                html.A("Terms of Service", href="#", style={
                    "color": "#475569", "fontSize": "11px", "textDecoration": "none",
                }),
                html.Span(" • ", style={"color": "#334155", "fontSize": "12px"}),
                html.A("Privacy Policy", href="#", style={
                    "color": "#475569", "fontSize": "11px", "textDecoration": "none",
                }),
            ], style={"textAlign": "center", "marginTop": "4px"}),

        ], style={
            "maxWidth": "400px", "width": "100%", "padding": "32px 28px",
            "background": "rgba(15,23,42,0.85)", "backdropFilter": "blur(24px)",
            "border": "1px solid rgba(99,102,241,0.12)", "borderRadius": "20px",
            "boxShadow": "0 20px 60px rgba(0,0,0,0.5)",
        }),
    ], style={
        "width": "440px", "minWidth": "380px", "flexShrink": "0",
        "display": "flex", "alignItems": "center", "justifyContent": "center",
        "padding": "24px 28px",
        "borderLeft": "1px solid rgba(99,102,241,0.06)",
        "background": "rgba(8,12,28,0.4)",
        "overflowY": "auto", "maxHeight": "100vh",
    }, className="landing-login-panel")

    # ── COMBINED LAYOUT ─────────────────────────────────────────────────────
    return html.Div([
        landing_side,
        login_side,
    ], style={
        "background": "#0a0e1a", "minHeight": "100vh",
        "display": "flex", "flexDirection": "row",
        "fontFamily": "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    }, className="landing-wrapper")


# =============================================================================
# DASH UI — PRICING PAGE
# =============================================================================

def build_pricing_page(current_plan: str = "free"):
    """Build the pricing/upgrade page."""

    # Discovery info banner (shown if user is on Discovery or free-after-discovery)
    discovery_banner = html.Div([
        html.Div([
            html.Span("🎁 ", style={"fontSize": "20px"}),
            html.Span("Discovery Plan — Included Free With Every Account", style={
                "color": "#34d399", "fontSize": "15px", "fontWeight": "700",
            }),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "center", "gap": "6px",
                   "marginBottom": "8px"}),
        html.Div(
            f"14 days of enhanced access: 4 AI models, 5 predictions/day, backtesting, SHAP explanations, "
            f"regime detection, 7 tickers, and 3 timeframes — all free upon registration.",
            style={"color": "#94a3b8", "fontSize": "13px", "textAlign": "center", "maxWidth": "600px",
                   "margin": "0 auto", "lineHeight": "1.5"},
        ),
    ], style={
        "padding": "20px 24px", "borderRadius": "14px", "marginBottom": "28px",
        "background": "rgba(16,185,129,0.05)", "border": "1px solid rgba(16,185,129,0.12)",
        "maxWidth": "700px", "margin": "0 auto 28px auto",
    })

    plans_display = [
        ("starter", "⚡", "Starter", "€39", "€29", "For retail traders getting started", "#06b6d4", [
            "4-model AI ensemble", "10 predictions/day", "3 timeframes (1H, 4H, 1D)",
            "7 tickers (crypto, forex, stocks, gold)", "Backtesting",
            "SHAP explanations", "Regime detection",
            "Drift alerts", "Multi-TF analysis", "Email support",
        ]),
        ("professional", "🔥", "Professional", "€89", "€69", "Full AI power for serious traders", "#8b5cf6", [
            "Full 8-model ensemble", "25 predictions/day", "All timeframes (15m–1D)",
            "All 13+ tickers", "Walk-forward backtesting",
            "SHAP explanations", "Regime detection & drift alerts",
            "Multi-TF consensus analysis", "FTMO dashboard",
            "Model training", "Priority support",
        ]),
    ]

    cards = []
    for pid, emoji, name, m_price, y_price, tagline, color, features in plans_display:
        is_current = pid == current_plan
        is_popular = pid == "professional"

        card = html.Div([
            # Popular badge
            html.Div("MOST POPULAR", style={
                "position": "absolute", "top": "-12px", "left": "50%", "transform": "translateX(-50%)",
                "background": f"linear-gradient(135deg, {color}, #8b5cf6)", "color": "#fff",
                "padding": "4px 16px", "borderRadius": "20px", "fontSize": "10px",
                "fontWeight": "700", "letterSpacing": "1px",
                "display": "block" if is_popular else "none",
            }),
            html.Div(emoji, style={"fontSize": "32px", "marginBottom": "8px"}),
            html.H3(name, style={"color": "#e2e8f0", "margin": "0 0 4px 0", "fontSize": "1.3rem"}),
            html.P(tagline, style={"color": "#64748b", "fontSize": "12px", "margin": "0 0 16px 0"}),
            html.Div([
                html.Span(m_price, style={"fontSize": "2.2rem", "fontWeight": "800", "color": color}),
                html.Span("/mo", style={"color": "#64748b", "fontSize": "14px"}),
            ]),
            html.Div(f"or {y_price}/mo billed yearly (save ~25%)", style={
                "color": "#10b981", "fontSize": "11px", "marginBottom": "16px",
            }),
            html.Div(style={"borderTop": "1px solid rgba(99,102,241,0.12)", "paddingTop": "14px"}),
            *[html.Div([
                html.Span("✓ ", style={"color": color}),
                html.Span(f, style={"color": "#94a3b8", "fontSize": "13px"}),
            ], style={"marginBottom": "6px"}) for f in features],
            html.Button(
                "Current Plan" if is_current else f"Upgrade to {name}",
                id={"type": "upgrade-btn", "index": pid},
                n_clicks=0,
                disabled=is_current,
                style={
                    "width": "100%", "padding": "12px", "borderRadius": "10px",
                    "border": "none", "marginTop": "20px", "fontSize": "14px",
                    "fontWeight": "700", "cursor": "pointer" if not is_current else "default",
                    "background": f"linear-gradient(135deg, {color}, #8b5cf6)" if not is_current else "rgba(99,102,241,0.08)",
                    "color": "#fff" if not is_current else "#475569",
                    "opacity": "0.5" if is_current else "1",
                },
            ),
        ], style={
            "background": "rgba(15,23,42,0.6)" if not is_popular else f"linear-gradient(135deg, rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.08), rgba(15,23,42,0.8))",
            "border": f"1px solid {color if is_popular else 'rgba(99,102,241,0.12)'}",
            "borderRadius": "16px", "padding": "28px 24px", "position": "relative",
            "boxShadow": f"0 0 40px rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.1)" if is_popular else "none",
        })
        cards.append(card)

    # Enterprise card — Contact Support (no price, no Stripe checkout)
    enterprise_color = "#f59e0b"
    enterprise_features = [
        "Everything in Professional",
        "Unlimited predictions",
        "Portfolio optimization (Black-Litterman)",
        "Monte Carlo simulation",
        "Custom tickers on request",
        "REST API access",
        "Custom system development",
        "Dedicated support & onboarding",
        "White-label options",
    ]
    enterprise_card = html.Div([
        html.Div(html.Span("CUSTOM", style={
            "fontSize": "10px", "fontWeight": "700", "letterSpacing": "1px",
            "color": "#fcd34d", "background": "rgba(245,158,11,0.15)",
            "padding": "4px 14px", "borderRadius": "20px",
            "border": "1px solid rgba(245,158,11,0.3)",
        }), style={"textAlign": "center", "marginBottom": "12px"}),
        html.Div("🏛️", style={"fontSize": "32px", "marginBottom": "8px"}),
        html.H3("Enterprise", style={"color": "#e2e8f0", "margin": "0 0 4px 0", "fontSize": "1.3rem"}),
        html.P("For prop firms, funds & custom development", style={
            "color": "#64748b", "fontSize": "12px", "margin": "0 0 16px 0",
        }),
        html.Div([
            html.Span("Custom", style={"fontSize": "2.2rem", "fontWeight": "800", "color": enterprise_color}),
        ]),
        html.Div("Tailored pricing based on your needs", style={
            "color": "#94a3b8", "fontSize": "11px", "marginBottom": "16px",
        }),
        html.Div(style={"borderTop": "1px solid rgba(99,102,241,0.12)", "paddingTop": "14px"}),
        *[html.Div([
            html.Span("✓ ", style={"color": enterprise_color}),
            html.Span(f, style={"color": "#94a3b8", "fontSize": "13px"}),
        ], style={"marginBottom": "6px"}) for f in enterprise_features],
        html.A(
            "📧 Contact Support",
            href="mailto:itubusinesshub@gmail.com?subject=MarketLens%20AI%20-%20Enterprise%20Inquiry",
            style={
                "display": "block", "width": "100%", "padding": "12px", "borderRadius": "10px",
                "border": "none", "marginTop": "20px", "fontSize": "14px",
                "fontWeight": "700", "cursor": "pointer", "textAlign": "center",
                "background": f"linear-gradient(135deg, {enterprise_color}, #d97706)",
                "color": "#fff", "textDecoration": "none", "boxSizing": "border-box",
            },
        ),
    ], style={
        "background": "rgba(15,23,42,0.6)",
        "border": "1px solid rgba(245,158,11,0.2)",
        "borderRadius": "16px", "padding": "28px 24px", "position": "relative",
    })
    cards.append(enterprise_card)

    return html.Div([
        html.H2("Choose Your Plan", style={"textAlign": "center", "color": "#e2e8f0", "marginBottom": "8px"}),
        html.P("All plans include a free trial. Cancel anytime.", style={
            "textAlign": "center", "color": "#64748b", "fontSize": "14px", "marginBottom": "24px",
        }),
        discovery_banner,
        html.Div(cards, style={
            "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
            "gap": "20px", "maxWidth": "1000px", "margin": "0 auto",
        }),
        # Contact support & FAQ footer
        html.Div([
            html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent)",
                            "margin": "32px 0 20px 0"}),
            html.Div([
                html.Div("💎", style={"fontSize": "20px", "marginBottom": "8px"}),
                html.P("Need a custom plan or have questions?", style={
                    "color": "#94a3b8", "fontSize": "14px", "margin": "0 0 12px 0",
                }),
                html.A("📧 Contact Sales — itubusinesshub@gmail.com",
                       href="mailto:itubusinesshub@gmail.com?subject=MarketLens%20AI%20-%20Plan%20Inquiry",
                       style={
                    "color": "#06b6d4", "fontSize": "13px", "fontWeight": "600",
                    "textDecoration": "none", "display": "inline-block",
                    "padding": "8px 20px", "borderRadius": "8px",
                    "border": "1px solid rgba(6,182,212,0.3)",
                    "background": "rgba(6,182,212,0.08)",
                }),
            ], style={"textAlign": "center"}),
            html.Div([
                html.Span("🔒 Secure payments via Stripe", style={"color": "#475569", "fontSize": "11px"}),
                html.Span(" • ", style={"color": "#334155"}),
                html.Span("Cancel anytime", style={"color": "#475569", "fontSize": "11px"}),
                html.Span(" • ", style={"color": "#334155"}),
                html.Span("14-day money-back guarantee", style={"color": "#475569", "fontSize": "11px"}),
            ], style={"textAlign": "center", "marginTop": "16px"}),
        ]),
    ])


# =============================================================================
# DASH UI — USER ACCOUNT BADGE (sidebar)
# =============================================================================

def build_user_badge(user: Optional[Dict]):
    """Build the user info badge for the sidebar."""
    if not user:
        return html.Div([
            html.Button("Sign In", id="sidebar-login-btn", n_clicks=0, style={
                "width": "100%", "padding": "10px", "borderRadius": "10px",
                "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                "color": "#fff", "border": "none", "fontSize": "13px",
                "fontWeight": "700", "cursor": "pointer",
            }),
        ], style={"marginBottom": "16px"})

    badge = get_plan_badge_info(user)
    plan_c = badge["colors"]
    allowed, used, limit = check_prediction_limit(user)

    return html.Div([
        # Discovery countdown banner (only shown during Discovery period)
        *([html.Div([
            html.Div([
                html.Span("🎁 ", style={"fontSize": "14px"}),
                html.Span(f"Discovery: {get_discovery_days_remaining(user)} days left",
                           style={"color": "#34d399", "fontSize": "11px", "fontWeight": "700"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "4px", "marginBottom": "4px"}),
            html.Div(
                "Upgrade now to keep backtesting & advanced analytics"
                if get_discovery_days_remaining(user) <= 5
                else "Enjoy your full Discovery access!",
                style={"color": "#64748b", "fontSize": "10px"}
            ),
            # Progress bar showing time remaining
            html.Div([
                html.Div(style={
                    "width": f"{max(5, get_discovery_days_remaining(user) / DISCOVERY_DURATION_DAYS * 100):.0f}%",
                    "height": "3px", "borderRadius": "2px",
                    "background": "linear-gradient(90deg, #10b981, #34d399)",
                    "transition": "width 0.3s",
                }),
            ], style={"height": "3px", "background": "rgba(16,185,129,0.1)", "borderRadius": "2px",
                       "marginTop": "6px"}),
        ], style={
            "padding": "10px 12px", "borderRadius": "10px", "marginBottom": "10px",
            "background": "rgba(16,185,129,0.06)", "border": "1px solid rgba(16,185,129,0.15)",
        })] if user.get("plan") == "discovery" else []),

        # User info row
        html.Div([
            html.Div(user.get("name", "User")[0].upper(), style={
                "width": "32px", "height": "32px", "borderRadius": "50%",
                "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                "display": "flex", "alignItems": "center", "justifyContent": "center",
                "color": "#fff", "fontSize": "14px", "fontWeight": "700", "flexShrink": "0",
            }),
            html.Div([
                html.Div(user.get("name", "User"), style={
                    "color": "#e2e8f0", "fontSize": "13px", "fontWeight": "600",
                    "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap",
                }),
                html.Span(badge["plan_name"].upper(), style={
                    "fontSize": "9px", "fontWeight": "700", "letterSpacing": "1px",
                    "color": plan_c["text"], "background": plan_c["bg"],
                    "border": f"1px solid {plan_c['border']}",
                    "padding": "1px 8px", "borderRadius": "6px",
                }),
            ], style={"flex": "1", "minWidth": "0"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "10px"}),

        # Usage bar
        html.Div([
            html.Div([
                html.Span(f"{used}/{limit}", style={"color": "#94a3b8", "fontSize": "10px"}),
                html.Span("predictions today", style={"color": "#475569", "fontSize": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
            html.Div([
                html.Div(style={
                    "width": f"{min(used / limit * 100, 100):.0f}%" if limit > 0 else "0%",
                    "height": "3px", "borderRadius": "2px",
                    "background": "#10b981" if used < limit * 0.8 else "#f59e0b" if used < limit else "#ef4444",
                    "transition": "width 0.3s",
                }),
            ], style={"height": "3px", "background": "rgba(99,102,241,0.1)", "borderRadius": "2px"}),
        ], style={"marginBottom": "10px"}),

        # Upgrade / Manage buttons
        html.Div([
            html.Button("Upgrade" if user.get("plan") != "enterprise" else "Manage",
                         id="sidebar-upgrade-btn", n_clicks=0, style={
                "flex": "1", "padding": "6px", "borderRadius": "6px", "border": "none",
                "background": "rgba(99,102,241,0.12)", "color": "#a78bfa",
                "fontSize": "11px", "fontWeight": "600", "cursor": "pointer",
            }),
            html.Button("Sign Out", id="sidebar-logout-btn", n_clicks=0, style={
                "flex": "1", "padding": "6px", "borderRadius": "6px", "border": "none",
                "background": "rgba(239,68,68,0.08)", "color": "#f87171",
                "fontSize": "11px", "fontWeight": "600", "cursor": "pointer",
            }),
        ], style={"display": "flex", "gap": "8px"}),
        # Hidden contact button — keeps callback wired (visible contact is in footer)
        html.Button("", id="open-contact-btn-pricing", n_clicks=0,
                    style={"display": "none"}),
    ], style={
        "marginBottom": "16px", "padding": "14px",
        "background": "rgba(15,23,42,0.4)", "borderRadius": "12px",
        "border": "1px solid rgba(99,102,241,0.1)",
    })


# =============================================================================
# DASH UI — UPGRADE PROMPT (shown when feature is gated)
# =============================================================================

def build_upgrade_prompt(feature_name: str, required_plan: str = "professional"):
    """Build a UI prompt when user tries to access a gated feature."""
    plan = PLANS.get(required_plan, PLANS["professional"])
    price = plan.get("price_display", {}).get("monthly", 89) if plan.get("price_display") else None

    price_text = f" (€{price}/mo)" if price else ""

    return html.Div([
        html.Div("🔒", style={"fontSize": "48px", "marginBottom": "16px"}),
        html.H3(f"{feature_name} requires {plan['name']}",
                 style={"color": "#e2e8f0", "fontWeight": "700", "margin": "0 0 8px 0"}),
        html.P(f"Upgrade to the {plan['name']} plan{price_text} to unlock this feature.",
               style={"color": "#64748b", "fontSize": "14px", "margin": "0 0 20px 0"}),
        html.Button(f"Upgrade to {plan['name']} →", id="prompt-upgrade-btn", n_clicks=0, style={
            "padding": "12px 28px", "borderRadius": "10px", "border": "none",
            "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
            "color": "#fff", "fontSize": "14px", "fontWeight": "700", "cursor": "pointer",
        }),
    ], style={
        "textAlign": "center", "padding": "60px 20px",
        "background": "rgba(15,23,42,0.6)", "borderRadius": "16px",
        "border": "1px solid rgba(99,102,241,0.12)",
    })


def build_limit_reached_prompt(used: int, limit: int):
    """Build a UI prompt when daily prediction limit is reached."""
    return html.Div([
        html.Div("⏳", style={"fontSize": "48px", "marginBottom": "16px"}),
        html.H3("Daily Prediction Limit Reached",
                 style={"color": "#e2e8f0", "fontWeight": "700", "margin": "0 0 8px 0"}),
        html.P(f"You've used {used} of {limit} predictions today. Limits reset at midnight UTC.",
               style={"color": "#64748b", "fontSize": "14px", "margin": "0 0 8px 0"}),
        html.P("Upgrade your plan for more daily predictions.",
               style={"color": "#94a3b8", "fontSize": "13px", "margin": "0 0 20px 0"}),
        html.Button("View Plans →", id="limit-upgrade-btn", n_clicks=0, style={
            "padding": "12px 28px", "borderRadius": "10px", "border": "none",
            "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
            "color": "#fff", "fontSize": "14px", "fontWeight": "700", "cursor": "pointer",
        }),
    ], style={
        "textAlign": "center", "padding": "60px 20px",
        "background": "rgba(15,23,42,0.6)", "borderRadius": "16px",
        "border": "1px solid rgba(99,102,241,0.12)",
    })


# =============================================================================
# HELPERS — Shared styles
# =============================================================================

def _input_style():
    return {
        "width": "100%", "padding": "10px 14px", "borderRadius": "10px",
        "border": "1px solid rgba(99,102,241,0.2)", "background": "rgba(15,23,42,0.6)",
        "color": "#e2e8f0", "fontSize": "14px", "outline": "none",
        "boxSizing": "border-box",
    }


def _primary_btn_style():
    return {
        "width": "100%", "padding": "12px", "borderRadius": "10px",
        "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
        "color": "#fff", "border": "none", "fontSize": "14px",
        "fontWeight": "700", "cursor": "pointer",
        "boxShadow": "0 4px 20px rgba(99,102,241,0.3)",
    }


# =============================================================================
# SETUP HELPER — Call from app.py to wire everything up
# =============================================================================

def setup_auth_system(app, server):
    """
    Call once from app.py after creating the Dash app.

    Usage:
        from saas_auth import setup_auth_system
        setup_auth_system(app, server)
    """
    # Register Stripe webhook
    register_webhook_route(server)
    logger.info("✅ Auth & payments system initialized")