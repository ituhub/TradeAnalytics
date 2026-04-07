"""
=============================================================================
EMAIL SERVICE — TradeAnalytics Platform
=============================================================================
Handles all outbound email communications:
  - Platform updates & announcements
  - Market alerts & AI prediction summaries
  - Subscription/billing notifications (plan upgrades, trial expiry, receipts)
  - Bulk email to all registered users or filtered by plan

Uses Gmail SMTP with App Password authentication.
Integrates with Firestore user collection from saas_auth.py.

Setup:
  1. Enable 2FA on your Gmail account
  2. Generate an App Password: Google Account → Security → App Passwords
  3. Set env vars: SMTP_EMAIL, SMTP_APP_PASSWORD
  4. Optional: PLATFORM_NAME, PLATFORM_URL
=============================================================================
"""

import smtplib
import ssl
import logging
import os
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "")
SMTP_APP_PASSWORD = os.environ.get("SMTP_APP_PASSWORD", "")

PLATFORM_NAME = os.environ.get("PLATFORM_NAME", "MarketLens AI")
PLATFORM_URL = os.environ.get("PLATFORM_URL", "https://tradeanalytics-XXXXX.run.app")
SUPPORT_EMAIL = os.environ.get("SUPPORT_EMAIL", SMTP_EMAIL)

# Rate limiting: Gmail allows ~500 emails/day
MAX_EMAILS_PER_BATCH = 50
EMAIL_THREAD_WORKERS = 5

# Firestore reference (set by init_email_service)
_firestore_db = None


# =============================================================================
# INITIALIZATION
# =============================================================================

def init_email_service(firestore_db=None):
    """
    Initialize email service with Firestore reference.
    Call this from app.py after Firestore is initialized.

    Usage in app.py:
        from email_service import init_email_service
        init_email_service(firestore_db=db)
    """
    global _firestore_db
    _firestore_db = firestore_db

    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        logger.warning(
            "⚠️ Email service: SMTP_EMAIL or SMTP_APP_PASSWORD not set. "
            "Emails will be logged but not sent."
        )
        return False

    # Test SMTP connection
    try:
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
        logger.info("✅ Email service initialized — SMTP connection verified")
        return True
    except Exception as e:
        logger.error(f"❌ Email SMTP connection failed: {e}")
        return False


# =============================================================================
# HTML EMAIL TEMPLATES
# =============================================================================

def _base_template(content_html: str, preview_text: str = "") -> str:
    """Wrap content in the base email template with dark trading-themed design."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{PLATFORM_NAME}</title>
<!--[if !mso]><!-->
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
</style>
<!--<![endif]-->
</head>
<body style="margin:0;padding:0;background-color:#0a0e17;font-family:'Inter',Arial,sans-serif;">
<!-- Preview text (hidden) -->
<div style="display:none;max-height:0;overflow:hidden;mso-hide:all;">
  {preview_text}
</div>

<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a0e17;">
<tr><td align="center" style="padding:24px 16px;">

<!-- Main container -->
<table role="presentation" width="600" cellpadding="0" cellspacing="0"
       style="max-width:600px;width:100%;background-color:#111827;border-radius:12px;
              border:1px solid #1e293b;overflow:hidden;">

  <!-- Header -->
  <tr>
    <td style="background:linear-gradient(135deg,#0f172a 0%,#1e293b 100%);
               padding:32px 40px;border-bottom:1px solid #334155;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <span style="font-size:22px;font-weight:700;color:#f1f5f9;letter-spacing:-0.5px;">
              📊 {PLATFORM_NAME}
            </span>
          </td>
          <td align="right">
            <span style="font-size:12px;color:#64748b;text-transform:uppercase;letter-spacing:1px;">
              {datetime.now().strftime('%b %d, %Y')}
            </span>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- Content -->
  <tr>
    <td style="padding:40px;">
      {content_html}
    </td>
  </tr>

  <!-- Footer -->
  <tr>
    <td style="background-color:#0f172a;padding:24px 40px;border-top:1px solid #1e293b;">
      <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="font-size:12px;color:#475569;line-height:1.6;">
            You're receiving this because you're registered on {PLATFORM_NAME}.<br>
            <a href="{PLATFORM_URL}" style="color:#38bdf8;text-decoration:none;">
              Open Platform
            </a>
            &nbsp;&middot;&nbsp;
            <a href="mailto:{SUPPORT_EMAIL}" style="color:#38bdf8;text-decoration:none;">
              Contact Support
            </a>
          </td>
        </tr>
        <tr>
          <td style="font-size:11px;color:#334155;padding-top:12px;">
            &copy; {datetime.now().year} {PLATFORM_NAME}. All rights reserved.
          </td>
        </tr>
      </table>
    </td>
  </tr>

</table>
<!-- /Main container -->

</td></tr>
</table>
</body>
</html>"""


def _template_announcement(title: str, message: str, cta_text: str = "",
                           cta_url: str = "") -> str:
    """Platform update / announcement email."""
    cta_html = ""
    if cta_text and cta_url:
        cta_html = f"""
      <tr><td style="padding-top:28px;" align="center">
        <a href="{cta_url}"
           style="display:inline-block;background:linear-gradient(135deg,#2563eb,#7c3aed);
                  color:#ffffff;font-size:15px;font-weight:600;padding:14px 36px;
                  border-radius:8px;text-decoration:none;letter-spacing:0.3px;">
          {cta_text}
        </a>
      </td></tr>"""

    content = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr><td>
        <h1 style="margin:0 0 8px;font-size:24px;font-weight:700;color:#f1f5f9;">
          {title}
        </h1>
        <div style="width:60px;height:3px;background:linear-gradient(90deg,#2563eb,#7c3aed);
                    border-radius:2px;margin-bottom:24px;"></div>
      </td></tr>
      <tr><td style="font-size:15px;line-height:1.7;color:#cbd5e1;">
        {message}
      </td></tr>
      {cta_html}
    </table>"""
    return _base_template(content, preview_text=title)


def _template_market_alert(ticker: str, signal: str, confidence: float,
                           predicted_price: float, current_price: float,
                           details: str = "") -> str:
    """Market alert / AI prediction notification."""
    signal_color = "#22c55e" if signal.upper() == "BUY" else "#ef4444" if signal.upper() == "SELL" else "#f59e0b"
    signal_label = signal.upper()
    change_pct = ((predicted_price - current_price) / current_price) * 100

    content = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr><td>
        <h1 style="margin:0 0 8px;font-size:22px;font-weight:700;color:#f1f5f9;">
          🚨 Market Alert: {ticker}
        </h1>
        <div style="width:60px;height:3px;background:{signal_color};
                    border-radius:2px;margin-bottom:24px;"></div>
      </td></tr>

      <!-- Signal badge -->
      <tr><td style="padding-bottom:24px;">
        <table role="presentation" cellpadding="0" cellspacing="0">
          <tr>
            <td style="background:{signal_color};color:#ffffff;font-size:14px;font-weight:700;
                       padding:8px 20px;border-radius:6px;letter-spacing:1px;">
              {signal_label}
            </td>
            <td style="padding-left:16px;font-size:14px;color:#94a3b8;">
              Confidence: <strong style="color:#f1f5f9;">{confidence:.1f}%</strong>
            </td>
          </tr>
        </table>
      </td></tr>

      <!-- Price box -->
      <tr><td>
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0"
               style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;">
          <tr>
            <td style="padding:20px;text-align:center;width:50%;border-right:1px solid #1e293b;">
              <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;
                          margin-bottom:6px;">Current Price</div>
              <div style="font-size:24px;font-weight:700;color:#f1f5f9;">
                ${current_price:,.2f}
              </div>
            </td>
            <td style="padding:20px;text-align:center;width:50%;">
              <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:1px;
                          margin-bottom:6px;">AI Predicted</div>
              <div style="font-size:24px;font-weight:700;color:{signal_color};">
                ${predicted_price:,.2f}
              </div>
              <div style="font-size:13px;color:{signal_color};margin-top:4px;">
                ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)
              </div>
            </td>
          </tr>
        </table>
      </td></tr>

      {'<tr><td style="font-size:14px;line-height:1.6;color:#94a3b8;padding-top:20px;">'
       + details + '</td></tr>' if details else ''}

      <tr><td align="center" style="padding-top:28px;">
        <a href="{PLATFORM_URL}"
           style="display:inline-block;background:linear-gradient(135deg,#2563eb,#7c3aed);
                  color:#ffffff;font-size:14px;font-weight:600;padding:12px 32px;
                  border-radius:8px;text-decoration:none;">
          View Full Analysis →
        </a>
      </td></tr>
    </table>"""
    return _base_template(content, preview_text=f"{signal_label} signal for {ticker} — {confidence:.0f}% confidence")


def _template_subscription(user_name: str, event: str, plan: str = "",
                           details: str = "") -> str:
    """Subscription/billing notification (upgrade, downgrade, trial, receipt)."""
    event_titles = {
        "welcome": "Welcome aboard! 🎉",
        "upgrade": "Plan Upgraded Successfully ✅",
        "downgrade": "Plan Changed",
        "trial_started": "Your Pro Trial Has Started 🚀",
        "trial_expiring": "Your Trial Expires Soon ⏰",
        "trial_expired": "Your Trial Has Ended",
        "payment_success": "Payment Received ✅",
        "payment_failed": "Payment Issue ⚠️",
        "cancellation": "Subscription Cancelled",
    }
    title = event_titles.get(event, "Account Update")

    content = f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr><td>
        <h1 style="margin:0 0 8px;font-size:24px;font-weight:700;color:#f1f5f9;">
          {title}
        </h1>
        <div style="width:60px;height:3px;background:linear-gradient(90deg,#2563eb,#7c3aed);
                    border-radius:2px;margin-bottom:24px;"></div>
      </td></tr>
      <tr><td style="font-size:15px;line-height:1.7;color:#cbd5e1;">
        <p style="margin:0 0 16px;">Hi <strong>{user_name}</strong>,</p>
        {details if details else f'<p style="margin:0;">Your plan: <strong style="color:#38bdf8;">{plan.upper()}</strong></p>'}
      </td></tr>
      <tr><td align="center" style="padding-top:28px;">
        <a href="{PLATFORM_URL}"
           style="display:inline-block;background:linear-gradient(135deg,#2563eb,#7c3aed);
                  color:#ffffff;font-size:14px;font-weight:600;padding:12px 32px;
                  border-radius:8px;text-decoration:none;">
          Go to Dashboard →
        </a>
      </td></tr>
    </table>"""
    return _base_template(content, preview_text=title)


# =============================================================================
# CORE SENDING FUNCTIONS
# =============================================================================

def send_email(to_email: str, subject: str, html_body: str,
               reply_to: str = "") -> Dict:
    """
    Send a single email via Gmail SMTP.
    Returns {"success": True/False, "email": ..., "error": ...}
    """
    if not SMTP_EMAIL or not SMTP_APP_PASSWORD:
        logger.warning(f"📧 [DRY RUN] Would send to {to_email}: {subject}")
        return {"success": False, "email": to_email,
                "error": "SMTP credentials not configured"}

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{PLATFORM_NAME} <{SMTP_EMAIL}>"
        msg["To"] = to_email
        msg["Subject"] = subject
        if reply_to:
            msg["Reply-To"] = reply_to

        # Plain text fallback
        plain_text = (
            f"This email is best viewed in an HTML-compatible email client.\n"
            f"Visit {PLATFORM_URL} to view this content.\n"
        )
        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=context)
            server.login(SMTP_EMAIL, SMTP_APP_PASSWORD)
            server.sendmail(SMTP_EMAIL, to_email, msg.as_string())

        logger.info(f"✅ Email sent to {to_email}: {subject}")
        return {"success": True, "email": to_email}

    except smtplib.SMTPAuthenticationError:
        error = "SMTP authentication failed — check App Password"
        logger.error(f"❌ {error}")
        return {"success": False, "email": to_email, "error": error}
    except Exception as e:
        logger.error(f"❌ Email to {to_email} failed: {e}")
        return {"success": False, "email": to_email, "error": str(e)}


def send_bulk_email(recipients: List[Dict], subject: str,
                    html_body: str) -> Dict:
    """
    Send email to multiple recipients in parallel.
    recipients: list of {"email": ..., "name": ...}
    Returns summary: {"sent": N, "failed": N, "errors": [...]}
    """
    results = {"sent": 0, "failed": 0, "errors": [], "total": len(recipients)}

    if not recipients:
        return results

    # Process in batches
    for batch_start in range(0, len(recipients), MAX_EMAILS_PER_BATCH):
        batch = recipients[batch_start:batch_start + MAX_EMAILS_PER_BATCH]

        with ThreadPoolExecutor(max_workers=EMAIL_THREAD_WORKERS) as executor:
            futures = {
                executor.submit(send_email, r["email"], subject, html_body): r
                for r in batch
            }
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    results["sent"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(result)

    logger.info(
        f"📧 Bulk email complete: {results['sent']}/{results['total']} sent, "
        f"{results['failed']} failed"
    )
    return results


# =============================================================================
# USER RETRIEVAL (from Firestore)
# =============================================================================

def get_all_users(plan_filter: str = "all") -> List[Dict]:
    """
    Fetch registered users from Firestore.
    plan_filter: 'all', 'free', 'pro', 'premium'
    """
    users = []

    if _firestore_db:
        try:
            query = _firestore_db.collection("users")
            if plan_filter and plan_filter != "all":
                query = query.where("plan", "==", plan_filter)

            docs = query.stream()
            for doc in docs:
                data = doc.to_dict()
                if data.get("email"):
                    users.append({
                        "email": data["email"],
                        "name": data.get("name", data["email"].split("@")[0]),
                        "plan": data.get("plan", "free"),
                        "user_id": data.get("user_id", doc.id),
                    })
            logger.info(f"📋 Fetched {len(users)} users (filter: {plan_filter})")
        except Exception as e:
            logger.error(f"❌ Firestore user fetch failed: {e}")
    else:
        # Fallback: local JSON users
        try:
            local_path = os.path.join(
                os.path.dirname(__file__), "saved_results", "users.json"
            )
            if os.path.exists(local_path):
                with open(local_path) as f:
                    local_users = json.load(f)
                for uid, data in local_users.items():
                    if data.get("email"):
                        if plan_filter == "all" or data.get("plan") == plan_filter:
                            users.append({
                                "email": data["email"],
                                "name": data.get("name", ""),
                                "plan": data.get("plan", "free"),
                                "user_id": uid,
                            })
            logger.info(f"📋 Fetched {len(users)} local users (filter: {plan_filter})")
        except Exception as e:
            logger.error(f"❌ Local user fetch failed: {e}")

    return users


# =============================================================================
# HIGH-LEVEL SEND FUNCTIONS
# =============================================================================

def send_announcement(title: str, message: str, plan_filter: str = "all",
                      cta_text: str = "", cta_url: str = "") -> Dict:
    """
    Send a platform announcement to all users (or filtered by plan).

    Usage:
        result = send_announcement(
            title="New Feature: Walk-Forward Backtesting",
            message="We've added walk-forward validation with quarter-Kelly...",
            plan_filter="all",
            cta_text="Try It Now",
            cta_url="https://your-app.run.app"
        )
    """
    users = get_all_users(plan_filter)
    if not users:
        return {"sent": 0, "failed": 0, "total": 0,
                "errors": [{"error": "No users found"}]}

    html = _template_announcement(title, message, cta_text, cta_url)
    subject = f"📊 {title} — {PLATFORM_NAME}"
    return send_bulk_email(users, subject, html)


def send_market_alert(ticker: str, signal: str, confidence: float,
                      predicted_price: float, current_price: float,
                      details: str = "",
                      plan_filter: str = "pro") -> Dict:
    """
    Send a market alert to pro/premium users.

    Usage:
        result = send_market_alert(
            ticker="AAPL",
            signal="BUY",
            confidence=87.5,
            predicted_price=198.50,
            current_price=192.30,
            details="8-model ensemble consensus with high conviction.",
            plan_filter="pro"
        )
    """
    users = get_all_users(plan_filter)
    if not users:
        return {"sent": 0, "failed": 0, "total": 0,
                "errors": [{"error": "No pro users found"}]}

    html = _template_market_alert(
        ticker, signal, confidence, predicted_price, current_price, details
    )
    subject = f"🚨 {signal.upper()} Alert: {ticker} — {confidence:.0f}% Confidence"
    return send_bulk_email(users, subject, html)


def send_subscription_notification(user_email: str, user_name: str,
                                   event: str, plan: str = "",
                                   details: str = "") -> Dict:
    """
    Send a subscription event email to a single user.

    Events: welcome, upgrade, downgrade, trial_started, trial_expiring,
            trial_expired, payment_success, payment_failed, cancellation

    Usage:
        send_subscription_notification(
            user_email="john@example.com",
            user_name="John",
            event="upgrade",
            plan="pro",
            details="<p>You now have unlimited predictions and priority support.</p>"
        )
    """
    html = _template_subscription(user_name, event, plan, details)
    event_subjects = {
        "welcome": f"Welcome to {PLATFORM_NAME}!",
        "upgrade": f"Plan Upgraded — {PLATFORM_NAME}",
        "downgrade": f"Plan Changed — {PLATFORM_NAME}",
        "trial_started": f"Your Pro Trial Has Started — {PLATFORM_NAME}",
        "trial_expiring": f"Trial Expiring Soon — {PLATFORM_NAME}",
        "trial_expired": f"Trial Ended — {PLATFORM_NAME}",
        "payment_success": f"Payment Confirmed — {PLATFORM_NAME}",
        "payment_failed": f"Payment Issue — {PLATFORM_NAME}",
        "cancellation": f"Subscription Cancelled — {PLATFORM_NAME}",
    }
    subject = event_subjects.get(event, f"Account Update — {PLATFORM_NAME}")
    return send_email(user_email, subject, html)


# =============================================================================
# ADMIN: SEND CUSTOM EMAIL (for Dash callbacks)
# =============================================================================

def send_custom_email(subject: str, body_html: str,
                      plan_filter: str = "all") -> Dict:
    """
    Send a custom HTML email to users.
    body_html can contain raw HTML (paragraphs, lists, links, etc.)
    It will be wrapped in the base template automatically.
    """
    users = get_all_users(plan_filter)
    if not users:
        return {"sent": 0, "failed": 0, "total": 0,
                "errors": [{"error": "No users found"}]}

    full_html = _base_template(f"""
    <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
      <tr><td style="font-size:15px;line-height:1.7;color:#cbd5e1;">
        {body_html}
      </td></tr>
    </table>
    """, preview_text=subject)

    return send_bulk_email(users, subject, full_html)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def send_test_email(to_email: str) -> Dict:
    """Send a test email to verify SMTP configuration."""
    html = _template_announcement(
        title="Test Email — Configuration Verified ✅",
        message=(
            "<p>If you're reading this, your email service is working correctly.</p>"
            "<p style='margin-top:12px;'>You can now send announcements, market alerts, "
            "and subscription notifications to your registered users.</p>"
        ),
        cta_text="Open Platform",
        cta_url=PLATFORM_URL,
    )
    return send_email(to_email, f"✅ Test Email — {PLATFORM_NAME}", html)
