"""
STRIPE PRODUCTS & PRICES SETUP — Run once to create your subscription plans
=============================================================================
Usage:
    export STRIPE_SECRET_KEY=sk_test_...
    python setup_stripe.py

Creates 3 products × 2 prices (monthly + yearly) = 6 Stripe Price IDs.
Prints the environment variables to add to your .env or Cloud Run config.
=============================================================================
"""

import os
import stripe

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")

if not stripe.api_key:
    print("❌ Set STRIPE_SECRET_KEY environment variable first")
    print("   export STRIPE_SECRET_KEY=sk_test_YOUR_KEY_HERE")
    exit(1)

print("🚀 Creating Stripe products and prices...\n")

PLANS = [
    {
        "id": "starter",
        "name": "AI Trading Pro — Starter",
        "description": "3-model AI ensemble, 5 predictions/day, daily timeframe, 5 tickers",
        "monthly_cents": 4900,   # $49/mo
        "yearly_cents": 3900,    # $39/mo billed yearly
    },
    {
        "id": "professional",
        "name": "AI Trading Pro — Professional",
        "description": "Full 8-model ensemble, 50 predictions/day, all timeframes, all tickers, backtesting, SHAP",
        "monthly_cents": 12900,  # $129/mo
        "yearly_cents": 9900,    # $99/mo billed yearly
    },
    {
        "id": "institutional",
        "name": "AI Trading Pro — Institutional",
        "description": "Unlimited predictions, portfolio optimization, API access, dedicated support",
        "monthly_cents": 34900,  # $349/mo
        "yearly_cents": 27900,   # $279/mo billed yearly
    },
]

env_lines = []

for plan in PLANS:
    print(f"📦 Creating product: {plan['name']}")

    # Create product
    product = stripe.Product.create(
        name=plan["name"],
        description=plan["description"],
        metadata={"plan_id": plan["id"]},
    )

    # Monthly price
    monthly_price = stripe.Price.create(
        product=product.id,
        unit_amount=plan["monthly_cents"],
        currency="usd",
        recurring={"interval": "month"},
        metadata={"plan_id": plan["id"], "billing": "monthly"},
    )

    # Yearly price (billed monthly amount × 12 as yearly total)
    yearly_price = stripe.Price.create(
        product=product.id,
        unit_amount=plan["yearly_cents"] * 12,  # total yearly amount
        currency="usd",
        recurring={"interval": "year"},
        metadata={"plan_id": plan["id"], "billing": "yearly"},
    )

    env_key_prefix = plan["id"].upper()
    env_lines.append(f"STRIPE_{env_key_prefix}_MONTHLY={monthly_price.id}")
    env_lines.append(f"STRIPE_{env_key_prefix}_YEARLY={yearly_price.id}")

    print(f"   ✅ Monthly: {monthly_price.id} (${plan['monthly_cents']/100:.0f}/mo)")
    print(f"   ✅ Yearly:  {yearly_price.id} (${plan['yearly_cents']/100:.0f}/mo billed yearly)")
    print()

print("=" * 60)
print("📋 Add these to your .env file or Cloud Run environment:\n")
for line in env_lines:
    print(f"  {line}")

print()
print("  # Also set these:")
print("  STRIPE_SECRET_KEY=sk_live_YOUR_KEY")
print("  STRIPE_PUBLISHABLE_KEY=pk_live_YOUR_KEY")
print("  STRIPE_WEBHOOK_SECRET=whsec_YOUR_WEBHOOK_SECRET")
print("  APP_DOMAIN=https://your-domain.com")
print()
print("✅ Done! Products created in Stripe Dashboard.")
print("   Next: Set up webhook endpoint at /webhook/stripe in Stripe Dashboard")
print("   Events to listen for: checkout.session.completed, customer.subscription.updated,")
print("   customer.subscription.deleted, invoice.payment_failed")
