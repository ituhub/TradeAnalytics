"""
STRIPE PRODUCTS & PRICES SETUP — MarketLens AI
=============================================================================
Usage:
    export STRIPE_SECRET_KEY=sk_test_...
    python setup_stripe.py

Creates 2 products × 2 prices (monthly + yearly) = 4 Stripe Price IDs.
Enterprise tier is contact-only (no Stripe product needed).
Currency: EUR

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

print("🚀 Creating MarketLens AI Stripe products and prices (EUR)...\n")

PLANS = [
    {
        "id": "starter",
        "name": "MarketLens AI — Starter",
        "description": "4-model AI ensemble, 10 predictions/day, 3 timeframes, 7 tickers, backtesting, SHAP, regime detection, drift alerts, MTF analysis",
        "monthly_cents": 3900,   # €39/mo
        "yearly_cents": 2900,    # €29/mo billed yearly (€348/year)
    },
    {
        "id": "professional",
        "name": "MarketLens AI — Professional",
        "description": "Full 8-model ensemble, 25 predictions/day, all timeframes, all tickers, full backtesting suite, FTMO dashboard, model training",
        "monthly_cents": 8900,   # €89/mo
        "yearly_cents": 6900,    # €69/mo billed yearly (€828/year)
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
        currency="eur",
        recurring={"interval": "month"},
        metadata={"plan_id": plan["id"], "billing": "monthly"},
    )

    # Yearly price (total yearly amount)
    yearly_price = stripe.Price.create(
        product=product.id,
        unit_amount=plan["yearly_cents"] * 12,  # total yearly amount in cents
        currency="eur",
        recurring={"interval": "year"},
        metadata={"plan_id": plan["id"], "billing": "yearly"},
    )

    env_key_prefix = plan["id"].upper()
    env_lines.append(f"STRIPE_{env_key_prefix}_MONTHLY={monthly_price.id}")
    env_lines.append(f"STRIPE_{env_key_prefix}_YEARLY={yearly_price.id}")

    print(f"   ✅ Monthly: {monthly_price.id} (€{plan['monthly_cents']/100:.0f}/mo)")
    print(f"   ✅ Yearly:  {yearly_price.id} (€{plan['yearly_cents']/100:.0f}/mo billed yearly)")
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
print("=" * 60)
print("📝 Note: Enterprise tier is contact-only — no Stripe product needed.")
print("   Enterprise clients are manually onboarded via itubusinesshub@gmail.com")
print()
print("✅ Done! Products created in Stripe Dashboard.")
print("   Next: Set up webhook endpoint at /webhook/stripe in Stripe Dashboard")
print("   Events to listen for: checkout.session.completed, customer.subscription.updated,")
print("   customer.subscription.deleted, invoice.payment_failed")
