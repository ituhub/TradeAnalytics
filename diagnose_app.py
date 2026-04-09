#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║     MarketLens AI — Pre-Launch Validation & Health Check v2          ║
║                                                                      ║
║  Run on GCP Cloud Shell:                                             ║
║    cd ~/TradeAnalytics && python3 diagnose_app.py                    ║
║                                                                      ║
║  Validates EVERY landing page claim against actual code,             ║
║  tests all plan gating, auth flow, API connectivity,                 ║
║  responsive CSS, and deployment status.                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, ast, re, subprocess
from datetime import datetime

PASS = "✅"; FAIL = "❌"; WARN = "⚠️"; INFO = "ℹ️"; SKIP = "⏭️"
results = {"pass": 0, "fail": 0, "warn": 0, "skip": 0}

def check(label, cond, detail="", warn_only=False):
    if cond: results["pass"] += 1; print(f"  {PASS} {label}")
    elif warn_only: results["warn"] += 1; print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    else: results["fail"] += 1; print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))

def skip(label, reason=""): results["skip"] += 1; print(f"  {SKIP} {label}" + (f" — {reason}" if reason else ""))

def section(title): print(f"\n{'─'*64}\n  {title}\n{'─'*64}")

_src = {}
for f in ["app.py", "saas_auth.py", "pages.py"]:
    if os.path.isfile(f):
        with open(f) as fh: _src[f] = fh.read()
def S(f): return _src.get(f, "")

# ═══════════════════════════════════════════════════════════════
section("1. ENVIRONMENT & SYNTAX")
check(f"Python {sys.version.split()[0]} >= 3.10", sys.version_info >= (3, 10))
for f in ["app.py", "saas_auth.py", "pages.py"]:
    if os.path.isfile(f):
        try: ast.parse(S(f)); check(f"{f} syntax OK ({os.path.getsize(f):,}b)", True)
        except SyntaxError as e: check(f, False, f"Line {e.lineno}: {e.msg}")
    else: check(f, False, "MISSING")

# ═══════════════════════════════════════════════════════════════
section("2. FILE STRUCTURE")
for f in ["app.py","saas_auth.py","pages.py","requirements.txt","Dockerfile"]:
    check(f, os.path.isfile(f), "MISSING")
check("assets/ directory", os.path.isdir("assets"))
if os.path.isdir("assets"):
    css = [f for f in os.listdir("assets") if f.endswith(".css")]
    check(f"CSS: {', '.join(css)}", len(css) > 0)
    check("responsive.css", "responsive.css" in css, "No mobile CSS", warn_only=True)
    check("glassmorphism.css", "glassmorphism.css" in css)
for f in ["enhprog.py","gcs_model_loader.py","ai_backtest_engine.py","ai_portfolio_system.py","email_service.py","disclaimer.py","app_guide.py"]:
    if os.path.isfile(f): check(f"{f} ({os.path.getsize(f):,}b)", True)
    else: skip(f)

# ═══════════════════════════════════════════════════════════════
section("3. MOBILE READINESS")
check("Viewport meta tag", "viewport" in S("app.py") and "width=device-width" in S("app.py"), "Mobile won't scale")
check("apple-mobile-web-app-capable", "apple-mobile-web-app-capable" in S("app.py"), warn_only=True)
check("theme-color meta", "theme-color" in S("app.py"), warn_only=True)
if os.path.isfile("assets/responsive.css"):
    with open("assets/responsive.css") as f: rcss = f.read()
    check("@media 768px breakpoint", "@media (max-width: 768px)" in rcss)
    check("@media 480px breakpoint", "@media (max-width: 480px)" in rcss)
    check("Sidebar responsive", ".sidebar-container" in rcss)
    check("Landing page responsive", "flexDirection" in rcss or "flex-direction" in rcss)
    check("Grid responsive", "ecard-grid" in rcss)
    check("Touch targets 44px", "44px" in rcss, warn_only=True)
    check("iOS zoom prevention (16px inputs)", "font-size: 16px" in rcss, warn_only=True)
    check("Safe area insets (notched phones)", "safe-area-inset" in rcss, warn_only=True)
else: skip("Responsive CSS", "not found")

# ═══════════════════════════════════════════════════════════════
section("4. LANDING PAGE CLAIMS vs CODE")
a, auth, pg = S("app.py"), S("saas_auth.py"), S("pages.py")

models = re.findall(r'"label":\s*"[^"]+",\s*"value":\s*"[^"]+"', a)
check(f"'8 AI Models' — found {len(models)} in MODEL_OPTIONS", len(models) == 8)

tg = re.findall(r'"([A-Z\^=]+)"', a[:5000])
ut = set(t for t in tg if len(t) >= 3 and t not in ("ALL",))
check(f"'13+ Tickers' — found {len(ut)} unique", len(ut) >= 13, f"Only {len(ut)}")

claims = [
    ("Analyze market behavior (PredictionEngine)", "class PredictionEngine" in a),
    ("Multi-model perspectives", "available_trained_models" in a or "models_used" in a),
    ("Backtest strategies (run_backtest)", "def run_backtest" in pg),
    ("Backtest page builder", "def build_backtest_page" in pg),
    ("SHAP analysis in gating", "shap_explanations" in auth),
    ("SHAP in code", "shap" in a.lower() or "shap" in pg.lower()),
    ("Regime analysis engine", "def run_regime_analysis" in pg),
    ("Regime detection gating", "regime_detection" in auth),
    ("MTF analysis engine", "_run_multi_timeframe_analysis" in a),
    ("MTF checklist in UI", "mtf-checklist" in a),
    ("MTF plan gating", "mtf_analysis" in auth),
    ("Portfolio optimization", "def run_portfolio_optimization" in pg),
    ("'What You Can Do' section", "What You Can Do" in auth),
    ("'Built for Research' section", "Built for Research" in auth),
    ("FTMO removed from landing", "FTMO Challenge Dashboard" not in auth),
    ("MSE removed from landing", "MSE Score" not in auth),
]
for label, cond in claims: check(label, cond)

# ═══════════════════════════════════════════════════════════════
section("5. SUBSCRIPTION PLANS")
EXPECTED = {"free","discovery","starter","professional","enterprise"}
found = set(re.findall(r'"(free|discovery|starter|professional|enterprise)":\s*\{', auth))
check(f"All 5 plans: {', '.join(sorted(found))}", EXPECTED.issubset(found), f"Missing: {EXPECTED-found}")

# Discovery tickers
disco_block = auth.split('"discovery"')[1].split('"starter"')[0] if '"discovery"' in auth and '"starter"' in auth else ""
check("Discovery: no SPY", '"SPY"' not in disco_block, "SPY in Discovery")
check("Discovery: no AAPL", '"AAPL"' not in disco_block, "AAPL in Discovery")

# Free tickers
free_block = auth.split('"free"')[1].split('"discovery"')[0] if '"free"' in auth and '"discovery"' in auth else ""
check("Free: no SPY", '"SPY"' not in free_block)
check("Free: no AAPL", '"AAPL"' not in free_block)

# MTF gating per plan
check("Free: mtf_analysis=False", re.search(r'"free".*?"mtf_analysis":\s*False', auth, re.DOTALL) is not None)
check("Discovery: mtf_analysis=False", re.search(r'"discovery".*?"mtf_analysis":\s*False', auth, re.DOTALL) is not None)
check("Starter: mtf_analysis=True", re.search(r'"starter".*?"mtf_analysis":\s*True', auth, re.DOTALL) is not None)

# MTF UI gating
check("MTF options disabled by plan", "not can_access_feature" in a and "mtf-checklist" in a)
check("MTF upgrade prompt in results", "mtf_upgrade_required" in a)
check("MTF 🔒 badge", "🔒" in a)
check("Quick select filtered by plan", "any(t in allowed_tickers" in a)
check("Quick select callback checks user", "get_allowed_tickers" in a and "quick_select_group" in a)

# ═══════════════════════════════════════════════════════════════
section("6. AUTH & SESSION")
for fn in ["build_login_page","create_user","authenticate_user","get_user_by_token",
           "logout_user","check_prediction_limit","record_prediction","get_user_plan",
           "get_allowed_tickers","get_allowed_timeframes","can_access_feature","is_admin",
           "reset_user_password","is_discovery_active","get_discovery_days_remaining"]:
    check(f"{fn}", f"def {fn}" in auth)

check("Session: localStorage persistence", 'storage_type="local"' in a and "user-session" in a)
check("Login: 'MarketLens AI' (not Welcome Back)", "Welcome Back" not in auth, warn_only=True)
check("Login: 'AI-Powered Market Intelligence'", "AI-Powered" in auth and "Market Intelligence" in auth, warn_only=True)
check("Contact Support hidden from sidebar", 'display": "none"' in auth and "open-contact-btn-pricing" in auth, warn_only=True)

# ═══════════════════════════════════════════════════════════════
section("7. CALLBACK CHAIN")
check("timeframe-dropdown in layout", 'id="timeframe-dropdown"' in a, "CRITICAL: Analysis button broken!")
check("timeframe-dropdown is dcc.Store", 'Store(id="timeframe-dropdown"' in a)
check("Callback reads 'data' not 'value'", '"timeframe-dropdown", "data"' in a)
check("sync_mtf_to_timeframe callback", "def sync_mtf_to_timeframe" in a)

for fn, desc in [("route_auth","Auth routing"),("handle_login","Login"),("handle_register","Register"),
                 ("handle_demo_mode","Demo"),("handle_logout","Logout"),("run_prediction","AI analysis"),
                 ("route_page","Page routing"),("toggle_auth_tabs","Auth tabs"),("quick_select_group","Quick select"),
                 ("sync_mtf_to_timeframe","MTF sync")]:
    check(f"Callback: {desc}", f"def {fn}" in a)

IDS = ["app-container","url","user-session","prediction-store","predict-btn","ticker-dropdown",
       "model-checklist","timeframe-dropdown","mtf-checklist","page-nav","page-header","page-content",
       "redirect-store","login-btn","login-email","login-password","register-btn","demo-mode-link"]
combined = a + auth
missing = [i for i in IDS if f'"{i}"' not in combined]
check(f"All {len(IDS)} component IDs", len(missing)==0, f"Missing: {', '.join(missing)}")

# ═══════════════════════════════════════════════════════════════
section("8. ALL PAGES & FEATURES")
for fn, desc, src in [
    ("build_ai_prediction_page","Market Analysis",a),("build_prediction_results","Results",a),
    ("_build_mtf_section","MTF section",a),("build_trading_strategy_tab","Strategy tab",a),
    ("build_forecast_tab","Forecast tab",a),("build_risk_tab","Risk tab",a),
    ("generate_risk_metrics","Risk metrics",a),("build_price_trajectory_chart","Price chart",a),
    ("build_analytics_page","Analytics",pg),("build_backtest_page","Backtesting",pg),
    ("build_portfolio_page","Portfolio",pg),("build_model_training_page","Model Training",pg),
    ("build_admin_page","Admin Panel",pg),("build_pricing_page","Pricing",auth),
    ("build_user_badge","User badge",auth),("build_upgrade_prompt","Upgrade prompt",auth),
    ("run_regime_analysis","Regime engine",pg),("run_drift_detection","Drift engine",pg),
    ("run_backtest","Backtest engine",pg),("run_portfolio_optimization","Portfolio optimizer",pg),
]: check(f"{desc} ({fn})", f"def {fn}" in src)

# ═══════════════════════════════════════════════════════════════
section("9. AI MODELS IN enhprog.py")
if os.path.isfile("enhprog.py"):
    with open("enhprog.py") as f: ep = f.read()
    for cls, name in [("XGBoostTimeSeriesModel","XGBoost"),("SklearnEnsemble","Sklearn"),
                      ("AdvancedTransformer","Transformer"),("CNNLSTMAttention","CNN-LSTM"),
                      ("EnhancedTCN","TCN"),("EnhancedInformer","Informer"),
                      ("EnhancedNBeats","N-BEATS"),("LSTMGRUEnsemble","LSTM-GRU")]:
        check(f"{name} ({cls})", f"class {cls}" in ep)
else: skip("enhprog.py checks", "File not found")

# ═══════════════════════════════════════════════════════════════
section("10. ENVIRONMENT VARIABLES")
print(f"  {INFO} Env vars may only be set on Cloud Run, not Cloud Shell")
for var, desc in [("STRIPE_SECRET_KEY","Stripe"),("FMP_API_KEY","FMP API"),("APP_DOMAIN","Domain")]:
    v = os.environ.get(var,"")
    if v: check(f"{desc}: set", True)
    else: check(f"{desc} ({var})", False, "Not set locally", warn_only=True)

# ═══════════════════════════════════════════════════════════════
section("11. FIRESTORE")
try:
    from google.cloud import firestore
    db = firestore.Client()
    check("Firestore connected", True)
    users = list(db.collection("users").stream())
    plans = {}
    for d in users:
        p = d.to_dict().get("plan","free")
        plans[p] = plans.get(p,0)+1
    check(f"Users: {len(users)}", len(users)>0)
    for p,c in sorted(plans.items()): print(f"    {INFO} {p}: {c}")
    for coll in ["users","trading_results","contact_submissions"]:
        list(db.collection(coll).limit(1).stream())
        check(f"Collection: {coll}", True)
except ImportError: skip("Firestore", "not installed")
except Exception as e: check("Firestore", False, str(e), warn_only=True)

# ═══════════════════════════════════════════════════════════════
section("12. REQUIREMENTS.TXT")
if os.path.isfile("requirements.txt"):
    with open("requirements.txt") as f: reqs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    check(f"{len(reqs)} packages", len(reqs)>10)
    for pkg in ["dash","plotly","numpy","pandas","stripe","gunicorn"]:
        check(f"'{pkg}'", any(pkg in l.lower() for l in reqs), "Missing")
else: check("requirements.txt", False, "MISSING")

# ═══════════════════════════════════════════════════════════════
section("13. CLOUD RUN DEPLOYMENT")
try:
    r = subprocess.run(["gcloud","run","services","describe","tradeanalytics","--region","us-central1",
                        "--format","value(status.url)"], capture_output=True, text=True, timeout=15)
    if r.returncode==0 and r.stdout.strip():
        url = r.stdout.strip()
        print(f"  {INFO} URL: {url}")
        check("Service active", True)
    else: check("Service", False, warn_only=True)

    r = subprocess.run(["gcloud","builds","list","--limit=1","--format=value(status,createTime)"],
                       capture_output=True, text=True, timeout=15)
    if r.returncode==0 and r.stdout.strip():
        parts = r.stdout.strip().split("\t")
        check(f"Last build: {parts[0]} ({parts[1] if len(parts)>1 else '?'})", parts[0]=="SUCCESS")

    r = subprocess.run(["gcloud","run","services","describe","tradeanalytics","--region","us-central1",
                        "--format=value(spec.template.spec.containers[0].env)"],
                       capture_output=True, text=True, timeout=15)
    if r.returncode==0:
        env = r.stdout
        for v in ["STRIPE_SECRET_KEY","FMP_API_KEY","APP_DOMAIN"]:
            check(f"Cloud Run: {v}", v in env, f"Not set on Cloud Run", warn_only=True)

    r = subprocess.run(["git","status","--short"], capture_output=True, text=True, timeout=5)
    if r.returncode==0:
        uc = r.stdout.strip()
        if uc:
            check(f"Git: {uc.count(chr(10))+1} uncommitted", False, "Deploy needed", warn_only=True)
        else: check("Git: clean", True)
except FileNotFoundError: skip("Cloud Run", "gcloud not available")
except Exception as e: check("Cloud Run", False, str(e), warn_only=True)

# ═══════════════════════════════════════════════════════════════
section("14. LIVE HTTP CHECK")
url = os.environ.get("APP_DOMAIN","")
if not url:
    try:
        r = subprocess.run(["gcloud","run","services","describe","tradeanalytics","--region","us-central1",
                            "--format=value(status.url)"], capture_output=True, text=True, timeout=15)
        if r.returncode==0: url = r.stdout.strip()
    except: pass
if url:
    try:
        import requests
        resp = requests.get(url, timeout=20)
        check(f"GET {url} → {resp.status_code}", resp.status_code==200)
        check(f"Response: {len(resp.content):,} bytes", len(resp.content)>1000)
        check("Contains Dash app", "dash" in resp.text.lower() or "_dash-" in resp.text)
    except Exception as e: check("HTTP", False, str(e))
else: skip("HTTP check", "No URL")

# ═══════════════════════════════════════════════════════════════
section("PRE-LAUNCH SUMMARY")
total = sum(results.values())
testable = total - results["skip"]
score = results["pass"] / max(testable,1) * 100
print(f"""
  {PASS} Passed:   {results['pass']}
  {FAIL} Failed:   {results['fail']}
  {WARN} Warnings: {results['warn']}
  {SKIP} Skipped:  {results['skip']}
  ────────────────────
  Total:  {total} checks
  Score:  {score:.0f}%
""")
if results["fail"]==0 and results["warn"]<=5: print(f"  🟢 READY FOR LAUNCH")
elif results["fail"]==0: print(f"  🟡 LAUNCH OK — {results['warn']} minor warnings")
elif results["fail"]<=3: print(f"  🟠 FIX FIRST — {results['fail']} issues")
else: print(f"  🔴 NOT READY — {results['fail']} failures")
print(f"\n  {INFO} {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print(f"{'═'*64}\n")
sys.exit(0 if results["fail"]==0 else 1)
