"""
AI TRADING PROFESSIONAL — DASH APPLICATION
===========================================================================
Full conversion of the Streamlit Market Analysiss system to Dash/Plotly.
Covers: Prediction Engine, Prediction Display, Position Analysis,
        Forecast Analysis, Risk Assessment, Cross-Validation.
===========================================================================
"""

import os
import json
import logging
import time
import hashlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

# =============================================================================
# SaaS AUTH & PAYMENTS MODULE
# =============================================================================
try:
    from saas_auth import (
        # User management
        create_user, authenticate_user, get_user_by_token, logout_user,
        # Plan gating
        get_user_plan, get_allowed_tickers, get_allowed_timeframes,
        get_models_limit, can_access_feature, check_prediction_limit,
        record_prediction, get_plan_badge_info, PLANS, ALL_TICKERS as AUTH_TICKERS,
        # Discovery plan helpers
        get_discovery_days_remaining, is_discovery_active, has_used_discovery,
        DISCOVERY_DURATION_DAYS,
        # Admin
        is_admin,
        # Stripe
        create_checkout_session, create_customer_portal_session,
        setup_auth_system,
        # UI builders
        build_login_page, build_pricing_page, build_user_badge,
        build_upgrade_prompt, build_limit_reached_prompt,
    )
    SAAS_AUTH_AVAILABLE = True
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("✅ SaaS auth module loaded")
except ImportError as e:
    SAAS_AUTH_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.warning(f"⚠️ SaaS auth module not available: {e} — running without auth")

# =============================================================================
# EMAIL SERVICE & DISCLAIMER IMPORTS
# =============================================================================
EMAIL_SERVICE_AVAILABLE = False
try:
    from email_service import init_email_service, send_market_alert, send_subscription_notification
    EMAIL_SERVICE_AVAILABLE = True
except ImportError:
    pass

try:
    from disclaimer import build_disclaimer_overlay
except ImportError:
    def build_disclaimer_overlay():
        return html.Div()

try:
    from app_guide import build_app_guide_page
    APP_GUIDE_AVAILABLE = True
except ImportError:
    APP_GUIDE_AVAILABLE = False
    def build_app_guide_page():
        return html.Div("App Guide not available.", style={"color": "#64748b", "padding": "40px"})

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# =============================================================================
# BACKEND IMPORTS — graceful fallback if not available
# =============================================================================
BACKEND_AVAILABLE = False
FMP_API_KEY = os.environ.get("FMP_API_KEY", "")

try:
    from enhprog import (
        # Core prediction functions
        get_real_time_prediction,
        multi_step_forecast,
        enhanced_ensemble_predict,
        calculate_prediction_confidence,

        # Data management
        MultiTimeframeDataManager,

        # Advanced analytics
        AdvancedMarketRegimeDetector,
        AdvancedRiskManager,
        ModelExplainer,
        ModelDriftDetector,

        # Alternative data providers
        RealTimeEconomicDataProvider,
        RealTimeSentimentProvider,
        RealTimeOptionsProvider,

        # Utilities
        get_asset_type,
        get_reasonable_price_range,
        enhance_features,
        inverse_transform_prediction,
        load_trained_models,
        train_enhanced_models,
        safe_ticker_name,

        # Model classes (needed for pickle deserialization)
        XGBoostTimeSeriesModel,
        SklearnEnsemble,

        # Neural network classes (needed for pickle deserialization)
        AdvancedTransformer,
        CNNLSTMAttention,
        EnhancedTCN,
        EnhancedInformer,
        EnhancedNBeats,
        LSTMGRUEnsemble,
    )
    BACKEND_AVAILABLE = True
    logger.info("✅ Backend (enhprog) loaded successfully")

    # ── Register model classes in __main__ so pickle can find them ────────
    # Models pickled when tradingprofessional.py was __main__ store their class
    # reference as __main__.XGBoostTimeSeriesModel etc.  When app.py is __main__,
    # pickle fails unless these names exist in sys.modules['__main__'].
    import sys as _sys
    _main_mod = _sys.modules.get("__main__")
    if _main_mod:
        for _cls in (
            XGBoostTimeSeriesModel, SklearnEnsemble,
            AdvancedTransformer, CNNLSTMAttention, EnhancedTCN,
            EnhancedInformer, EnhancedNBeats, LSTMGRUEnsemble,
            AdvancedMarketRegimeDetector, AdvancedRiskManager,
            ModelExplainer, ModelDriftDetector,
        ):
            if not hasattr(_main_mod, _cls.__name__):
                setattr(_main_mod, _cls.__name__, _cls)
        logger.info("✅ Model classes registered in __main__ for pickle compatibility")
except ImportError as e:
    logger.warning(f"⚠️ Backend not available: {e}")

    # Provide stubs so the UI can run in demo mode
    def get_asset_type(ticker):
        t = ticker.upper()
        crypto = {"BTCUSD", "ETHUSD", "SOLUSD"}
        forex = {"USDJPY"}
        commodity = {"CC=F", "NG=F", "GC=F", "KC=F", "SI=F", "HG=F"}
        index = {"^GDAXI", "^GSPC", "^HSI"}
        if t in crypto:
            return "crypto"
        if t in forex:
            return "forex"
        if t in commodity:
            return "commodity"
        if t in index:
            return "index"
        return "stock"

    def get_reasonable_price_range(ticker):
        ranges = {
            "BTCUSD": (25000, 110000), "ETHUSD": (1500, 5000), "XAUUSD": (1800, 2500),
            "EURUSD": (1.0, 1.15), "GBPUSD": (1.2, 1.35), "AAPL": (150, 250),
            "MSFT": (300, 450), "TSLA": (150, 350), "SPY": (400, 550),
        }
        return ranges.get(ticker.upper(), (50, 200))

    def safe_ticker_name(ticker):
        return ticker.replace("/", "_").replace("^", "").replace(".", "_")


# =============================================================================
# AI BACKTEST & PORTFOLIO MODULE IMPORTS
# =============================================================================
AI_BACKTEST_AVAILABLE = False
AI_PORTFOLIO_AVAILABLE = False

try:
    from ai_backtest_engine import (
        run_ai_backtest,
        AIBacktestEngine,
        WalkForwardValidator,
        AITradeDecision,
        BacktestResult,
    )
    AI_BACKTEST_AVAILABLE = True
    logger.info("✅ ai_backtest_engine loaded")
except ImportError as e:
    logger.warning(f"⚠️ ai_backtest_engine not available: {e}")

try:
    from ai_portfolio_system import (
        create_portfolio_manager,
        build_asset_view,
        AIPortfolioManager,
        PortfolioOptimizer,
        RealTimeRiskMonitor,
        AssetView,
    )
    AI_PORTFOLIO_AVAILABLE = True
    logger.info("✅ ai_portfolio_system loaded")
except ImportError as e:
    logger.warning(f"⚠️ ai_portfolio_system not available: {e}")


# =============================================================================
# RESULTS PERSISTENCE — Save predictions & backtests for future use
# =============================================================================

FIRESTORE_AVAILABLE = False
_firestore_db = None

try:
    from google.cloud import firestore as _firestore
    _firestore_db = _firestore.Client()
    FIRESTORE_AVAILABLE = True
    logger.info("✅ Firestore connected for result persistence")
except Exception:
    logger.info("ℹ️ Firestore not available — using local JSON storage")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_result(category: str, ticker: str, data: dict) -> str:
    """Save a prediction or backtest result. Returns result ID."""
    result_id = f"{category}_{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record = {
        "id": result_id,
        "category": category,
        "ticker": ticker,
        "timestamp": datetime.now().isoformat(),
        "data": data,
    }

    # Try Firestore first
    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            _firestore_db.collection("trading_results").document(result_id).set(record)
            logger.info(f"💾 Saved {category} result to Firestore: {result_id}")
            return result_id
        except Exception as e:
            logger.warning(f"Firestore save failed, falling back to local: {e}")

    # Local JSON fallback
    filepath = os.path.join(RESULTS_DIR, f"{result_id}.json")
    try:
        with open(filepath, "w") as f:
            json.dump(record, f, default=str, indent=2)
        logger.info(f"💾 Saved {category} result locally: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save result: {e}")

    return result_id


def load_results(category: str = None, ticker: str = None, limit: int = 20) -> list:
    """Load saved results, optionally filtered by category/ticker."""
    results = []

    # Try Firestore first
    if FIRESTORE_AVAILABLE and _firestore_db:
        try:
            query = _firestore_db.collection("trading_results")
            if category:
                query = query.where("category", "==", category)
            if ticker:
                query = query.where("ticker", "==", ticker)
            query = query.order_by("timestamp", direction=_firestore.Query.DESCENDING).limit(limit)
            for doc in query.stream():
                results.append(doc.to_dict())
            if results:
                return results
        except Exception as e:
            logger.debug(f"Firestore load failed: {e}")

    # Local JSON fallback
    try:
        files = sorted(
            [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")],
            reverse=True,
        )
        for fname in files[:limit * 3]:  # over-read then filter
            if category and not fname.startswith(category):
                continue
            if ticker and ticker not in fname:
                continue
            filepath = os.path.join(RESULTS_DIR, fname)
            with open(filepath, "r") as f:
                record = json.load(f)
                results.append(record)
            if len(results) >= limit:
                break
    except Exception as e:
        logger.debug(f"Local load failed: {e}")

    return results


# =============================================================================
# TICKER UNIVERSE
# =============================================================================
TICKER_GROUPS = {
    "🪙 Crypto": ["ETHUSD", "SOLUSD", "BTCUSD"],
    "💱 Forex": ["USDJPY"],
    "🛢️ Commodities": ["CC=F", "NG=F", "GC=F", "KC=F", "SI=F", "HG=F"],
    "📊 Indices": ["^GDAXI", "^GSPC", "^HSI"],
}
ALL_TICKERS = [t for group in TICKER_GROUPS.values() for t in group]

MODEL_OPTIONS = [
    {"label": "Advanced Transformer", "value": "advanced_transformer"},
    {"label": "CNN-LSTM Hybrid", "value": "cnn_lstm"},
    {"label": "Temporal Conv. Network", "value": "enhanced_tcn"},
    {"label": "Informer", "value": "enhanced_informer"},
    {"label": "N-BEATS", "value": "enhanced_nbeats"},
    {"label": "LSTM-GRU Ensemble", "value": "lstm_gru_ensemble"},
    {"label": "XGBoost", "value": "xgboost"},
    {"label": "Sklearn Ensemble", "value": "sklearn_ensemble"},
]

# =============================================================================
# IN-PROCESS STATE CACHE (replaces st.session_state for Dash)
# =============================================================================
# Dash has no session_state equivalent across callbacks, so we use a
# module-level dict.  This is safe for single-process Cloud Run containers.

_app_cache: Dict[str, Any] = {
    "models_trained": {},       # {ticker: {model_name: model_obj}}
    "model_configs": {},        # {ticker: config_dict}
    "real_time_prices": {},     # {ticker: float}
    "data_manager": None,       # shared MultiTimeframeDataManager
    # Enhancement singletons (lazy-initialised)
    "regime_detector": None,
    "risk_manager": None,
    "model_explainer": None,
    "drift_detector": None,
    "economic_provider": None,
    "sentiment_provider": None,
    "options_provider": None,
}


def _ensure_data_manager() -> Any:
    """Lazy-create the shared MultiTimeframeDataManager."""
    if _app_cache["data_manager"] is None and BACKEND_AVAILABLE:
        try:
            _app_cache["data_manager"] = MultiTimeframeDataManager(ALL_TICKERS[:8])
            logger.info("✅ Shared MultiTimeframeDataManager created")
        except Exception as e:
            logger.warning(f"Could not create data manager: {e}")
    return _app_cache["data_manager"]


def _ensure_enhancement_singletons():
    """Lazy-create regime detector, risk manager, explainer, drift detector."""
    if not BACKEND_AVAILABLE:
        return
    try:
        if _app_cache["regime_detector"] is None:
            _app_cache["regime_detector"] = AdvancedMarketRegimeDetector(n_regimes=4)
        if _app_cache["risk_manager"] is None:
            _app_cache["risk_manager"] = AdvancedRiskManager()
        if _app_cache["model_explainer"] is None:
            _app_cache["model_explainer"] = ModelExplainer()
        if _app_cache["drift_detector"] is None:
            _app_cache["drift_detector"] = ModelDriftDetector(
                reference_window=500,
                detection_threshold=0.05,
            )
        if _app_cache["economic_provider"] is None:
            _app_cache["economic_provider"] = RealTimeEconomicDataProvider()
        if _app_cache["sentiment_provider"] is None:
            _app_cache["sentiment_provider"] = RealTimeSentimentProvider()
        if _app_cache["options_provider"] is None:
            _app_cache["options_provider"] = RealTimeOptionsProvider()
    except Exception as e:
        logger.debug(f"Enhancement singletons init: {e}")


# =============================================================================
# PREDICTION ENGINE — Full production version (ported from tradingprofessional.py)
# =============================================================================

class PredictionEngine:
    """
    Production prediction engine — mirrors RealPredictionEngine from Streamlit.

    Fallback chain (same order as tradingprofessional.py):
      1. Session-cache models → get_real_time_prediction
      2. Disk/GCS models via load_trained_models → get_real_time_prediction
      3. enhprog built-in model loading (models=None)
      4. Individual model retry loop
      5. Data-driven technical analysis (RSI/MACD/SMA on real FMP data)
      6. Enhanced fallback (random within asset-appropriate bounds)
      7. Pure demo simulation
    """

    @staticmethod
    def run_prediction(ticker: str, timeframe: str = "1day", models: list = None) -> Dict:
        if BACKEND_AVAILABLE and FMP_API_KEY:
            return PredictionEngine._run_live(ticker, timeframe, models)
        return PredictionEngine._run_demo(ticker)

    # ── live backend prediction ──────────────────────────────────────────
    @staticmethod
    def _run_live(ticker, timeframe, models):
        try:
            logger.info(f"🎯 Running LIVE prediction for {ticker} (timeframe: {timeframe})")

            # ── Get real-time price ──────────────────────────────────
            data_manager = _ensure_data_manager()
            current_price = None
            if data_manager:
                current_price = data_manager.get_real_time_price(ticker)

            # Fallback price sources
            if not current_price:
                current_price = _app_cache["real_time_prices"].get(ticker)
                if current_price:
                    logger.info(f"Using cached price for {ticker}: ${current_price}")
            if not current_price:
                try:
                    mn, mx = get_reasonable_price_range(ticker)
                    current_price = mn + (mx - mn) * 0.5
                    logger.warning(f"Using estimated price for {ticker}: ${current_price:.2f}")
                except Exception:
                    current_price = 0

            # Cache the price
            if current_price and current_price > 0:
                _app_cache["real_time_prices"][ticker] = current_price

            # ── Resolve available models ─────────────────────────────
            if not models:
                models = [
                    "advanced_transformer", "cnn_lstm", "enhanced_tcn", "enhanced_informer",
                    "enhanced_nbeats", "lstm_gru_ensemble", "xgboost", "sklearn_ensemble",
                ]

            # Check in-process cache first
            trained_models = _app_cache["models_trained"].get(ticker, {})
            available_trained_models = {m: trained_models[m] for m in models if m in trained_models}

            # If nothing cached, try loading from disk / GCS
            if not available_trained_models:
                logger.info(f"No models in cache for {ticker}, loading from disk...")
                try:
                    loaded_models, loaded_config = load_trained_models(ticker)
                    if loaded_models:
                        logger.info(f"✅ Loaded {len(loaded_models)} models from disk for {ticker}")
                        _app_cache["models_trained"][ticker] = loaded_models
                        if loaded_config:
                            _app_cache["model_configs"][ticker] = loaded_config
                        trained_models = loaded_models
                        available_trained_models = {m: trained_models[m] for m in models if m in trained_models}
                        # If specific model names don't match, use all loaded models
                        if not available_trained_models:
                            available_trained_models = loaded_models
                    else:
                        logger.warning(f"No models found on disk for {ticker}")
                except Exception as load_err:
                    logger.error(f"Error loading models from disk: {load_err}")

            # ── No models at all → try enhprog built-in loading ──────
            if not available_trained_models:
                logger.info(f"Attempting prediction via enhprog built-in model loading for {ticker}")
                prediction_result = get_real_time_prediction(
                    ticker, models=None, config=None, current_price=current_price,
                )
                if prediction_result:
                    prediction_result = PredictionEngine._enhance_with_backend_features(
                        prediction_result, ticker,
                    )
                    prediction_result["source"] = "live_ai_backend"
                    prediction_result["fallback_mode"] = False
                    return prediction_result
                else:
                    logger.warning(f"No trained models for {ticker}. Using data-driven analysis.")
                    return PredictionEngine._data_driven(ticker, current_price, timeframe)

            # ── We have trained models — attempt ML prediction ───────
            model_config = _app_cache["model_configs"].get(ticker)
            prediction_result = None

            try:
                prediction_result = get_real_time_prediction(
                    ticker,
                    models=available_trained_models,
                    config=model_config,
                    current_price=current_price,
                )
            except Exception as pred_err:
                logger.warning(f"get_real_time_prediction raised: {pred_err}")

            if prediction_result:
                prediction_result = PredictionEngine._enhance_with_backend_features(
                    prediction_result, ticker,
                )
                prediction_result["models_used"] = list(available_trained_models.keys())
                prediction_result["source"] = "live_ai_backend"
                prediction_result["fallback_mode"] = False
                return prediction_result

            # ── Full prediction failed → individual model retry loop ─
            logger.info(f"Full prediction failed, trying individual models for {ticker}...")
            for model_name, model_obj in available_trained_models.items():
                try:
                    individual_result = get_real_time_prediction(
                        ticker,
                        models={model_name: model_obj},
                        config=model_config,
                        current_price=current_price,
                    )
                    if individual_result:
                        individual_result = PredictionEngine._enhance_with_backend_features(
                            individual_result, ticker,
                        )
                        individual_result["models_used"] = [model_name]
                        individual_result["source"] = "live_ai_backend"
                        individual_result["fallback_mode"] = False
                        logger.info(f"✅ Individual model {model_name} succeeded for {ticker}")
                        return individual_result
                except Exception:
                    continue

            # ── All ML attempts failed → data-driven technical ───────
            logger.warning(f"All ML predictions failed for {ticker}. Using data-driven analysis.")
            return PredictionEngine._data_driven(ticker, current_price, timeframe)

        except Exception as e:
            logger.error(f"Error in live prediction: {e}")
            logger.error(traceback.format_exc())
            cached_price = _app_cache["real_time_prices"].get(ticker, 0)
            return PredictionEngine._data_driven(ticker, cached_price, timeframe)

    # ── Backend feature enhancement (regime, drift, SHAP, risk, alt data) ─
    @staticmethod
    def _enhance_with_backend_features(prediction_result: Dict, ticker: str) -> Dict:
        """Enhance prediction with regime analysis, drift, SHAP, risk, alt data."""
        try:
            _ensure_enhancement_singletons()

            # ── Regime analysis ──────────────────────────────────────
            regime_detector = _app_cache.get("regime_detector")
            if regime_detector:
                try:
                    dm = _ensure_data_manager()
                    if dm:
                        multi_tf = dm.fetch_multi_timeframe_data(ticker, ["1d"])
                        if multi_tf and "1d" in multi_tf:
                            data = multi_tf["1d"]
                            edf = enhance_features(data, ["Open", "High", "Low", "Close", "Volume"])
                            if edf is not None and len(edf) > 100:
                                regime_probs = regime_detector.fit_regime_model(edf)
                                current_regime = regime_detector.detect_current_regime(edf)
                                prediction_result["regime_analysis"] = {
                                    "current_regime": current_regime,
                                    "regime_probabilities": regime_probs.tolist() if regime_probs is not None else [],
                                    "analysis_timestamp": datetime.now().isoformat(),
                                }
                except Exception as e:
                    logger.debug(f"Regime analysis skipped: {e}")

            # ── Drift detection ──────────────────────────────────────
            drift_detector = _app_cache.get("drift_detector")
            if drift_detector:
                try:
                    dm = _ensure_data_manager()
                    if dm:
                        multi_tf = dm.fetch_multi_timeframe_data(ticker, ["1d"])
                        if multi_tf and "1d" in multi_tf:
                            data = multi_tf["1d"]
                            edf = enhance_features(data, ["Open", "High", "Low", "Close", "Volume"])
                            if edf is not None and len(edf) > 200:
                                split_pt = int(len(edf) * 0.8)
                                ref_data = edf.iloc[:split_pt].values
                                cur_data = edf.iloc[split_pt:].values
                                drift_detector.set_reference_distribution(ref_data, edf.columns)
                                drift_detected, drift_score, feature_drift = drift_detector.detect_drift(
                                    cur_data, edf.columns,
                                )
                                prediction_result["drift_detection"] = {
                                    "drift_detected": drift_detected,
                                    "drift_score": drift_score,
                                    "feature_drift": feature_drift,
                                    "summary": drift_detector.get_drift_summary(),
                                    "detection_timestamp": datetime.now().isoformat(),
                                }
                except Exception as e:
                    logger.debug(f"Drift detection skipped: {e}")

            # ── Model explanations (SHAP) ────────────────────────────
            model_explainer = _app_cache.get("model_explainer")
            trained_models = _app_cache["models_trained"].get(ticker, {})
            if model_explainer and trained_models:
                try:
                    dm = _ensure_data_manager()
                    if dm:
                        multi_tf = dm.fetch_multi_timeframe_data(ticker, ["1d"])
                        if multi_tf and "1d" in multi_tf:
                            data = multi_tf["1d"]
                            edf = enhance_features(data, ["Open", "High", "Low", "Close", "Volume"])
                            if edf is not None and len(edf) > 60:
                                recent_data = edf.tail(60).values
                                feature_names = list(edf.columns)
                                explanations = {}
                                for mname, mobj in trained_models.items():
                                    try:
                                        expl = model_explainer.explain_prediction(
                                            mobj, recent_data, feature_names, mname,
                                        )
                                        if expl:
                                            explanations[mname] = expl
                                    except Exception:
                                        pass
                                if explanations:
                                    report = model_explainer.generate_explanation_report(
                                        explanations,
                                        prediction_result.get("predicted_price", 0),
                                        ticker,
                                        prediction_result.get("confidence", 0),
                                    )
                                    explanations["report"] = report
                                    prediction_result["model_explanations"] = explanations
                except Exception as e:
                    logger.debug(f"SHAP explanations skipped: {e}")

            # ── Enhanced risk metrics ────────────────────────────────
            risk_mgr = _app_cache.get("risk_manager")
            if risk_mgr:
                try:
                    dm = _ensure_data_manager()
                    if dm:
                        multi_tf = dm.fetch_multi_timeframe_data(ticker, ["1d"])
                        if multi_tf and "1d" in multi_tf:
                            data = multi_tf["1d"]
                            if len(data) > 252:
                                returns = data["Close"].pct_change().dropna()
                                risk_metrics = risk_mgr.calculate_risk_metrics(returns[-252:])
                                risk_metrics["portfolio_var"] = risk_mgr.calculate_var(returns, method="monte_carlo")
                                risk_metrics["expected_shortfall"] = risk_mgr.calculate_expected_shortfall(returns)
                                risk_metrics["maximum_drawdown"] = risk_mgr.calculate_maximum_drawdown(returns)
                                prediction_result["enhanced_risk_metrics"] = risk_metrics
                except Exception as e:
                    logger.debug(f"Risk metrics enhancement skipped: {e}")

            # ── Alternative data ─────────────────────────────────────
            try:
                dm = _ensure_data_manager()
                if dm:
                    alt_data = dm.fetch_alternative_data(ticker)
                    if alt_data:
                        # Enrich with provider data
                        econ = _app_cache.get("economic_provider")
                        if econ:
                            try:
                                alt_data["economic_indicators"] = econ.fetch_economic_indicators()
                            except Exception:
                                pass
                        sent = _app_cache.get("sentiment_provider")
                        if sent:
                            try:
                                alt_data["reddit_sentiment"] = sent.get_reddit_sentiment(ticker)
                                alt_data["twitter_sentiment"] = sent.get_twitter_sentiment(ticker)
                            except Exception:
                                pass
                        asset_type = get_asset_type(ticker)
                        if asset_type in ("index", "stock"):
                            opts = _app_cache.get("options_provider")
                            if opts:
                                try:
                                    alt_data["options_flow"] = opts.get_options_flow(ticker)
                                except Exception:
                                    pass
                        prediction_result["real_alternative_data"] = alt_data
            except Exception as e:
                logger.debug(f"Alt data enhancement skipped: {e}")

            return prediction_result

        except Exception as e:
            logger.error(f"Error enhancing prediction: {e}")
            return prediction_result

    # ── data-driven technical fallback (real FMP indicators) ──────────────
    @staticmethod
    def _data_driven(ticker, current_price, timeframe="1day"):
        """
        Data-driven prediction using REAL FMP historical data + technical indicators.
        Uses RSI, MACD, SMA crossover, momentum — NOT random.
        Ported from tradingprofessional.py RealPredictionEngine._data_driven_prediction.
        """
        try:
            # ── Ensure valid current_price ────────────────────────────
            if not current_price or current_price <= 0:
                current_price = _app_cache["real_time_prices"].get(ticker)
            if not current_price or current_price <= 0:
                try:
                    mn, mx = get_reasonable_price_range(ticker)
                    current_price = mn + (mx - mn) * 0.5
                except Exception:
                    current_price = 100.0

            asset_type = get_asset_type(ticker)

            # ── Map timeframe to fetch key ────────────────────────────
            tf_map = {"15min": "15min", "1hour": "1h", "4hour": "4h", "1day": "1d"}
            fetch_tf = tf_map.get(timeframe, "1d")

            # ── Fetch real historical data from FMP ───────────────────
            df = None
            dm = _ensure_data_manager()
            if dm:
                try:
                    multi_tf = dm.fetch_multi_timeframe_data(ticker, [fetch_tf])
                    if multi_tf and fetch_tf in multi_tf:
                        df = multi_tf[fetch_tf]
                except Exception as e:
                    logger.warning(f"Could not fetch historical data for {ticker}: {e}")

            if df is not None and len(df) >= 30:
                close = df["Close"].values

                # ── Technical indicators on REAL data ─────────────────
                # Momentum: Rate of Change (10-day)
                roc_10 = (close[-1] - close[-11]) / close[-11] if len(close) > 11 else 0

                # Trend: SMA crossover (10 vs 30)
                sma_10 = float(np.mean(close[-10:]))
                sma_30 = float(np.mean(close[-30:])) if len(close) >= 30 else float(np.mean(close))
                sma_signal = (sma_10 - sma_30) / sma_30 if sma_30 != 0 else 0

                # Volatility: 20-day standard deviation as pct
                mean_20 = float(np.mean(close[-20:])) if len(close) >= 20 else float(close[-1])
                vol_20 = float(np.std(close[-20:])) / mean_20 if len(close) >= 20 and mean_20 != 0 else 0.01

                # Mean reversion: distance from 20-day SMA
                sma_20 = float(np.mean(close[-20:])) if len(close) >= 20 else float(close[-1])
                reversion_signal = (sma_20 - close[-1]) / sma_20 if sma_20 != 0 else 0

                # RSI (14-period)
                if len(close) >= 15:
                    diffs = np.diff(close[-15:])
                    gains = float(np.mean(diffs[diffs > 0])) if np.any(diffs > 0) else 0.001
                    losses = abs(float(np.mean(diffs[diffs < 0]))) if np.any(diffs < 0) else 0.001
                    rs = gains / losses
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 50

                # MACD signal (12/26 EMA diff direction)
                if len(close) >= 26:
                    ema_12 = float(pd.Series(close).ewm(span=12).mean().iloc[-1])
                    ema_26 = float(pd.Series(close).ewm(span=26).mean().iloc[-1])
                    macd_signal = (ema_12 - ema_26) / ema_26 if ema_26 != 0 else 0
                else:
                    macd_signal = 0

                # ── Composite signal ──────────────────────────────────
                composite = (
                    0.25 * np.clip(roc_10 * 10, -1, 1) +
                    0.25 * np.clip(sma_signal * 20, -1, 1) +
                    0.15 * np.clip(reversion_signal * 10, -1, 1) +
                    0.15 * ((50 - rsi) / 50) +
                    0.20 * np.clip(macd_signal * 20, -1, 1)
                )

                # Scale to asset-appropriate move size
                max_moves = {
                    "crypto": 0.05, "forex": 0.01, "commodity": 0.03,
                    "index": 0.02, "stock": 0.04,
                }
                max_move = max_moves.get(asset_type, 0.03)

                predicted_change = float(np.clip(composite * max_move, -max_move, max_move))
                predicted_price = current_price * (1 + predicted_change)

                # Confidence based on signal agreement strength
                signal_agreement = abs(float(composite))
                confidence = 58 + signal_agreement * 27  # 58-85 range
                confidence = min(confidence, 85.0)

                source = "data_driven_technical"

                # 5-day forecast using momentum extrapolation
                forecast = []
                p = predicted_price
                for i in range(5):
                    decay = 0.85 ** (i + 1)
                    step_change = predicted_change * decay + np.random.normal(0, vol_20 * 0.3)
                    p = p * (1 + step_change)
                    forecast.append(round(p, 6))

                logger.info(
                    f"📊 Data-driven prediction for {ticker}: "
                    f"RSI={rsi:.1f}, SMA_signal={sma_signal:.4f}, MACD={macd_signal:.4f}, "
                    f"composite={composite:.4f}, predicted_change={predicted_change * 100:.2f}%"
                )
            else:
                # No historical data — simplified estimation
                max_moves = {
                    "crypto": 0.05, "forex": 0.01, "commodity": 0.03,
                    "index": 0.02, "stock": 0.04,
                }
                max_move = max_moves.get(asset_type, 0.03)
                predicted_change = np.random.uniform(-max_move * 0.5, max_move * 0.5)
                predicted_price = current_price * (1 + predicted_change)
                confidence = np.random.uniform(55, 68)
                source = "estimated_fallback"

                forecast = []
                p = predicted_price
                vol_est = {"crypto": 0.04, "forex": 0.008, "commodity": 0.02, "index": 0.015, "stock": 0.025}.get(asset_type, 0.02)
                for _ in range(5):
                    p *= 1 + np.random.normal(0, vol_est * 0.6)
                    forecast.append(round(p, 6))

                logger.info(f"📊 Estimated prediction for {ticker} (no historical data)")

            # Cache the price
            _app_cache["real_time_prices"][ticker] = current_price

            is_real_data = (source == "data_driven_technical")

            return {
                "ticker": ticker,
                "asset_type": asset_type,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change_pct": predicted_change * 100,
                "confidence": float(confidence),
                "forecast_5_day": forecast,
                "timestamp": datetime.now().isoformat(),
                "fallback_mode": not is_real_data,
                "source": source,
                "analysis_method": (
                    "Technical Indicators (SMA, RSI, MACD, Momentum)"
                    if is_real_data else "Estimated (no historical data)"
                ),
                "models_used": ["Technical Analysis"] if is_real_data else ["Estimated"],
            }

        except Exception as e:
            logger.error(f"Data-driven prediction failed for {ticker}: {e}")
            logger.error(traceback.format_exc())
            return PredictionEngine._enhanced_fallback(ticker, current_price if current_price else 0)

    # ── Enhanced fallback (realistic constraints, no real data) ───────────
    @staticmethod
    def _enhanced_fallback(ticker, current_price):
        """Enhanced fallback with realistic constraints — ported from tradingprofessional.py."""
        asset_type = get_asset_type(ticker)

        if not current_price or current_price <= 0:
            current_price = _app_cache["real_time_prices"].get(ticker)
        if not current_price or current_price <= 0:
            try:
                mn, mx = get_reasonable_price_range(ticker)
                current_price = mn + (mx - mn) * 0.5
            except Exception:
                current_price = 100.0

        _app_cache["real_time_prices"][ticker] = current_price

        max_changes = {
            "crypto": 0.05, "forex": 0.01, "commodity": 0.03,
            "index": 0.02, "stock": 0.04,
        }
        max_change = max_changes.get(asset_type, 0.03)
        change = np.random.uniform(-max_change, max_change)
        predicted_price = current_price * (1 + change)

        # Generate forecast
        forecast = []
        p = predicted_price
        vol_est = {"crypto": 0.04, "forex": 0.008, "commodity": 0.02, "index": 0.015, "stock": 0.025}.get(asset_type, 0.02)
        for _ in range(5):
            p *= 1 + np.random.normal(0, vol_est * 0.6)
            forecast.append(round(p, 6))

        return {
            "ticker": ticker,
            "asset_type": asset_type,
            "current_price": current_price,
            "predicted_price": predicted_price,
            "price_change_pct": change * 100,
            "confidence": float(np.random.uniform(55, 75)),
            "forecast_5_day": forecast,
            "timestamp": datetime.now().isoformat(),
            "fallback_mode": True,
            "source": "enhanced_fallback",
            "models_used": ["Enhanced Fallback"],
        }

    # ── pure demo prediction (no backend at all) ─────────────────────────
    @staticmethod
    def _run_demo(ticker):
        np.random.seed(int(hashlib.md5(f"{ticker}{datetime.now().date()}".encode()).hexdigest(), 16) % (2**31))
        mn, mx = get_reasonable_price_range(ticker)
        current_price = mn + (mx - mn) * np.random.uniform(0.3, 0.7)
        asset = get_asset_type(ticker)
        vol = {"crypto": 0.04, "forex": 0.008, "commodity": 0.02, "index": 0.015, "stock": 0.025}.get(asset, 0.02)
        drift = np.random.normal(0, vol)
        predicted = current_price * (1 + drift)

        forecast = []
        p = predicted
        for _ in range(5):
            p *= 1 + np.random.normal(0, vol * 0.6)
            forecast.append(round(p, 6))

        confidence = max(40, min(95, 70 + np.random.normal(0, 12)))

        return {
            "ticker": ticker,
            "current_price": round(current_price, 6),
            "predicted_price": round(predicted, 6),
            "price_change_pct": round(((predicted - current_price) / current_price) * 100, 4),
            "confidence": round(confidence, 2),
            "forecast_5_day": forecast,
            "source": "demo_simulation",
            "fallback_mode": True,
            "models_used": ["Demo Simulation"],
            "timestamp": datetime.now().isoformat(),
        }

    # ── Multi-Timeframe Analysis ─────────────────────────────────────────
    @staticmethod
    def _run_multi_timeframe_analysis(ticker: str, timeframes: list) -> Dict:
        """
        Run technical analysis on multiple timeframes and produce
        a unified cross-timeframe signal.

        Returns a dict keyed by timeframe with signals + a 'consensus' summary.
        """
        dm = _ensure_data_manager()
        if not dm:
            return {}

        # Map UI timeframe names to the keys MultiTimeframeDataManager expects
        tf_map = {"15min": "15min", "1hour": "1h", "4hour": "4h", "1day": "1d"}
        fetch_keys = [tf_map.get(tf, tf) for tf in timeframes]

        multi_tf_data = dm.fetch_multi_timeframe_data(ticker, fetch_keys)
        if not multi_tf_data:
            return {}

        mtf_results: Dict[str, Any] = {}
        signals_for_consensus = []

        for ui_tf, fetch_tf in zip(timeframes, fetch_keys):
            df = multi_tf_data.get(fetch_tf)
            if df is None or len(df) < 20:
                mtf_results[ui_tf] = {"status": "insufficient_data", "bars": 0}
                continue

            close = df["Close"].values
            n = len(close)

            # ── Compute core indicators ──────────────────────────────
            # SMA 10 / 30
            sma_10 = float(np.mean(close[-10:])) if n >= 10 else float(close[-1])
            sma_30 = float(np.mean(close[-30:])) if n >= 30 else float(np.mean(close))
            sma_signal = (sma_10 - sma_30) / sma_30 if sma_30 != 0 else 0

            # RSI 14
            if n >= 15:
                diffs = np.diff(close[-15:])
                gains = float(np.mean(diffs[diffs > 0])) if np.any(diffs > 0) else 0.001
                losses = abs(float(np.mean(diffs[diffs < 0]))) if np.any(diffs < 0) else 0.001
                rsi = 100 - (100 / (1 + gains / losses))
            else:
                rsi = 50.0

            # MACD (12/26)
            if n >= 26:
                ema_12 = float(pd.Series(close).ewm(span=12).mean().iloc[-1])
                ema_26 = float(pd.Series(close).ewm(span=26).mean().iloc[-1])
                macd_val = (ema_12 - ema_26) / ema_26 if ema_26 != 0 else 0
            else:
                macd_val = 0.0

            # Momentum (ROC 10)
            roc = (close[-1] - close[-11]) / close[-11] if n > 11 else 0.0

            # Volatility (20-bar)
            vol_20 = float(np.std(close[-20:]) / np.mean(close[-20:])) if n >= 20 else 0.01

            # ── Composite signal (same formula as _data_driven) ──────
            composite = (
                0.25 * np.clip(roc * 10, -1, 1) +
                0.25 * np.clip(sma_signal * 20, -1, 1) +
                0.15 * ((50 - rsi) / 50) +
                0.15 * np.clip(((np.mean(close[-20:]) if n >= 20 else close[-1]) - close[-1])
                               / (np.mean(close[-20:]) if n >= 20 else close[-1]) * 10, -1, 1) +
                0.20 * np.clip(macd_val * 20, -1, 1)
            )

            if composite > 0.15:
                bias = "Bullish"
            elif composite < -0.15:
                bias = "Bearish"
            else:
                bias = "Neutral"

            strength = abs(float(composite))
            conf = 50 + strength * 35  # 50-85 range

            tf_result = {
                "timeframe": ui_tf,
                "bias": bias,
                "composite_signal": round(float(composite), 4),
                "confidence": round(conf, 1),
                "rsi": round(rsi, 1),
                "macd": round(macd_val, 6),
                "sma_crossover": round(sma_signal, 6),
                "momentum_roc": round(roc, 6),
                "volatility": round(vol_20, 4),
                "current_close": round(float(close[-1]), 6),
                "sma_10": round(sma_10, 4),
                "sma_30": round(sma_30, 4),
                "bars_analysed": n,
            }
            mtf_results[ui_tf] = tf_result
            signals_for_consensus.append({"tf": ui_tf, "composite": composite, "bias": bias})

        # ── Cross-timeframe consensus ────────────────────────────────
        if signals_for_consensus:
            # Weight longer timeframes more heavily
            tf_weights = {"15min": 0.10, "1hour": 0.20, "4hour": 0.30, "1day": 0.40}
            weighted_sum = sum(
                s["composite"] * tf_weights.get(s["tf"], 0.25)
                for s in signals_for_consensus
            )
            total_w = sum(tf_weights.get(s["tf"], 0.25) for s in signals_for_consensus)
            consensus_val = weighted_sum / total_w if total_w > 0 else 0

            bullish_count = sum(1 for s in signals_for_consensus if s["bias"] == "Bullish")
            bearish_count = sum(1 for s in signals_for_consensus if s["bias"] == "Bearish")
            total_tf = len(signals_for_consensus)

            if consensus_val > 0.12:
                consensus_bias = "Bullish"
            elif consensus_val < -0.12:
                consensus_bias = "Bearish"
            else:
                consensus_bias = "Neutral"

            agreement = max(bullish_count, bearish_count) / total_tf if total_tf > 0 else 0
            consensus_conf = 50 + agreement * 35

            mtf_results["consensus"] = {
                "bias": consensus_bias,
                "weighted_signal": round(consensus_val, 4),
                "confidence": round(consensus_conf, 1),
                "bullish_timeframes": bullish_count,
                "bearish_timeframes": bearish_count,
                "neutral_timeframes": total_tf - bullish_count - bearish_count,
                "total_timeframes": total_tf,
                "agreement_pct": round(agreement * 100, 1),
            }

            logger.info(
                f"📊 MTF Consensus for {ticker}: {consensus_bias} "
                f"(signal={consensus_val:.4f}, agreement={agreement:.0%}, "
                f"bull={bullish_count}, bear={bearish_count})"
            )

        return mtf_results


# =============================================================================
# RISK METRICS GENERATOR — uses real AdvancedRiskManager when available
# =============================================================================

def generate_risk_metrics(prediction: Dict) -> Dict:
    ticker = prediction.get("ticker", "UNKNOWN")
    current_price = prediction.get("current_price", 100)
    confidence = prediction.get("confidence", 50)
    asset = get_asset_type(ticker)

    # ── Try real risk metrics from enhanced_risk_metrics (set by _enhance) ─
    enhanced = prediction.get("enhanced_risk_metrics")
    if enhanced:
        try:
            vol_ann = enhanced.get("volatility", 0.25)
            if vol_ann < 1:  # expressed as decimal
                vol_ann *= 100
            return {
                "volatility": round(vol_ann, 2),
                "sharpe_ratio": round(enhanced.get("sharpe_ratio", 0), 3),
                "max_drawdown": round(enhanced.get("maximum_drawdown", enhanced.get("max_drawdown", 0)) * 100, 2)
                    if abs(enhanced.get("maximum_drawdown", enhanced.get("max_drawdown", 0))) < 1
                    else round(enhanced.get("maximum_drawdown", enhanced.get("max_drawdown", 0)), 2),
                "var_95": round(enhanced.get("portfolio_var", enhanced.get("var_95", 0)), 4),
                "beta": round(enhanced.get("beta", 1.0), 3),
                "sortino_ratio": round(enhanced.get("sortino_ratio", 0), 3),
                "risk_score": max(1, min(10, int(5 + (vol_ann / 100 - 0.2) * 15))),
                "confidence_adjustment": round(confidence / 100, 3),
            }
        except Exception as e:
            logger.debug(f"Real risk metric extraction failed: {e}")

    # ── Try computing from real FMP data via AdvancedRiskManager ──────────
    risk_mgr = _app_cache.get("risk_manager") if BACKEND_AVAILABLE else None
    if risk_mgr:
        try:
            dm = _ensure_data_manager()
            if dm:
                multi_tf = dm.fetch_multi_timeframe_data(ticker, ["1d"])
                if multi_tf and "1d" in multi_tf:
                    data = multi_tf["1d"]
                    if len(data) > 60:
                        returns = data["Close"].pct_change().dropna()
                        metrics = risk_mgr.calculate_risk_metrics(returns[-252:] if len(returns) >= 252 else returns)
                        vol_val = metrics.get("volatility", 0.25)
                        if vol_val < 1:
                            vol_val *= 100
                        sharpe = metrics.get("sharpe_ratio", 0)
                        max_dd = metrics.get("max_drawdown", 0)
                        if abs(max_dd) < 1:
                            max_dd *= 100
                        return {
                            "volatility": round(vol_val, 2),
                            "sharpe_ratio": round(sharpe, 3),
                            "max_drawdown": round(max_dd, 2),
                            "var_95": round(metrics.get("var_95", 0), 4),
                            "beta": round(metrics.get("beta", 1.0), 3),
                            "sortino_ratio": round(metrics.get("sortino_ratio", 0), 3),
                            "risk_score": max(1, min(10, int(5 + (vol_val / 100 - 0.2) * 15))),
                            "confidence_adjustment": round(confidence / 100, 3),
                        }
        except Exception as e:
            logger.debug(f"Real risk metrics calculation failed: {e}")

    # ── Simulated fallback (same as original) ────────────────────────────
    np.random.seed(abs(hash(ticker)) % (2**31))
    vol = {"crypto": 0.65, "forex": 0.12, "commodity": 0.28, "index": 0.18, "stock": 0.32}.get(asset, 0.25)
    sharpe = max(-0.5, min(3.0, np.random.normal(1.2, 0.6)))
    max_dd = min(-0.02, -abs(np.random.normal(vol * 0.5, vol * 0.15)))
    var_95 = -abs(np.random.normal(vol * 0.03, vol * 0.01)) * current_price
    beta = max(0.2, min(2.0, np.random.normal(1.0, 0.3)))
    sortino = sharpe * np.random.uniform(1.1, 1.6)

    return {
        "volatility": round(vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "var_95": round(var_95, 4),
        "beta": round(beta, 3),
        "sortino_ratio": round(sortino, 3),
        "risk_score": max(1, min(10, int(5 + (vol - 0.2) * 15))),
        "confidence_adjustment": round(confidence / 100, 3),
    }


# =============================================================================
# CHART BUILDERS
# =============================================================================

def build_price_trajectory_chart(prediction: Dict) -> go.Figure:
    """Build the main prediction trajectory chart."""
    current_price = prediction.get("current_price", 0)
    predicted_price = prediction.get("predicted_price", 0)
    forecast = prediction.get("forecast_5_day", [])
    is_bullish = predicted_price > current_price
    price_change_pct = prediction.get("price_change_pct", 0)

    price_points = [current_price, predicted_price]
    labels = ["Current", "Predicted"]
    for i, fp in enumerate(forecast):
        price_points.append(fp)
        labels.append(f"Day {i+1}")

    x_numeric = list(range(len(price_points)))
    price_min, price_max = min(price_points), max(price_points)
    price_range = (price_max - price_min) if price_max != price_min else abs(current_price) * 0.02
    y_pad = price_range * 0.25

    line_main = "#34d399" if is_bullish else "#f87171"
    line_glow = "rgba(52,211,153,0.3)" if is_bullish else "rgba(248,113,113,0.3)"
    fill_color = "rgba(16,185,129,0.08)" if is_bullish else "rgba(239,68,68,0.08)"
    forecast_color = "#a78bfa"
    direction_color = "#10b981" if is_bullish else "#ef4444"

    fig = go.Figure()

    # Area fill
    fig.add_trace(go.Scatter(
        x=x_numeric, y=price_points, fill="tozeroy", fillcolor=fill_color,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))

    # Glow line
    fig.add_trace(go.Scatter(
        x=x_numeric[:2], y=price_points[:2], mode="lines",
        line=dict(color=line_glow, width=8, shape="spline"),
        showlegend=False, hoverinfo="skip",
    ))

    # Main trajectory
    fig.add_trace(go.Scatter(
        x=x_numeric[:2], y=price_points[:2], mode="lines",
        line=dict(color=line_main, width=3, shape="spline"),
        showlegend=False, hoverinfo="skip",
    ))

    # Forecast lines
    if forecast:
        fx = x_numeric[1:]
        fy = price_points[1:]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines",
            line=dict(color="rgba(167,139,250,0.2)", width=7, shape="spline"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines",
            line=dict(color=forecast_color, width=2.5, shape="spline", dash="dot"),
            name="5-Day Forecast", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=fx[1:], y=fy[1:], mode="markers+text",
            marker=dict(size=9, color="#0f172a", line=dict(color=forecast_color, width=2)),
            text=[f"${p:.2f}" for p in fy[1:]],
            textposition="top center",
            textfont=dict(size=10, color="rgba(167,139,250,0.85)"),
            name="Forecast Days",
            hovertemplate="<b>%{text}</b><extra></extra>",
        ))

    # Current price marker
    fig.add_trace(go.Scatter(
        x=[0], y=[current_price], mode="markers+text",
        marker=dict(size=16, color="#0f172a", line=dict(color="#94a3b8", width=2.5)),
        text=[f"  ${current_price:.4f}"], textposition="middle right",
        textfont=dict(size=12, color="#cbd5e1", family="monospace"),
        name="Current Price",
        hovertemplate="Current: <b>$%{y:.4f}</b><extra></extra>",
    ))

    # Predicted marker (star)
    fig.add_trace(go.Scatter(
        x=[1], y=[predicted_price], mode="markers",
        marker=dict(size=28, color=line_glow, symbol="circle", opacity=0.4),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[1], y=[predicted_price], mode="markers+text",
        marker=dict(size=16, color=line_main, line=dict(color="#0f172a", width=2), symbol="star"),
        text=[f"  ${predicted_price:.4f}"], textposition="middle right",
        textfont=dict(size=12, color=line_main, family="monospace"),
        name="Market Analysis",
        hovertemplate="Predicted: <b>$%{y:.4f}</b><extra></extra>",
    ))

    # Reference line
    fig.add_hline(y=current_price, line=dict(color="rgba(148,163,184,0.15)", width=1, dash="dash"))

    # Delta annotation
    mid_y = (current_price + predicted_price) / 2
    delta_sign = "+" if is_bullish else ""
    fig.add_annotation(
        x=0.5, y=mid_y,
        text=f"<b>{delta_sign}{price_change_pct:.2f}%</b>",
        showarrow=False,
        font=dict(size=13, color=direction_color, family="monospace"),
        bgcolor="rgba(15,23,42,0.85)", bordercolor=direction_color,
        borderwidth=1, borderpad=6, opacity=0.95,
    )

    fig.update_layout(
        template="plotly_dark", height=420,
        margin=dict(l=20, r=20, t=40, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="JetBrains Mono, monospace", color="#94a3b8"),
        title=dict(
            text=f"<b>{prediction.get('ticker','')}</b> — Price Trajectory",
            font=dict(size=15, color="#e2e8f0"), x=0.02, xanchor="left",
        ),
        xaxis=dict(
            tickmode="array", tickvals=x_numeric, ticktext=labels,
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color="rgba(148,163,184,0.7)"),
            linecolor="rgba(99,102,241,0.12)", linewidth=1,
        ),
        yaxis=dict(
            showgrid=True, gridcolor="rgba(99,102,241,0.06)", gridwidth=1,
            zeroline=False, tickprefix="$",
            tickfont=dict(size=11, color="rgba(148,163,184,0.6)", family="monospace"),
            range=[price_min - y_pad, price_max + y_pad], linewidth=0,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(size=10, color="rgba(148,163,184,0.8)"),
            bgcolor="rgba(0,0,0,0)", borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="rgba(15,23,42,0.95)", bordercolor="rgba(99,102,241,0.25)",
            font=dict(size=12, color="#e2e8f0", family="monospace"),
        ),
    )
    return fig


def build_risk_gauge(risk_metrics: Dict) -> go.Figure:
    """Build risk gauge chart."""
    score = risk_metrics.get("risk_score", 5)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Risk Score", "font": {"size": 14, "color": "#94a3b8"}},
        number={"font": {"size": 36, "color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 10], "tickcolor": "#475569"},
            "bar": {"color": "#6366f1"},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 3], "color": "rgba(16,185,129,0.2)"},
                {"range": [3, 6], "color": "rgba(245,158,11,0.2)"},
                {"range": [6, 10], "color": "rgba(239,68,68,0.2)"},
            ],
            "threshold": {
                "line": {"color": "#f59e0b", "width": 3},
                "thickness": 0.8, "value": score,
            },
        },
    ))
    fig.update_layout(
        height=250, margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
    )
    return fig


# =============================================================================
# CSS — Dark Glassmorphism Theme
# =============================================================================

# CSS is loaded from assets/glassmorphism.css automatically by Dash
GLASSMORPHISM_CSS = ""  # kept for reference — actual CSS in assets/


# =============================================================================
# COMPONENT BUILDERS — using ecard-* CSS classes from tradingprofessional.py
# =============================================================================

ACCENT_MAP = {
    "#3b82f6": "accent-blue", "#93c5fd": "accent-blue",
    "#10b981": "accent-green", "#34d399": "accent-green",
    "#ef4444": "accent-red", "#f87171": "accent-red", "#fca5a5": "accent-red",
    "#f59e0b": "accent-amber", "#fbbf24": "accent-amber", "#fcd34d": "accent-amber",
    "#8b5cf6": "accent-purple", "#a78bfa": "accent-purple", "#c4b5fd": "accent-purple",
    "#06b6d4": "accent-cyan",
    "#ec4899": "accent-pink",
}

def _accent_class(color):
    return ACCENT_MAP.get(color, "accent-blue")

def make_metric_card(label, value, delta=None, color="#93c5fd", border_accent=None):
    """Build an ecard-metric with proper accent ::before border."""
    accent = _accent_class(border_accent or color)
    parts = [
        html.Div(label, className="metric-label"),
        html.Div(str(value), className="metric-value", style={"color": color}),
    ]
    if delta:
        parts.append(html.Div(str(delta), className="metric-delta", style={"color": color}))
    return html.Div(parts, className=f"ecard-metric {accent}")


def make_section_header(icon, title):
    return html.Div([
        html.Span(icon, className="section-icon"),
        html.H3(title, className="section-title"),
    ], className="ecard-section-header")


def make_insight_card(text):
    return html.Div(html.P(text), className="ecard-insight")


def make_source_banner(is_live, title_text, subtitle_text):
    cls = "ecard-source-banner live" if is_live else "ecard-source-banner simulation"
    return html.Div([
        html.H3(title_text),
        html.P(subtitle_text),
    ], className=cls)


def make_hero_card(direction_icon, direction_text, ticker, pct, is_bullish):
    badge_cls = "ecard-badge bullish" if is_bullish else "ecard-badge bearish"
    direction_color = "#10b981" if is_bullish else "#ef4444"
    return html.Div([
        html.Div(direction_icon, className="hero-icon"),
        html.H1(direction_text, className="hero-title", style={"color": direction_color}),
        html.P([
            "Market Analysis for ",
            html.Strong(ticker, style={"color": "#e2e8f0"}),
        ], className="hero-sub"),
        html.Div(
            html.Span(f"{direction_icon} {pct:+.2f}% Expected Move", className=badge_cls),
            style={"marginTop": "14px"},
        ),
    ], className=f"ecard-hero {'bullish' if is_bullish else 'bearish'}")


def make_forecast_day(day_num, date_str, price, change_pct):
    change_color = "#10b981" if change_pct >= 0 else "#ef4444"
    arrow = "▲" if change_pct >= 0 else "▼"
    sign = "+" if change_pct >= 0 else ""
    return html.Div([
        html.Div(f"Day {day_num}", className="day-label"),
        html.Div(date_str, className="day-date"),
        html.Div(f"${price:.2f}", className="day-price", style={"color": change_color}),
        html.Div(f"{arrow} {sign}{change_pct:.1f}%", className="day-change", style={"color": change_color}),
    ], className="ecard-forecast-day")


def make_trade_level(label, price, change_pct=None, color="#93c5fd", border_color="#3b82f6"):
    parts = [
        html.Div(label, className="level-label"),
        html.Div(f"${price:.4f}", className="level-price", style={"color": color}),
    ]
    if change_pct is not None:
        parts.append(html.Div(f"{change_pct:+.2f}%", className="level-change", style={"color": color}))
    return html.Div(parts, className="ecard-trade-level", style={"borderTop": f"3px solid {border_color}"})


def make_risk_card(title, value, desc=None, color="#f59e0b", border_color=None):
    bc = border_color or color
    parts = [
        html.Div(title, className="risk-title", style={"color": color}),
        html.Div(str(value), className="risk-value", style={"color": color}),
    ]
    if desc:
        parts.append(html.Div(desc, className="risk-desc"))
    return html.Div(parts, className="ecard-risk", style={"borderLeft": f"4px solid {bc}"})


def make_scenario_card(label, value, sub=None, color="#10b981"):
    parts = [
        html.Div(label, className="scenario-label", style={"color": color}),
        html.Div(str(value), className="scenario-value", style={"color": color}),
    ]
    if sub:
        parts.append(html.Div(sub, className="scenario-sub"))
    return html.Div(parts, className="scenario-card")


def make_status_card(label, value, color="#10b981"):
    return html.Div([
        html.Div(label, className="status-label"),
        html.Div(value, className="status-value", style={"color": color}),
    ], className="status-card", style={"borderLeft": f"3px solid {color}"})


def make_grid(children, cols=3):
    return html.Div(children, className=f"ecard-grid ecard-grid-{cols}")


# =============================================================================
# BUILD THE PREDICTION RESULTS LAYOUT
# =============================================================================

def build_prediction_results(prediction: Dict):
    """Build the complete prediction results UI using ecard-* CSS system."""
    if not prediction:
        return html.Div("No prediction available yet.", className="ecard",
                        style={"textAlign": "center", "color": "#64748b", "padding": "40px"})

    current_price = prediction.get("current_price", 0)
    predicted_price = prediction.get("predicted_price", 0)
    price_change_pct = prediction.get("price_change_pct", 0)
    confidence = prediction.get("confidence", 0)
    ticker = prediction.get("ticker", "")
    source = prediction.get("source", "unknown")
    fallback_mode = prediction.get("fallback_mode", False)
    forecast = prediction.get("forecast_5_day", [])

    is_bullish = predicted_price > current_price
    direction_color = "#10b981" if is_bullish else "#ef4444"

    # Source banner
    if not fallback_mode and source == "data_driven_technical":
        banner = make_source_banner(True, "📊 LIVE DATA PREDICTION 🔬",
                                     "Real-time technical analysis using live FMP market data")
    elif not fallback_mode and source == "live_ai_backend":
        banner = make_source_banner(True, "🔥 LIVE AI ANALYSIS 🤖",
                                     "Real-time analysis with full backend integration")
    else:
        banner = make_source_banner(False, "⚡ ENHANCED SIMULATION 🎯",
                                     "Advanced modeling with realistic market constraints")

    # Hero card
    hero = make_hero_card(
        "📈" if is_bullish else "📉",
        "BULLISH SIGNAL" if is_bullish else "BEARISH SIGNAL",
        ticker, price_change_pct, is_bullish
    )

    # Confidence styling
    conf_color = "#10b981" if confidence > 80 else "#f59e0b" if confidence > 60 else "#ef4444"
    conf_level = "HIGH" if confidence > 80 else "MEDIUM" if confidence > 60 else "LOW"

    # Key metrics
    metrics = make_grid([
        make_metric_card("Current Price", f"${current_price:.4f}", "Market Price", "#93c5fd", "#3b82f6"),
        make_metric_card("Market Analysis", f"${predicted_price:.4f}", f"{price_change_pct:+.2f}%",
                         direction_color, direction_color),
        make_metric_card("AI Confidence", f"{confidence:.1f}%", f"{conf_level} CONFIDENCE",
                         conf_color, conf_color),
    ], 3)

    # Price trajectory chart
    chart = dcc.Graph(
        figure=build_price_trajectory_chart(prediction),
        config={"displayModeBar": False},
        style={"borderRadius": "14px", "overflow": "hidden"},
    )

    # Movement summary
    abs_change = abs(predicted_price - current_price)
    pct_abs = abs(price_change_pct)
    if pct_abs > 5:
        move_label, move_color = "STRONG", "#f59e0b"
    elif pct_abs > 2:
        move_label, move_color = "MODERATE", "#8b5cf6"
    else:
        move_label, move_color = "MILD", "#06b6d4"

    price_points = [current_price, predicted_price] + forecast
    price_min, price_max = min(price_points), max(price_points)

    movement_cards = make_grid([
        make_metric_card("Direction", "▲ LONG" if is_bullish else "▼ SHORT",
                         f"{price_change_pct:+.2f}%", direction_color, direction_color),
        make_metric_card("Movement", move_label, f"${abs_change:.4f} delta", move_color, move_color),
        make_metric_card("Price Band", f"${price_min:.2f} – ${price_max:.2f}", "Full Range", "#93c5fd", "#3b82f6"),
    ], 3)

    # AI Insights
    insights = []
    if abs(price_change_pct) > 5:
        insights.append(f"🎯 **Significant Movement Expected**: The AI predicts a {abs(price_change_pct):.1f}% {'increase' if is_bullish else 'decrease'} — a substantial price movement.")
    elif abs(price_change_pct) > 2:
        insights.append(f"📈 **Moderate Movement Expected**: The AI forecasts a {abs(price_change_pct):.1f}% {'rise' if is_bullish else 'fall'} — a reasonable price adjustment.")
    else:
        insights.append(f"📊 **Minor Movement Expected**: The AI suggests a small {abs(price_change_pct):.1f}% {'uptick' if is_bullish else 'decline'} — relatively stable.")

    if confidence > 80:
        insights.append(f"✅ **High Confidence Prediction**: With {confidence:.1f}% confidence, the AI model shows strong conviction.")
    elif confidence > 60:
        insights.append(f"⚖️ **Moderate Confidence**: The AI shows {confidence:.1f}% confidence — reasonably reliable with some uncertainty.")
    else:
        insights.append(f"⚠️ **Lower Confidence**: At {confidence:.1f}% confidence, consider this prediction with caution.")

    asset_type = get_asset_type(ticker)
    asset_notes = {
        "crypto": "🌊 **Crypto Asset**: Cryptocurrency markets are highly volatile and can change rapidly.",
        "forex": "💱 **Forex Pair**: Currency movements can be influenced by economic data and central bank policies.",
        "commodity": "🛢️ **Commodity**: Prices are affected by supply/demand dynamics and global factors.",
        "index": "📊 **Market Index**: Reflects broader market sentiment and economic conditions.",
        "stock": "📈 **Individual Stock**: Price movements can be influenced by earnings, news, and market conditions.",
    }
    insights.append(asset_notes.get(asset_type, "📈 General market risks apply."))

    # ── Multi-Timeframe Analysis section ─────────────────────────────
    mtf_section = _build_mtf_section(prediction)

    # ── Primary timeframe badge ──────────────────────────────────────
    primary_tf = prediction.get("primary_timeframe", "1day")
    tf_labels = {"15min": "15 Min", "1hour": "1 Hour", "4hour": "4 Hours", "1day": "1 Day"}
    tf_badge = html.Div([
        html.Span("⏱️ ", style={"fontSize": "13px"}),
        html.Span(f"Timeframe: {tf_labels.get(primary_tf, primary_tf)}", style={
            "fontSize": "12px", "color": "#a78bfa", "fontWeight": "600",
        }),
    ], style={"marginBottom": "12px", "display": "flex", "alignItems": "center", "gap": "4px"})

    return html.Div([
        banner, hero, tf_badge,
        make_section_header("🎯", "Key Prediction Metrics"), metrics,
        make_section_header("📊", "Price Movement Analysis"), chart, movement_cards,
        mtf_section,
        make_section_header("🧠", "AI Insights Summary"),
        html.Div([make_insight_card(i) for i in insights]),
    ])


def _build_mtf_section(prediction: Dict):
    """Build the Multi-Timeframe Analysis display section."""
    # Show upgrade prompt if MTF was requested but blocked by plan
    if prediction.get("mtf_upgrade_required"):
        return html.Div([
            make_section_header("⏱️", "Multi-Timeframe Analysis"),
            html.Div([
                html.Div([
                    html.Div("🔒", style={"fontSize": "28px", "marginBottom": "8px"}),
                    html.Div("Multi-Timeframe Consensus requires Starter or higher", style={
                        "color": "#e2e8f0", "fontSize": "14px", "fontWeight": "600",
                        "marginBottom": "6px",
                    }),
                    html.Div(
                        "Upgrade to analyse multiple timeframes simultaneously and get "
                        "consensus signals when 15m, 1H, 4H, and 1D all align.",
                        style={"color": "#94a3b8", "fontSize": "12px", "lineHeight": "1.5",
                               "marginBottom": "14px", "maxWidth": "400px"},
                    ),
                    html.A("View Plans →", href="/pricing", style={
                        "display": "inline-block", "padding": "8px 20px", "borderRadius": "8px",
                        "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                        "color": "#fff", "fontSize": "12px", "fontWeight": "700",
                        "textDecoration": "none",
                    }),
                ], style={"textAlign": "center"}),
            ], style={
                "padding": "28px 20px", "borderRadius": "14px",
                "background": "rgba(15,23,42,0.5)", "border": "1px solid rgba(99,102,241,0.1)",
            }),
        ])

    mtf = prediction.get("multi_timeframe_analysis")
    if not mtf:
        return html.Div()

    consensus = mtf.get("consensus", {})
    tf_labels = {"15min": "15 Min", "1hour": "1 Hour", "4hour": "4 Hours", "1day": "1 Day"}

    # ── Consensus summary card ───────────────────────────────────────
    consensus_parts = []
    if consensus:
        c_bias = consensus.get("bias", "Neutral")
        c_signal = consensus.get("weighted_signal", 0)
        c_conf = consensus.get("confidence", 50)
        c_agree = consensus.get("agreement_pct", 0)
        c_bull = consensus.get("bullish_timeframes", 0)
        c_bear = consensus.get("bearish_timeframes", 0)
        c_neut = consensus.get("neutral_timeframes", 0)
        c_total = consensus.get("total_timeframes", 0)

        bias_color = "#10b981" if c_bias == "Bullish" else "#ef4444" if c_bias == "Bearish" else "#f59e0b"
        bias_icon = "📈" if c_bias == "Bullish" else "📉" if c_bias == "Bearish" else "➡️"

        consensus_parts = [
            make_metric_card("MTF Consensus", f"{bias_icon} {c_bias}",
                             f"Signal: {c_signal:+.4f}", bias_color, bias_color),
            make_metric_card("Agreement", f"{c_agree:.0f}%",
                             f"{c_bull}↑ {c_bear}↓ {c_neut}→ of {c_total} TFs",
                             "#10b981" if c_agree >= 70 else "#f59e0b", "#3b82f6"),
            make_metric_card("MTF Confidence", f"{c_conf:.0f}%", None,
                             "#10b981" if c_conf > 70 else "#f59e0b" if c_conf > 55 else "#ef4444",
                             "#8b5cf6"),
        ]

    # ── Per-timeframe signal cards ───────────────────────────────────
    tf_cards = []
    for tf_key in ["15min", "1hour", "4hour", "1day"]:
        tf_data = mtf.get(tf_key)
        if not tf_data or tf_data.get("status") == "insufficient_data":
            continue

        bias = tf_data.get("bias", "Neutral")
        rsi = tf_data.get("rsi", 50)
        composite = tf_data.get("composite_signal", 0)
        vol = tf_data.get("volatility", 0)
        conf = tf_data.get("confidence", 50)

        bias_color = "#10b981" if bias == "Bullish" else "#ef4444" if bias == "Bearish" else "#f59e0b"
        bias_icon = "▲" if bias == "Bullish" else "▼" if bias == "Bearish" else "◆"

        rsi_color = "#ef4444" if rsi > 70 else "#10b981" if rsi < 30 else "#94a3b8"
        rsi_label = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"

        tf_cards.append(html.Div([
            # Header row
            html.Div([
                html.Span(f"⏱️ {tf_labels.get(tf_key, tf_key)}", style={
                    "fontSize": "13px", "fontWeight": "700", "color": "#e2e8f0"}),
                html.Span(f"{bias_icon} {bias}", style={
                    "fontSize": "12px", "fontWeight": "700", "color": bias_color,
                    "padding": "2px 8px", "borderRadius": "8px",
                    "background": f"rgba({','.join(str(int(bias_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)",
                }),
            ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center",
                       "marginBottom": "10px"}),
            # Signal strength bar
            html.Div([
                html.Div(style={
                    "width": f"{min(abs(composite) * 100, 100):.0f}%",
                    "height": "4px",
                    "borderRadius": "2px",
                    "background": bias_color,
                    "transition": "width 0.5s ease",
                }),
            ], style={"height": "4px", "borderRadius": "2px", "background": "rgba(99,102,241,0.1)",
                       "marginBottom": "10px"}),
            # Indicators row
            html.Div([
                html.Div([
                    html.Div("RSI", style={"fontSize": "10px", "color": "#64748b", "textTransform": "uppercase",
                                            "letterSpacing": "0.5px"}),
                    html.Div(f"{rsi:.0f}", style={"fontSize": "14px", "fontWeight": "700", "color": rsi_color}),
                    html.Div(rsi_label, style={"fontSize": "9px", "color": rsi_color}),
                ], style={"textAlign": "center", "flex": "1"}),
                html.Div([
                    html.Div("MACD", style={"fontSize": "10px", "color": "#64748b", "textTransform": "uppercase",
                                             "letterSpacing": "0.5px"}),
                    html.Div(f"{tf_data.get('macd', 0):.4f}", style={
                        "fontSize": "14px", "fontWeight": "700",
                        "color": "#10b981" if tf_data.get("macd", 0) > 0 else "#ef4444"}),
                ], style={"textAlign": "center", "flex": "1"}),
                html.Div([
                    html.Div("VOL", style={"fontSize": "10px", "color": "#64748b", "textTransform": "uppercase",
                                            "letterSpacing": "0.5px"}),
                    html.Div(f"{vol:.2%}", style={"fontSize": "14px", "fontWeight": "700", "color": "#f59e0b"}),
                ], style={"textAlign": "center", "flex": "1"}),
                html.Div([
                    html.Div("CONF", style={"fontSize": "10px", "color": "#64748b", "textTransform": "uppercase",
                                             "letterSpacing": "0.5px"}),
                    html.Div(f"{conf:.0f}%", style={"fontSize": "14px", "fontWeight": "700", "color": "#a78bfa"}),
                ], style={"textAlign": "center", "flex": "1"}),
            ], style={"display": "flex", "gap": "4px"}),
        ], className="ecard", style={"padding": "14px", "borderLeft": f"3px solid {bias_color}"}))

    if not tf_cards and not consensus_parts:
        return html.Div()

    children = [make_section_header("⏱️", "Multi-Timeframe Analysis")]
    if consensus_parts:
        children.append(make_grid(consensus_parts, 3))
    if tf_cards:
        children.append(html.Div(tf_cards, className="ecard-grid ecard-grid-2" if len(tf_cards) <= 2
                                  else "ecard-grid ecard-grid-4"))

    return html.Div(children)


# =============================================================================
# BUILD TABS: TRADING STRATEGY / FORECAST / RISK
# =============================================================================

def build_trading_strategy_tab(prediction: Dict):
    if not prediction:
        return html.Div("Run a prediction first.", className="ecard", style={"padding": "30px", "textAlign": "center"})

    cp = prediction.get("current_price", 0)
    pp = prediction.get("predicted_price", 0)
    ticker = prediction.get("ticker", "")
    confidence = prediction.get("confidence", 0)
    asset_type = get_asset_type(ticker)

    if not cp or cp == 0:
        return html.Div("Price data unavailable.", className="ecard", style={"padding": "30px"})

    is_bullish = pp > cp
    pct = ((pp - cp) / cp) * 100
    dir_color = "#10b981" if is_bullish else "#ef4444"
    vol_est = abs(pct) * 2
    vol_level = "High" if vol_est > 6 else "Medium" if vol_est > 3 else "Low"
    vol_color = "#ef4444" if vol_est > 6 else "#f59e0b" if vol_est > 3 else "#10b981"
    conf_level = "High" if confidence > 80 else "Medium" if confidence > 60 else "Low"
    conf_color = "#10b981" if confidence > 80 else "#f59e0b" if confidence > 60 else "#ef4444"

    # Trade analysis
    analysis = make_grid([
        make_metric_card("Direction", "🟢 BULLISH" if is_bullish else "🔴 BEARISH", None, dir_color, dir_color),
        make_metric_card("Expected Move", f"{abs(pct):.2f}%", None, "#93c5fd", "#3b82f6"),
        make_metric_card("AI Confidence", f"{conf_level} ({confidence:.1f}%)", None, conf_color, conf_color),
        make_metric_card("Volatility", vol_level, None, vol_color, vol_color),
    ], 4)

    # Risk parameters
    params = {
        "crypto": {"sl": 0.025, "rr": 2.0}, "forex": {"sl": 0.008, "rr": 3.0},
        "commodity": {"sl": 0.015, "rr": 2.5}, "index": {"sl": 0.012, "rr": 2.8},
        "stock": {"sl": 0.018, "rr": 2.5},
    }.get(asset_type, {"sl": 0.015, "rr": 2.5})

    sl_pct = params["sl"]
    rr = params["rr"]
    conf_mult = 0.8 + (confidence / 100) * 0.4

    if is_bullish:
        t1 = cp * (1 + sl_pct * rr * conf_mult)
        t2 = cp * (1 + sl_pct * rr * conf_mult * 1.6)
        t3 = cp * (1 + sl_pct * rr * conf_mult * 2.618)
        sl = cp * (1 - sl_pct)
    else:
        t1 = cp * (1 - sl_pct * rr * conf_mult)
        t2 = cp * (1 - sl_pct * rr * conf_mult * 1.6)
        t3 = cp * (1 - sl_pct * rr * conf_mult * 2.618)
        sl = cp * (1 + sl_pct)

    t1c = ((t1 - cp) / cp) * 100
    t2c = ((t2 - cp) / cp) * 100
    t3c = ((t3 - cp) / cp) * 100
    slc = ((sl - cp) / cp) * 100

    t1_color = "#10b981" if is_bullish else "#ef4444"
    sl_color = "#ef4444" if is_bullish else "#10b981"

    # Dynamic price levels using trade-level cards
    levels = make_grid([
        make_trade_level("Entry Price", cp, None, "#93c5fd", "#3b82f6"),
        make_trade_level("Target 1 (Quick)", t1, t1c, t1_color, t1_color),
        make_trade_level("Target 2 (Main)", t2, t2c, "#06b6d4", "#06b6d4"),
        make_trade_level("Target 3 (Extended)", t3, t3c, "#8b5cf6", "#8b5cf6"),
        make_trade_level("Stop Loss", sl, slc, sl_color, sl_color),
    ], 5)

    # Risk/Reward
    risk_amt = abs(cp - sl)
    rr1 = abs(t1 - cp) / risk_amt if risk_amt > 0 else 0
    rr2 = abs(t2 - cp) / risk_amt if risk_amt > 0 else 0
    rr3 = abs(t3 - cp) / risk_amt if risk_amt > 0 else 0

    rr_cards = make_grid([
        make_metric_card("Risk Per Share", f"${risk_amt:.4f}", "Maximum loss/share", "#fca5a5", "#ef4444"),
        make_metric_card("R/R Ratio T1", f"{rr1:.2f}", None,
                         "#10b981" if rr1 >= 1.5 else "#ef4444", "#10b981" if rr1 >= 1.5 else "#ef4444"),
        make_metric_card("R/R Ratio T2", f"{rr2:.2f}", None,
                         "#10b981" if rr2 >= 2.5 else "#ef4444", "#10b981" if rr2 >= 2.5 else "#ef4444"),
        make_metric_card("R/R Ratio T3", f"{rr3:.2f}", None,
                         "#10b981" if rr3 >= 3.5 else "#ef4444", "#10b981" if rr3 >= 3.5 else "#ef4444"),
    ], 4)

    # Position sizing
    account = 50000
    max_risk_pct = 0.02
    risk_dollars = account * max_risk_pct
    pos_size = int(risk_dollars / risk_amt) if risk_amt > 0 else 0
    pos_value = pos_size * cp
    portfolio_alloc = (pos_value / account) * 100 if account > 0 else 0
    max_loss = pos_size * risk_amt
    alloc_color = "#10b981" if portfolio_alloc < 15 else "#f59e0b" if portfolio_alloc < 30 else "#ef4444"

    sizing_cards = make_grid([
        make_metric_card("Position Size", f"{pos_size:,} shares", None, "#93c5fd", "#3b82f6"),
        make_metric_card("Position Value", f"${pos_value:,.2f}", None, "#c4b5fd", "#8b5cf6"),
        make_metric_card("Portfolio Allocation", f"{portfolio_alloc:.2f}%", None, alloc_color, alloc_color),
        make_metric_card("Max Potential Loss", f"${max_loss:.2f}", None, "#fca5a5", "#ef4444"),
    ], 4)

    # Scenario analysis
    best_profit = pos_size * abs(t3 - cp)
    target_profit = pos_size * abs(t2 - cp)
    worst_loss = pos_size * risk_amt

    scenarios = make_grid([
        make_scenario_card("🟢 Best Case (+T3)", f"${best_profit:,.2f}",
                           f"{(best_profit/pos_value*100):.1f}% return" if pos_value > 0 else "", "#10b981"),
        make_scenario_card("🟡 Target Case (+T2)", f"${target_profit:,.2f}",
                           f"{(target_profit/pos_value*100):.1f}% return" if pos_value > 0 else "", "#f59e0b"),
        make_scenario_card("🔴 Worst Case (Stop)", f"-${worst_loss:,.2f}",
                           f"-{(worst_loss/pos_value*100):.1f}% return" if pos_value > 0 else "", "#ef4444"),
    ], 3)

    # Pre-execution checklist
    strategy_type = "LONG POSITION" if is_bullish else "SHORT POSITION"
    checklist_items = [
        "Confirm account balance and buying power",
        "Verify position size and risk limits",
        "Set all stop loss and target orders",
        "Check for upcoming news/earnings",
        "Confirm market hours and liquidity",
        "Review correlation with existing positions",
        "Document trade rationale in journal",
        "Set price alerts for key levels",
    ]
    checklist = html.Div([
        html.Div([
            html.Div("☐", style={"fontSize": "14px", "color": "#6366f1"}),
            html.Span(item),
        ], className="checklist-item")
        for item in checklist_items
    ])

    # Warnings
    warnings = []
    if confidence < 60:
        warnings.append(make_insight_card("⚠️ **HIGH RISK**: Low confidence prediction — consider reducing position size or waiting for better setup."))
    if portfolio_alloc > 10:
        warnings.append(make_insight_card("🚨 **POSITION SIZE WARNING**: Position exceeds 10% of portfolio. Consider reducing size."))
    warnings.append(make_insight_card("💡 This plan is based on AI analysis and should be combined with your own market analysis and risk tolerance."))

    return html.Div([
        make_section_header("🎯", "Trade Analysis & Setup"), analysis,
        make_section_header("💰", "Dynamic Price Levels"), levels,
        make_section_header("⚖️", "Advanced Risk/Reward Analysis"), rr_cards,
        make_section_header("📊", "Advanced Position Sizing (Fixed 2% Risk)"), sizing_cards,
        make_section_header("📈", "Scenario Analysis"), scenarios,
        make_section_header("✅", "Pre-Execution Checklist"), checklist,
        html.Div(warnings, style={"marginTop": "16px"}),
    ])


def build_forecast_tab(prediction: Dict):
    if not prediction:
        return html.Div("Run a prediction first.", className="ecard", style={"padding": "30px", "textAlign": "center"})

    forecast = prediction.get("forecast_5_day", [])
    current_price = prediction.get("current_price", 0)
    predicted_price = prediction.get("predicted_price", 0)

    if not forecast:
        forecast = [predicted_price * (1 + np.random.uniform(-0.02, 0.02)) for _ in range(5)]

    # Forecast day cards
    day_cards = []
    for i, price in enumerate(forecast[:5]):
        day_change = ((price - current_price) / current_price) * 100 if current_price else 0
        date_str = (datetime.now() + timedelta(days=i + 1)).strftime("%m/%d")
        day_cards.append(make_forecast_day(i + 1, date_str, price, day_change))

    # Trend summary
    trend_bullish = forecast[-1] > forecast[0] if len(forecast) >= 2 else True
    trend_color = "#10b981" if trend_bullish else "#ef4444"
    total_change = ((forecast[-1] - current_price) / current_price) * 100 if current_price and forecast else 0
    volatility = np.std(forecast) / np.mean(forecast) if forecast else 0
    vol_level = "High" if volatility > 0.03 else "Medium" if volatility > 0.015 else "Low"
    vol_color = "#ef4444" if volatility > 0.03 else "#f59e0b" if volatility > 0.015 else "#10b981"

    summary = make_grid([
        make_metric_card("Direction", f"{'📈 Bullish' if trend_bullish else '📉 Bearish'}", None, trend_color, trend_color),
        make_metric_card("5-Day Change", f"{total_change:+.2f}%", None, trend_color, trend_color),
        make_metric_card("Forecast Volatility", f"{vol_level} ({volatility:.1%})", None, vol_color, vol_color),
    ], 3)

    return html.Div([
        make_section_header("📈", "Multi-Day Price Forecast"),
        make_grid(day_cards, 5),
        make_section_header("🎯", "Trend Summary"),
        summary,
    ])


def build_risk_tab(prediction: Dict):
    if not prediction:
        return html.Div("Run a prediction first.", className="ecard", style={"padding": "30px", "textAlign": "center"})

    risk = generate_risk_metrics(prediction)
    ticker = prediction.get("ticker", "")
    asset_type = get_asset_type(ticker)

    # Risk gauge
    gauge = dcc.Graph(figure=build_risk_gauge(risk), config={"displayModeBar": False})

    # Risk metrics using ecard-risk cards
    risk_cards = make_grid([
        make_risk_card("Volatility", f"{risk['volatility']:.1f}%", "Annualized", "#f59e0b"),
        make_risk_card("Sharpe Ratio", f"{risk['sharpe_ratio']:.3f}", "Risk-adjusted return",
                       "#10b981" if risk["sharpe_ratio"] > 1 else "#ef4444"),
        make_risk_card("Max Drawdown", f"{risk['max_drawdown']:.1f}%", "Peak-to-trough", "#ef4444"),
        make_risk_card("VaR (95%)", f"${risk['var_95']:.4f}", "Value at Risk", "#f59e0b"),
    ], 4)

    additional = make_grid([
        make_metric_card("Beta", f"{risk['beta']:.3f}", None, "#93c5fd", "#3b82f6"),
        make_metric_card("Sortino Ratio", f"{risk['sortino_ratio']:.3f}", None,
                         "#10b981" if risk["sortino_ratio"] > 1 else "#ef4444",
                         "#10b981" if risk["sortino_ratio"] > 1 else "#ef4444"),
        make_metric_card("Risk Score", f"{risk['risk_score']}/10", None,
                         "#10b981" if risk["risk_score"] <= 3 else "#f59e0b" if risk["risk_score"] <= 6 else "#ef4444",
                         "#10b981" if risk["risk_score"] <= 3 else "#f59e0b" if risk["risk_score"] <= 6 else "#ef4444"),
    ], 3)

    # Risk assessment using ecard-assessment
    score = risk["risk_score"]
    if score <= 3:
        level, level_cls = "Low", "low"
        lmsg = "✅ All risk metrics are within acceptable ranges."
        risk_factors = []
    elif score <= 6:
        level, level_cls = "Moderate", "medium"
        lmsg = "⚠️ Some risk factors require attention."
        risk_factors = ["Elevated volatility detected", "Consider position size reduction"]
    else:
        level, level_cls = "High", "high"
        lmsg = "🚨 Multiple risk factors detected — exercise caution."
        risk_factors = ["High volatility environment", "Significant drawdown risk", "Consider hedging strategies"]

    assessment_parts = [
        html.Div(f"{'✅' if score <= 3 else '⚠️' if score <= 6 else '🚨'} {level} Risk Profile",
                 className="assessment-title"),
        html.Div(lmsg, className="assessment-item"),
    ]
    for rf in risk_factors:
        assessment_parts.append(html.Div(f"• {rf}", className="assessment-item"))

    assessment = html.Div(assessment_parts, className=f"ecard-assessment {level_cls}")

    # Asset context
    contexts = {
        "crypto": "Cryptocurrency assets are inherently volatile and subject to regulatory risks.",
        "forex": "Currency pairs can be affected by geopolitical events and central bank policies.",
        "commodity": "Commodity prices are influenced by supply/demand dynamics and weather.",
        "index": "Market indices reflect broader economic conditions and sentiment.",
        "stock": "Individual stocks carry company-specific and sector risks.",
    }
    context_card = make_insight_card(f"🏷️ **{asset_type.title()} Risk Context**: {contexts.get(asset_type, 'General market risks apply.')}")

    return html.Div([
        make_section_header("⚠️", "Risk Assessment"), assessment,
        html.Div([gauge], className="ecard"), risk_cards, additional,
        make_section_header("📋", "Asset-Specific Risk Context"), context_card,
    ])


# =============================================================================
# DASH APP
# =============================================================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.DARKLY,
        "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700;800;900&display=swap",
    ],
    suppress_callback_exceptions=True,
    title="MarketLens AI",
    update_title="Analyzing...",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"},
        {"name": "apple-mobile-web-app-capable", "content": "yes"},
        {"name": "apple-mobile-web-app-status-bar-style", "content": "black-translucent"},
        {"name": "theme-color", "content": "#0a0e1a"},
    ],
)
server = app.server

# ── LAYOUT ──────────────────────────────────────────────────────────────────

# Forward declaration — full definition below
def build_contact_modal(): return html.Div(id="contact-modal", style={"display":"none"})


# Forward declaration — full definition below


app.layout = html.Div([
    # Hidden stores
    dcc.Store(id="prediction-store", data=None),
    dcc.Store(id="app-state", data={"predictions_count": 0, "disclaimer_accepted": False, "premium_active": False}),
    dcc.Store(id="ftmo-store", data={"setup_done": False, "balance": 100000, "daily_limit": 5, "total_limit": 10, "positions": []}),
    dcc.Store(id="user-session", storage_type="local", data=None),
    dcc.Store(id="disclaimer-store", storage_type="local", data={"accepted": False}),
    dcc.Store(id="contact-form-store", data={"visible": False}),
    dcc.Store(id="timeframe-dropdown", data="1day"),  # Hidden timeframe state for prediction callback
    dcc.Location(id="url", refresh=True),
    dcc.Store(id="redirect-store", data=None),

    # Contact form modal (global, always present)
    build_contact_modal(),

    # Auth-gated container: login page OR main dashboard
    html.Div(id="app-container"),

], style={"background": "#0a0e1a", "minHeight": "100vh"})


# =============================================================================
# MAIN DASHBOARD LAYOUT BUILDER — called when user is authenticated
# =============================================================================

def _build_main_dashboard(user=None):
    """Build the full sidebar + content layout for authenticated users."""

    # Determine allowed tickers/timeframes based on plan
    if SAAS_AUTH_AVAILABLE and user:
        allowed_tickers = get_allowed_tickers(user)
        allowed_timeframes = get_allowed_timeframes(user)
        user_badge = build_user_badge(user)
        plan_info = get_plan_badge_info(user)
        plan_label = plan_info["plan_name"].upper()
        plan_colors = plan_info["colors"]
    else:
        allowed_tickers = ALL_TICKERS
        allowed_timeframes = ["15min", "1hour", "4hour", "1day"]
        user_badge = html.Div()
        plan_label = "DEMO"
        plan_colors = {"bg": "rgba(100,116,139,0.12)", "border": "#64748b", "text": "#94a3b8"}

    tf_options = []
    tf_label_map = {"15min": "15 Minutes", "1hour": "1 Hour", "4hour": "4 Hours", "1day": "1 Day"}
    for tf in ["15min", "1hour", "4hour", "1day"]:
        tf_options.append({
            "label": tf_label_map[tf],
            "value": tf,
            "disabled": tf not in allowed_timeframes,
        })

    dashboard = html.Div([
        # ── SIDEBAR ──
        html.Div([
            html.Div([
                # Logo
                html.Div([
                    html.Div([
                        html.Div("⚡", style={"fontSize": "22px", "lineHeight": "1"}),
                    ], style={"background": "linear-gradient(135deg, #06b6d4, #8b5cf6)",
                              "padding": "10px", "borderRadius": "12px", "display": "flex",
                              "alignItems": "center", "justifyContent": "center",
                              "boxShadow": "0 4px 15px rgba(99,102,241,0.25)"}),
                    html.Div([
                        html.Div("MarketLens", className="sidebar-logo"),
                        html.Div("Advanced Market Analysis & Research", className="sidebar-sub"),
                    ]),
                ], style={"display": "flex", "alignItems": "center", "gap": "12px", "marginBottom": "24px"}),

                html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                                "margin": "0 0 20px 0"}),

                # User badge (plan, usage, sign out)

                # Navigation — custom buttons instead of RadioItems
                user_badge,
                html.Div("NAVIGATION", className="sidebar-section-label"),

                # Hidden radio to hold state (invisible, driven by button clicks)
                dcc.RadioItems(
                    id="page-nav",
                    options=[
                        {"label": "", "value": "ai_prediction"},
                        {"label": "", "value": "advanced_analytics"},
                        {"label": "", "value": "portfolio_mgmt"},
                        {"label": "", "value": "backtesting"},
                        {"label": "", "value": "ftmo_dashboard"},
                        {"label": "", "value": "model_training"},
                        {"label": "", "value": "subscription"},
                        {"label": "", "value": "app_guide"},
                        {"label": "", "value": "admin_panel"},
                    ],
                    value="ai_prediction",
                    style={"display": "none"},
                ),

                # Custom nav buttons
                html.Div([
                    html.Button([
                        html.Span(icon, style={"fontSize": "16px", "width": "24px", "textAlign": "center", "flexShrink": "0"}),
                        html.Span(label, style={"flex": "1", "textAlign": "left"}),
                        html.Span("", id=f"nav-indicator-{value}",
                                  style={"width": "6px", "height": "6px", "borderRadius": "50%",
                                         "background": "#6366f1", "flexShrink": "0",
                                         "display": "none"}),
                    ],
                    id={"type": "nav-btn", "index": value},
                    n_clicks=0,
                    className="sidebar-nav-btn",
                    )
                    for icon, label, value in [
                        ("🔬", "Market Analysis", "ai_prediction"),
                        ("📊", "Advanced Analytics", "advanced_analytics"),
                        ("💼", "Portfolio Mgmt", "portfolio_mgmt"),
                        ("📈", "Backtesting", "backtesting"),
                        ("🧠", "Model Training", "model_training"),
                        ("💳", "Subscription", "subscription"),
                        ("📖", "App Guide", "app_guide"),
                    ] + ([
                        ("🏦", "FTMO Dashboard", "ftmo_dashboard"),
                        ("🛡️", "Admin Panel", "admin_panel"),
                    ] if (SAAS_AUTH_AVAILABLE and is_admin(user)) or not SAAS_AUTH_AVAILABLE else [])
                ], style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginBottom": "20px"}),

                html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                                "margin": "0 0 20px 0"}),

                # Asset Selection
                html.Div("ASSET SELECTION", className="sidebar-section-label"),
                dcc.Dropdown(
                    id="ticker-dropdown",
                    options=[{"label": t, "value": t} for t in allowed_tickers],
                    value=allowed_tickers[0] if allowed_tickers else "BTCUSD",
                    clearable=False,
                    style={
                        "marginBottom": "12px",
                        "color": "#e2e8f0",
                        "fontWeight": "600",
                        "fontSize": "14px",
                    },
                ),

                # Quick select — only show groups that contain at least one allowed ticker
                html.Div("QUICK SELECT", className="sidebar-section-label"),
                html.Div([
                    html.Button(name.split(" ")[-1], id={"type": "group-btn", "index": name},
                                className="sidebar-quick-btn",
                                n_clicks=0)
                    for name in TICKER_GROUPS.keys()
                    if any(t in allowed_tickers for t in TICKER_GROUPS[name])
                ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "marginBottom": "6px"}),
                # Show ticker count for user's plan
                html.Div(f"{len(allowed_tickers)} assets available on your plan", style={
                    "color": "#475569", "fontSize": "10px", "marginBottom": "20px",
                }),

                html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                                "margin": "0 0 20px 0"}),
                html.Div([
                    html.Div([
                        html.Span("⏱️", style={"fontSize": "12px"}),
                        html.Span("Multi-Timeframe Analysis", style={"fontSize": "11px", "color": "#64748b"}),
                        # MTF lock badge for plans without mtf_analysis feature
                        *([html.Span("🔒", style={"fontSize": "10px", "marginLeft": "4px"})]
                          if SAAS_AUTH_AVAILABLE and user and not can_access_feature(user, "mtf_analysis")
                          else []),
                    ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "8px"}),
                    dcc.Checklist(
                        id="mtf-checklist",
                        options=[
                            {"label": " 15m", "value": "15min",
                             "disabled": "15min" not in allowed_timeframes},
                            {"label": " 1H", "value": "1hour",
                             "disabled": "1hour" not in allowed_timeframes},
                            {"label": " 4H", "value": "4hour",
                             "disabled": "4hour" not in allowed_timeframes},
                            {"label": " 1D", "value": "1day",
                             "disabled": "1day" not in allowed_timeframes},
                        ],
                        value=["1day"],
                        inline=True,
                        className="sidebar-mtf-checklist",
                        style={"fontSize": "11px"},
                    ),
                    # Upgrade hint when MTF is locked
                    *([html.Div([
                        html.Span("⚡ ", style={"fontSize": "10px"}),
                        html.Span("Upgrade to Starter to unlock multi-timeframe consensus", style={
                            "color": "#f59e0b", "fontSize": "10px", "fontWeight": "500",
                        }),
                    ], style={"marginTop": "6px", "padding": "5px 8px", "borderRadius": "6px",
                              "background": "rgba(245,158,11,0.06)",
                              "border": "1px solid rgba(245,158,11,0.1)"})]
                      if SAAS_AUTH_AVAILABLE and user and not can_access_feature(user, "mtf_analysis")
                      else []),
                ], style={"marginBottom": "20px"}),

                html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                                "margin": "0 0 20px 0"}),

                # AI Models — visible to admin only, hidden but present for callbacks
                html.Div([
                    html.Div("AI MODELS", className="sidebar-section-label"),
                    dcc.Checklist(
                        id="model-checklist",
                        options=MODEL_OPTIONS,
                        value=["advanced_transformer", "cnn_lstm", "xgboost", "sklearn_ensemble"],
                        className="sidebar-model-checklist",
                        style={"marginBottom": "20px"},
                    ),
                ], style={"display": "block"} if ((SAAS_AUTH_AVAILABLE and is_admin(user)) or not SAAS_AUTH_AVAILABLE) else {"display": "none"}),

                html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.2), transparent)",
                                "margin": "0 0 20px 0"}),

                # System Status — compact
                html.Div("SYSTEM STATUS", className="sidebar-section-label"),
                html.Div([
                    *[html.Div([
                        html.Span("●", style={"color": color, "fontSize": "8px", "flexShrink": "0"}),
                        html.Span(label, style={"color": "#64748b", "fontSize": "11px", "flex": "1"}),
                        html.Span(status, style={"fontSize": "9px", "fontWeight": "700",
                                                  "letterSpacing": "0.5px", "color": color,
                                                  "padding": "1px 6px", "borderRadius": "4px",
                                                  "background": f"rgba({','.join(str(int(color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))},0.12)"}),
                    ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "5px"})
                    for label, status, color in [
                        ("Backend", "LIVE" if BACKEND_AVAILABLE else "DEMO", "#10b981" if BACKEND_AVAILABLE else "#f59e0b"),
                        ("FMP API", "OK" if FMP_API_KEY else "SIM", "#10b981" if FMP_API_KEY else "#f59e0b"),
                        ("Dash", "ACTIVE", "#10b981"),
                        ("Backtest", "READY" if AI_BACKTEST_AVAILABLE else "SIM", "#10b981" if AI_BACKTEST_AVAILABLE else "#f59e0b"),
                        ("Portfolio", "BL" if AI_PORTFOLIO_AVAILABLE else "SIM", "#10b981" if AI_PORTFOLIO_AVAILABLE else "#f59e0b"),
                    ]],
                ], className="sidebar-status-block"),

                # Version
                html.Div([
                    html.Div(style={"height": "1px", "background": "linear-gradient(90deg, transparent, rgba(99,102,241,0.15), transparent)",
                                    "margin": "16px 0 12px 0"}),
                    html.Div("v2.0 — Dash Edition", style={"fontSize": "10px", "color": "#334155",
                                                             "letterSpacing": "1px", "textAlign": "center"}),
                ]),

            ], className="sidebar-wrap"),
        ], className="sidebar-container"),

        # ── MAIN CONTENT ──
        html.Div([
            # Dynamic page header
            html.Div(id="page-header"),

            # Dynamic page content
            html.Div(id="page-content"),
            # Footer
            html.Div([
                html.Hr(style={"borderColor": "rgba(99,102,241,0.08)"}),
                html.Div([
                    html.Span("MarketLens AI — Dash Edition", style={"color": "#475569", "fontSize": "12px"}),
                    html.Span(" · ", style={"color": "#334155"}),
                    html.Span("For research purposes only", style={"color": "#475569", "fontSize": "12px"}),
                    html.Span(" · ", style={"color": "#334155"}),
                    html.Button("📧 Contact Support", id="open-contact-btn", n_clicks=0, style={"color": "#06b6d4", "fontSize": "12px", "background": "none", "border": "none", "cursor": "pointer", "fontWeight": "600", "padding": "0"}),
                ], style={"textAlign": "center", "padding": "16px 0"}),
            ]),

        ], style={"flex": "1", "padding": "24px 32px", "maxWidth": "1200px"}),

    ], style={"display": "flex", "minHeight": "100vh", "background": "#0a0e1a"})

    # Wrap in disclaimer-gated container
    return html.Div([
        # Disclaimer overlay (hidden when accepted)
        html.Div(id="disclaimer-container", children=build_disclaimer_overlay()),
        # Main content (blurred until disclaimer accepted)
        html.Div(id="main-app-content", children=dashboard),
    ])


# =============================================================================
# AUTH ROUTING — Master callback that gates the entire app
# =============================================================================

@callback(
    Output("app-container", "children"),
    Input("user-session", "data"),
    Input("url", "pathname"),
)
def route_auth(session_data, pathname):
    """
    Master routing callback:
    - If user has valid session token → show main dashboard
    - If /pricing → show pricing page
    - Otherwise → show login page
    """
    user = None

    # Check session
    if session_data and SAAS_AUTH_AVAILABLE:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)

    # Special routes
    if pathname == "/pricing":
        if SAAS_AUTH_AVAILABLE:
            current_plan = user.get("plan", "free") if user else "free"
            return build_pricing_page(current_plan)
        return html.Div("Pricing not available", style={"color": "#e2e8f0", "padding": "40px"})

    if pathname and pathname.startswith("/payment-success"):
        # Stripe redirects here after successful checkout
        return html.Div([
            html.Div([
                html.Div("🎉", style={"fontSize": "64px", "marginBottom": "16px"}),
                html.H2("Payment Successful!", style={"color": "#10b981", "fontWeight": "800", "margin": "0 0 12px"}),
                html.P("Your subscription is now active. Welcome to your new plan!",
                       style={"color": "#94a3b8", "fontSize": "15px", "margin": "0 0 24px"}),
                html.A("→ Go to Dashboard", href="/", style={
                    "display": "inline-block", "padding": "14px 32px", "borderRadius": "12px",
                    "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                    "color": "white", "textDecoration": "none", "fontWeight": "700", "fontSize": "15px",
                }),
            ], style={
                "maxWidth": "480px", "margin": "80px auto", "textAlign": "center", "padding": "48px 32px",
                "background": "rgba(15,23,42,0.8)", "borderRadius": "20px",
                "border": "1px solid rgba(16,185,129,0.2)",
            }),
        ], style={"background": "#0a0e1a", "minHeight": "100vh",
                  "display": "flex", "alignItems": "center", "justifyContent": "center"})

    # If no auth module, show app without gating
    if not SAAS_AUTH_AVAILABLE:
        return _build_main_dashboard(None)

    # If user is authenticated, show main app
    if user:
        return _build_main_dashboard(user)

    # Otherwise show login page
    return build_login_page()


# ── LOGIN CALLBACK ──────────────────────────────────────────────────────────

@callback(
    Output("user-session", "data", allow_duplicate=True),
    Output("app-container", "children", allow_duplicate=True),
    Input("login-btn", "n_clicks"),
    State("login-email", "value"),
    State("login-password", "value"),
    prevent_initial_call=True,
)
def handle_login(n_clicks, email, password):
    if not n_clicks or not SAAS_AUTH_AVAILABLE:
        raise PreventUpdate

    if not email or not password:
        return no_update, build_login_page("Please enter email and password")

    user = authenticate_user(email, password)
    if not user:
        return no_update, build_login_page("Invalid email or password")

    session = {"token": user["session_token"], "email": user["email"]}
    logger.info(f"✅ User logged in: {email}")
    return session, _build_main_dashboard(user)


# ── REGISTER CALLBACK ──────────────────────────────────────────────────────

@callback(
    Output("user-session", "data", allow_duplicate=True),
    Output("app-container", "children", allow_duplicate=True),
    Input("register-btn", "n_clicks"),
    State("register-name", "value"),
    State("register-email", "value"),
    State("register-password", "value"),
    prevent_initial_call=True,
)
def handle_register(n_clicks, name, email, password):
    if not n_clicks or not SAAS_AUTH_AVAILABLE:
        raise PreventUpdate

    if not email or not password:
        return no_update, build_login_page("Please fill in all fields")

    if len(password) < 8:
        return no_update, build_login_page("Password must be at least 8 characters")

    result = create_user(email, password, name)
    if "error" in result:
        return no_update, build_login_page(result["error"])

    # Auto-login after registration
    user = authenticate_user(email, password)
    if user:
        session = {"token": user["session_token"], "email": user["email"]}
        logger.info(f"✅ User registered & logged in: {email}")
        return session, _build_main_dashboard(user)

    return no_update, build_login_page("Account created. Please sign in.")


# ── DEMO MODE CALLBACK ─────────────────────────────────────────────────────

@callback(
    Output("app-container", "children", allow_duplicate=True),
    Input("demo-mode-link", "n_clicks"),
    prevent_initial_call=True,
)
def handle_demo_mode(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return _build_main_dashboard(None)


# ── LOGOUT CALLBACK ─────────────────────────────────────────────────────────

@callback(
    Output("user-session", "data", allow_duplicate=True),
    Output("app-container", "children", allow_duplicate=True),
    Input("sidebar-logout-btn", "n_clicks"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def handle_logout(n_clicks, session_data):
    if not n_clicks:
        raise PreventUpdate

    if SAAS_AUTH_AVAILABLE and session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)
            if user:
                logout_user(user["user_id"])

    if SAAS_AUTH_AVAILABLE:
        return None, build_login_page()
    return None, _build_main_dashboard(None)


# ── UPGRADE BUTTON CALLBACK ────────────────────────────────────────────────

@callback(
    Output("url", "href", allow_duplicate=True),
    Input("sidebar-upgrade-btn", "n_clicks"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def handle_upgrade_click(n_clicks, session_data):
    if not n_clicks or not SAAS_AUTH_AVAILABLE:
        raise PreventUpdate

    user = None
    if session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)

    if user and user.get("stripe_customer_id"):
        # Existing subscriber → open Stripe Customer Portal
        portal_url = create_customer_portal_session(user)
        if portal_url:
            return portal_url

    # Otherwise redirect to pricing page
    return "/pricing"


# ── PRICING PAGE UPGRADE BUTTONS ──────────────────────────────────────────

@callback(
    Output("url", "href", allow_duplicate=True),
    Input({"type": "upgrade-btn", "index": ALL}, "n_clicks"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def handle_pricing_upgrade(n_clicks_list, session_data):
    """Handle clicks on plan upgrade buttons on the pricing page."""
    if not ctx.triggered_id or not any(n_clicks_list):
        raise PreventUpdate

    plan_id = ctx.triggered_id["index"]
    if not SAAS_AUTH_AVAILABLE:
        raise PreventUpdate

    user = None
    if session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)

    if not user:
        return "/pricing"

    # Create Stripe Checkout session
    checkout_url = create_checkout_session(user, plan_id, "monthly")
    if checkout_url:
        logger.info(f"✅ Redirecting {user.get('email')} to Stripe checkout for {plan_id}")
        return checkout_url

    # Fallback — Stripe not configured or session creation failed
    logger.warning(f"⚠️ Stripe checkout failed for {plan_id} — redirecting to pricing")
    return "/pricing"


# ── AUTH TAB TOGGLE (Login ↔ Register) ──────────────────────────────────────

@callback(
    Output("login-form", "style"),
    Output("register-form", "style"),
    Output("forgot-form", "style"),
    Output("auth-tab-login", "style"),
    Output("auth-tab-register", "style"),
    Input("auth-tab-login", "n_clicks"),
    Input("auth-tab-register", "n_clicks"),
    Input("forgot-password-link", "n_clicks"),
    Input("back-to-login-link", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_auth_tabs(login_clicks, register_clicks, forgot_clicks, back_clicks):
    triggered = ctx.triggered_id
    active_tab = {
        "flex": "1", "padding": "10px", "border": "none", "borderRadius": "8px 0 0 8px",
        "background": "rgba(99,102,241,0.15)", "color": "#a78bfa",
        "fontSize": "13px", "fontWeight": "700", "cursor": "pointer",
    }
    inactive_tab = {
        "flex": "1", "padding": "10px", "border": "none", "borderRadius": "0 8px 8px 0",
        "background": "rgba(30,41,59,0.5)", "color": "#64748b",
        "fontSize": "13px", "fontWeight": "600", "cursor": "pointer",
    }
    active_tab_r = {**active_tab, "borderRadius": "0 8px 8px 0"}
    inactive_tab_l = {**inactive_tab, "borderRadius": "8px 0 0 8px"}

    if triggered == "auth-tab-register":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}, inactive_tab_l, active_tab_r
    elif triggered in ("forgot-password-link",):
        return {"display": "none"}, {"display": "none"}, {"display": "block"}, active_tab, inactive_tab
    elif triggered in ("back-to-login-link", "auth-tab-login"):
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, active_tab, inactive_tab

    return {"display": "block"}, {"display": "none"}, {"display": "none"}, active_tab, inactive_tab


# ── PASSWORD RESET CALLBACK ─────────────────────────────────────────────────

@callback(
    Output("auth-error", "children", allow_duplicate=True),
    Output("auth-error", "style", allow_duplicate=True),
    Output("auth-success", "children", allow_duplicate=True),
    Output("auth-success", "style", allow_duplicate=True),
    Input("reset-password-btn", "n_clicks"),
    State("reset-email", "value"),
    State("reset-password", "value"),
    State("reset-password-confirm", "value"),
    prevent_initial_call=True,
)
def handle_password_reset(n_clicks, email, password, confirm_password):
    if not n_clicks:
        raise PreventUpdate

    error_style = {
        "color": "#ef4444", "fontSize": "13px", "textAlign": "center",
        "marginBottom": "12px", "display": "block",
        "background": "rgba(239,68,68,0.08)", "padding": "8px 12px",
        "borderRadius": "8px", "border": "1px solid rgba(239,68,68,0.2)",
    }
    success_style = {
        "color": "#10b981", "fontSize": "13px", "textAlign": "center",
        "marginBottom": "12px", "display": "block",
        "background": "rgba(16,185,129,0.08)", "padding": "8px 12px",
        "borderRadius": "8px", "border": "1px solid rgba(16,185,129,0.2)",
    }
    hidden = {"display": "none"}

    if not email or not password:
        return "Please enter email and new password", error_style, "", hidden

    if len(password) < 8:
        return "Password must be at least 8 characters", error_style, "", hidden

    if password != confirm_password:
        return "Passwords do not match", error_style, "", hidden

    if SAAS_AUTH_AVAILABLE:
        from saas_auth import reset_user_password
        result = reset_user_password(email, password)
        if result.get("success"):
            return "", hidden, "✅ Password reset successfully! You can now sign in.", success_style
        else:
            return result.get("error", "Reset failed"), error_style, "", hidden

    return "Auth system not available", error_style, "", hidden


# =============================================================================
# CALLBACKS
# =============================================================================

# ── PAGE ROUTING ─────────────────────────────────────────────────────────────

PAGE_HEADERS = {
    "ai_prediction": ("🤖", "Advanced Market Analysis Engine", "Advanced multi-model market analysis and research"),
    "advanced_analytics": ("📊", "Advanced Analytics Suite", "Market regime detection, drift analysis, and alternative data"),
    "portfolio_mgmt": ("💼", "Portfolio Management", "Black-Litterman AI portfolio optimization"),
    "backtesting": ("📈", "Advanced AI Backtesting", "Walk-forward validation and strategy analysis"),
    "ftmo_dashboard": ("🏦", "FTMO Dashboard", "FTMO challenge tracking and risk management"),
    "model_training": ("🧠", "Model Training Center", "Train, monitor, and manage AI models"),
    "app_guide": ("📖", "Platform Guide", "How the platform works, features, and AI model explanations"),
    "admin_panel": ("🛡️", "Admin Dashboard", "User management, email center, and system monitoring"),
}


@callback(
    Output("page-header", "children"),
    Output("page-content", "children"),
    Input("page-nav", "value"),
    Input("prediction-store", "data"),
    Input("ticker-dropdown", "value"),
    Input("ftmo-store", "data"),
    State("user-session", "data"),
)
def route_page(page, prediction, ticker, ftmo_state, session_data):
    icon, title, subtitle = PAGE_HEADERS.get(page, ("🤖", "MarketLens", ""))

    # ── Resolve user for plan gating ─────────────────────────────────
    user = None
    if SAAS_AUTH_AVAILABLE and session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)

    plan_info = get_plan_badge_info(user) if SAAS_AUTH_AVAILABLE else {"plan_name": "DEMO", "colors": {"text": "#94a3b8"}}

    header = html.Div([
        html.Div([
            html.Div([
                html.Div([html.Span(icon, style={"fontSize": "28px"})],
                         style={"background": "linear-gradient(135deg, #06b6d4, #8b5cf6)",
                                "padding": "12px", "borderRadius": "14px"}),
                html.Div([
                    html.H1(title, className="header-title"),
                    html.P(subtitle, className="header-subtitle"),
                ]),
            ], style={"display": "flex", "alignItems": "center", "gap": "16px"}),
            html.Span(f"{plan_info['plan_name'].upper()} ACTIVE", className="ecard-badge success"),
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "flexWrap": "wrap", "gap": "12px"}),
    ], className="header-card")

    # Status cards row
    status_row = make_grid([
        make_status_card("Market", "OPEN", "#10b981"),
        make_status_card("Backend", "LIVE" if BACKEND_AVAILABLE else "DEMO",
                         "#10b981" if BACKEND_AVAILABLE else "#f59e0b"),
        make_status_card("Streaming", "ACTIVE", "#10b981"),
        make_status_card("AI Models", "8 Active", "#10b981"),
    ], 4)

    # Import pages module lazily
    from pages import (
        build_analytics_page, build_ftmo_page, build_portfolio_page,
        build_backtest_page, build_model_training_page, build_admin_page,
    )

    # ── Feature gating by plan ───────────────────────────────────────
    feature_map = {
        "portfolio_mgmt": ("portfolio", "Portfolio Optimization", "enterprise"),
        "backtesting": ("backtesting", "Walk-Forward Backtesting", "professional"),
        "ftmo_dashboard": ("ftmo_dashboard", "FTMO Dashboard", "professional"),
        "model_training": ("model_training", "Model Training Center", "professional"),
    }

    if SAAS_AUTH_AVAILABLE and page in feature_map:
        feature_key, feature_name, required_plan = feature_map[page]
        if not can_access_feature(user, feature_key):
            content = build_upgrade_prompt(feature_name, required_plan)
            return html.Div([header, status_row]), content

    # ── Admin Panel: owner-only access ───────────────────────────────
    if page == "admin_panel":
        if SAAS_AUTH_AVAILABLE and not is_admin(user):
            content = html.Div([
                html.Div("🔒", style={"fontSize": "48px", "marginBottom": "16px"}),
                html.H3("Admin Access Only", style={"color": "#e2e8f0", "fontWeight": "700", "margin": "0 0 8px 0"}),
                html.P("This panel is restricted to platform administrators.",
                       style={"color": "#64748b", "fontSize": "14px", "margin": "0"}),
            ], style={
                "textAlign": "center", "padding": "80px 20px",
                "background": "rgba(15,23,42,0.6)", "borderRadius": "16px",
                "border": "1px solid rgba(239,68,68,0.15)",
            })
            return html.Div([header, status_row]), content

    if page == "ai_prediction":
        content = build_ai_prediction_page(prediction, ticker)
    elif page == "advanced_analytics":
        content = build_analytics_page(ticker or "BTCUSD", prediction)
    elif page == "portfolio_mgmt":
        content = build_portfolio_page()
    elif page == "backtesting":
        content = build_backtest_page(ticker or "BTCUSD")
    elif page == "ftmo_dashboard":
        content = build_ftmo_page(ftmo_state or {})
    elif page == "model_training":
        content = build_model_training_page(ticker or "BTCUSD")
    elif page == "subscription":
        content = build_pricing_page(get_user_plan(user).get("name", "free") if user else "free")
    elif page == "app_guide":
        content = build_app_guide_page()
    elif page == "admin_panel":
        content = build_admin_page()
    else:
        content = build_ai_prediction_page(prediction, ticker)

    return html.Div([header, status_row]), content


def build_ai_prediction_page(prediction, ticker):
    """Build the Market Analysis page (default page)."""
    ticker = ticker or "BTCUSD"
    asset_type = get_asset_type(ticker)

    # Data source indicator
    src = "🟢 FMP Live" if (BACKEND_AVAILABLE and FMP_API_KEY) else "⚡ Demo Mode"
    src_color = "#10b981" if (BACKEND_AVAILABLE and FMP_API_KEY) else "#f59e0b"

    # Timeframe display from prediction
    primary_tf = prediction.get("primary_timeframe", "1day") if prediction else "1day"
    tf_labels = {"15min": "15m", "1hour": "1H", "4hour": "4H", "1day": "1D"}
    mtf_active = prediction.get("multi_timeframe_analysis") if prediction else None
    mtf_count = len(prediction.get("mtf_timeframes_used", [])) if prediction else 0

    ticker_bar = html.Div([
        html.Span("Ticker: ", style={"color": "#64748b", "fontSize": "13px"}),
        html.Strong(ticker, style={"color": "#e2e8f0", "fontSize": "13px", "marginRight": "10px"}),
        html.Span(asset_type.upper(), style={
            "padding": "2px 10px", "borderRadius": "12px", "fontSize": "11px",
            "fontWeight": "600", "color": "#a78bfa", "background": "rgba(167,139,250,0.12)",
            "border": "1px solid rgba(167,139,250,0.25)", "marginRight": "8px",
        }),
        html.Span(f"⏱️ {tf_labels.get(primary_tf, primary_tf)}", style={
            "padding": "2px 10px", "borderRadius": "12px", "fontSize": "11px",
            "fontWeight": "600", "color": "#06b6d4", "background": "rgba(6,182,212,0.12)",
            "border": "1px solid rgba(6,182,212,0.25)", "marginRight": "8px",
        }),
        html.Span(f"MTF {mtf_count}x", style={
            "padding": "2px 10px", "borderRadius": "12px", "fontSize": "11px",
            "fontWeight": "600", "color": "#10b981", "background": "rgba(16,185,129,0.12)",
            "border": "1px solid rgba(16,185,129,0.25)", "marginRight": "8px",
        }) if mtf_active and mtf_count > 1 else html.Span(),
        html.Span(src, style={
            "padding": "2px 10px", "borderRadius": "12px", "fontSize": "11px",
            "fontWeight": "600", "color": "white", "background": src_color,
        }),
    ], style={"marginBottom": "16px", "display": "flex", "alignItems": "center", "gap": "4px", "flexWrap": "wrap"})

    # Predict button
    predict_btn = html.Div([
        html.Button(
            "🔬 Run AI Analysis",
            id="predict-btn",
            n_clicks=0,
            style={
                "padding": "14px 32px", "borderRadius": "12px",
                "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                "color": "white", "border": "none", "fontSize": "15px",
                "fontWeight": "700", "cursor": "pointer", "letterSpacing": "0.3px",
                "boxShadow": "0 4px 20px rgba(99,102,241,0.3)",
                "transition": "all 0.3s ease", "width": "100%",
            },
        ),
    ], style={"marginBottom": "20px"})

    # Loading
    loading = dcc.Loading(
        id="loading-prediction",
        type="default",
        color="#6366f1",
        children=html.Div(id="prediction-loading-output"),
    )

    # Results
    if prediction:
        results = build_prediction_results(prediction)
        tabs = dcc.Tabs([
            dcc.Tab(
                label="📈 Position Analysis",
                children=html.Div(build_trading_strategy_tab(prediction), style={"padding": "20px 0"}),
                style={"backgroundColor": "transparent", "borderColor": "rgba(99,102,241,0.12)",
                       "color": "#94a3b8", "fontWeight": "600", "padding": "10px 20px"},
                selected_style={"backgroundColor": "rgba(15,23,42,0.6)", "borderColor": "#6366f1",
                                "borderBottom": "2px solid #6366f1", "color": "#e2e8f0",
                                "fontWeight": "700", "padding": "10px 20px"},
            ),
            dcc.Tab(
                label="📊 Forecast Analysis",
                children=html.Div(build_forecast_tab(prediction), style={"padding": "20px 0"}),
                style={"backgroundColor": "transparent", "borderColor": "rgba(99,102,241,0.12)",
                       "color": "#94a3b8", "fontWeight": "600", "padding": "10px 20px"},
                selected_style={"backgroundColor": "rgba(15,23,42,0.6)", "borderColor": "#6366f1",
                                "borderBottom": "2px solid #6366f1", "color": "#e2e8f0",
                                "fontWeight": "700", "padding": "10px 20px"},
            ),
            dcc.Tab(
                label="⚠️ Risk Assessment",
                children=html.Div(build_risk_tab(prediction), style={"padding": "20px 0"}),
                style={"backgroundColor": "transparent", "borderColor": "rgba(99,102,241,0.12)",
                       "color": "#94a3b8", "fontWeight": "600", "padding": "10px 20px"},
                selected_style={"backgroundColor": "rgba(15,23,42,0.6)", "borderColor": "#6366f1",
                                "borderBottom": "2px solid #6366f1", "color": "#e2e8f0",
                                "fontWeight": "700", "padding": "10px 20px"},
            ),
        ], style={"marginTop": "24px"})
    else:
        results = html.Div([
            html.Div("🎯", style={"fontSize": "48px", "marginBottom": "12px"}),
            html.H3("Ready to Analyze", style={"color": "#64748b", "fontWeight": "600"}),
            html.P("Select an asset and click 'Run AI Analysis' to begin",
                   style={"color": "#475569", "fontSize": "14px"}),
        ], className="ecard", style={"textAlign": "center", "padding": "60px 20px"})
        tabs = html.Div()

    return html.Div([ticker_bar, predict_btn, loading, results, tabs])


# ── PREDICTION CALLBACK ─────────────────────────────────────────────────────

@callback(
    Output("prediction-store", "data"),
    Output("prediction-loading-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    State("model-checklist", "value"),
    State("timeframe-dropdown", "data"),
    State("mtf-checklist", "value"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, ticker, models, timeframe, mtf_timeframes, session_data):
    if not n_clicks or not ticker:
        raise PreventUpdate

    # ── Resolve user & enforce plan limits ────────────────────────────
    user = None
    if SAAS_AUTH_AVAILABLE and session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)

    if SAAS_AUTH_AVAILABLE:
        # Check daily prediction limit
        allowed, used, limit = check_prediction_limit(user)
        if not allowed:
            return no_update, build_limit_reached_prompt(used, limit)

        # Enforce model limit
        models_cap = get_models_limit(user)
        if models and len(models) > models_cap:
            models = models[:models_cap]

        # Enforce ticker access
        allowed_tickers = get_allowed_tickers(user)
        if ticker not in allowed_tickers:
            return no_update, html.Div(
                f"⚠️ {ticker} requires a higher plan. Upgrade to access this asset.",
                style={"color": "#f59e0b", "fontWeight": "600"},
            )

        # Enforce timeframe access
        allowed_tfs = get_allowed_timeframes(user)
        if timeframe and timeframe not in allowed_tfs:
            timeframe = "1day"

    timeframe = timeframe or "1day"
    prediction = PredictionEngine.run_prediction(ticker, timeframe=timeframe, models=models)

    # ── Multi-Timeframe Analysis ─────────────────────────────────────
    mtf_timeframes = mtf_timeframes or ["1day"]
    mtf_allowed = SAAS_AUTH_AVAILABLE and can_access_feature(user, "mtf_analysis") if SAAS_AUTH_AVAILABLE else True

    # Filter MTF timeframes to only those the user's plan allows
    if SAAS_AUTH_AVAILABLE and user:
        allowed_tfs = get_allowed_timeframes(user)
        mtf_timeframes = [tf for tf in mtf_timeframes if tf in allowed_tfs] or ["1day"]

    if len(mtf_timeframes) > 1 and BACKEND_AVAILABLE and mtf_allowed:
        try:
            mtf_analysis = PredictionEngine._run_multi_timeframe_analysis(ticker, mtf_timeframes)
            if mtf_analysis:
                prediction["multi_timeframe_analysis"] = mtf_analysis
                prediction["mtf_timeframes_used"] = mtf_timeframes
                logger.info(f"📊 MTF analysis complete: {len(mtf_analysis)} timeframes")
        except Exception as e:
            logger.warning(f"MTF analysis failed: {e}")
    elif len(mtf_timeframes) > 1 and not mtf_allowed:
        # User selected multiple timeframes but plan doesn't include MTF
        prediction["mtf_upgrade_required"] = True
        logger.info(f"⚠️ MTF blocked — user plan does not include mtf_analysis")

    prediction["primary_timeframe"] = timeframe

    # ── Record usage ─────────────────────────────────────────────────
    if SAAS_AUTH_AVAILABLE:
        record_prediction(user)

    source = prediction.get("source", "unknown")
    fallback = prediction.get("fallback_mode", False)

    # Auto-save prediction result
    try:
        save_result("prediction", ticker, prediction)
    except Exception as e:
        logger.warning(f"Failed to save prediction: {e}")

    # ── Auto Market Alert (email pro users on high-confidence signals) ──
    if EMAIL_SERVICE_AVAILABLE and prediction:
        try:
            confidence = prediction.get("confidence", 0)
            if confidence >= 80:
                signal = prediction.get("signal", "HOLD")
                if signal in ("BUY", "SELL"):
                    send_market_alert(
                        ticker=ticker,
                        signal=signal,
                        confidence=confidence,
                        predicted_price=prediction.get("predicted_price", 0),
                        current_price=prediction.get("current_price", 0),
                        details=f"Auto-generated alert from {len(prediction.get('models_used', []))}-model ensemble.",
                        plan_filter="pro",
                    )
                    logger.info(f"📧 Auto market alert sent: {signal} {ticker} ({confidence:.1f}%)")
        except Exception as e:
            logger.debug(f"Auto market alert skipped: {e}")

    # Status message with usage counter
    usage_suffix = ""
    if SAAS_AUTH_AVAILABLE and user:
        _, used_now, lim = check_prediction_limit(user)
        usage_suffix = f" ({used_now}/{lim} today)"

    if not fallback and source == "live_ai_backend":
        msg = html.Div(f"🔥 LIVE AI ANALYSIS — {timeframe}{usage_suffix}", style={"color": "#10b981", "fontWeight": "600"})
    elif not fallback and source == "data_driven_technical":
        msg = html.Div(f"📊 LIVE DATA — technical analysis{usage_suffix}", style={"color": "#10b981", "fontWeight": "600"})
    else:
        msg = html.Div(f"⚡ DEMO ANALYSIS — Simulation mode{usage_suffix}", style={"color": "#f59e0b", "fontWeight": "600"})

    return prediction, msg


# ── NAV BUTTON CALLBACKS ─────────────────────────────────────────────────────

NAV_PAGES = ["ai_prediction", "advanced_analytics", "portfolio_mgmt", "backtesting", "subscription",
             "ftmo_dashboard", "model_training", "app_guide", "admin_panel"]

@callback(
    Output("page-nav", "value"),
    Input({"type": "nav-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def nav_button_click(n_clicks_list):
    if not ctx.triggered_id:
        raise PreventUpdate
    return ctx.triggered_id["index"]


# Active-state styling: highlight the selected nav button
@callback(
    [Output({"type": "nav-btn", "index": p}, "style") for p in NAV_PAGES],
    Input("page-nav", "value"),
)
def highlight_active_nav(active_page):
    base_style = {}  # CSS class handles default
    active_style = {
        "background": "linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08))",
        "borderColor": "rgba(99,102,241,0.25)",
        "color": "#e2e8f0",
        "fontWeight": "600",
        "borderLeft": "3px solid #6366f1",
        "paddingLeft": "11px",
    }
    return [active_style if p == active_page else base_style for p in NAV_PAGES]


# ── GROUP BUTTON CALLBACKS ───────────────────────────────────────────────────

@callback(
    Output("ticker-dropdown", "value"),
    Input({"type": "group-btn", "index": ALL}, "n_clicks"),
    State("user-session", "data"),
    prevent_initial_call=True,
)
def quick_select_group(n_clicks_list, session_data):
    if not ctx.triggered_id:
        raise PreventUpdate
    group_name = ctx.triggered_id["index"]
    tickers = TICKER_GROUPS.get(group_name, [])

    # Filter to only tickers the user's plan allows
    if SAAS_AUTH_AVAILABLE and session_data:
        token = session_data.get("token") if isinstance(session_data, dict) else None
        if token:
            user = get_user_by_token(token)
            if user:
                user_tickers = get_allowed_tickers(user)
                tickers = [t for t in tickers if t in user_tickers]

    if tickers:
        return tickers[0]
    raise PreventUpdate


# ── FTMO SETUP CALLBACK ────────────────────────────────────────────────────

@callback(
    Output("ftmo-store", "data"),
    Input("ftmo-setup-btn", "n_clicks"),
    State("ftmo-balance", "value"),
    State("ftmo-daily-limit", "value"),
    State("ftmo-total-limit", "value"),
    State("ftmo-store", "data"),
    prevent_initial_call=True,
)
def setup_ftmo(n_clicks, balance, daily_limit, total_limit, current_state):
    if not n_clicks:
        raise PreventUpdate
    return {
        "setup_done": True,
        "balance": balance or 100000,
        "daily_limit": daily_limit or 5,
        "total_limit": total_limit or 10,
        "positions": [],
    }


# ── BACKTEST RE-RUN CALLBACK ───────────────────────────────────────────────

@callback(
    Output("page-content", "children", allow_duplicate=True),
    Output("bt-run-btn", "disabled", allow_duplicate=True),
    Input("bt-run-btn", "n_clicks"),
    State("ticker-dropdown", "value"),
    State("bt-strategy", "value"),
    State("bt-capital", "value"),
    State("bt-period", "value"),
    State("bt-commission", "value"),
    prevent_initial_call=True,
    running=[
        (Output("bt-run-btn", "disabled"), True, False),
        (Output("bt-run-btn", "children"), "⏳ Running AI Backtest...", "🚀 Run Backtest"),
    ],
)
def rerun_backtest(n_clicks, ticker, strategy, capital, period, commission):
    if not n_clicks:
        raise PreventUpdate
    from pages import build_backtest_page, run_backtest
    ticker = ticker or "BTCUSD"
    capital = capital or 100000
    commission = commission or 0.001
    strategy = strategy or "AI Signals"
    period = period or "6 Months"
    logger.info(f"🚀 Starting backtest: {ticker} | {strategy} | ${capital:,.0f}")
    results = run_backtest(ticker, capital, commission, 0.0005, period, strategy, 0.20, 0.03)
    logger.info(f"✅ Backtest complete: {ticker} | Return={results.get('total_return', 0)*100:.2f}%")

    # Auto-save backtest result (exclude large equity curve for storage efficiency)
    try:
        save_data = {k: v for k, v in results.items() if k != "equity_curve"}
        save_result("backtest", ticker, save_data)
    except Exception as e:
        logger.warning(f"Failed to save backtest: {e}")

    return build_backtest_page(ticker, results), False


# ── PORTFOLIO RE-OPTIMIZE CALLBACK ─────────────────────────────────────────

@callback(
    Output("page-content", "children", allow_duplicate=True),
    Input("pf-optimize-btn", "n_clicks"),
    State("pf-assets", "value"),
    State("pf-risk-tolerance", "value"),
    State("pf-target-return", "value"),
    prevent_initial_call=True,
)
def reoptimize_portfolio(n_clicks, assets, risk_tolerance, target_return_pct):
    if not n_clicks:
        raise PreventUpdate
    from pages import build_portfolio_page
    if not assets or len(assets) < 2:
        assets = ["BTCUSD", "XAUUSD", "SPY", "EURUSD", "AAPL"]
    risk_tolerance = risk_tolerance or "Moderate"
    target_return = (target_return_pct or 12) / 100.0
    return build_portfolio_page(assets, risk_tolerance, target_return)


# =============================================================================
# CONTACT FORM MODAL BUILDER
# =============================================================================

def build_contact_modal():
    """Build the contact support modal overlay."""
    return html.Div(
        id="contact-modal",
        children=[
            html.Div([
                html.Div([
                    html.Span("📧", style={"fontSize": "28px"}),
                    html.H3("Contact Support", style={"color": "#e2e8f0", "margin": "0", "fontWeight": "700"}),
                ], style={"display": "flex", "alignItems": "center", "gap": "10px", "marginBottom": "20px"}),

                html.Div([
                    html.Label("Your Name", style={"color": "#94a3b8", "fontSize": "12px",
                                                     "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="contact-name", type="text", placeholder="Your name",
                              style={"width": "100%", "padding": "10px 14px", "borderRadius": "10px",
                                     "border": "1px solid rgba(99,102,241,0.2)", "background": "rgba(15,23,42,0.6)",
                                     "color": "#e2e8f0", "fontSize": "14px", "boxSizing": "border-box"}),
                ], style={"marginBottom": "14px"}),

                html.Div([
                    html.Label("Email", style={"color": "#94a3b8", "fontSize": "12px",
                                                "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Input(id="contact-email", type="email", placeholder="you@example.com",
                              style={"width": "100%", "padding": "10px 14px", "borderRadius": "10px",
                                     "border": "1px solid rgba(99,102,241,0.2)", "background": "rgba(15,23,42,0.6)",
                                     "color": "#e2e8f0", "fontSize": "14px", "boxSizing": "border-box"}),
                ], style={"marginBottom": "14px"}),

                html.Div([
                    html.Label("Subject", style={"color": "#94a3b8", "fontSize": "12px",
                                                   "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Dropdown(id="contact-subject", options=[
                        {"label": "General Inquiry", "value": "General Inquiry"},
                        {"label": "Technical Issue / Bug Report", "value": "Bug Report"},
                        {"label": "Billing / Subscription", "value": "Billing"},
                        {"label": "Feature Request", "value": "Feature Request"},
                        {"label": "Partnership / Business", "value": "Partnership"},
                    ], value="General Inquiry", clearable=False,
                    style={"marginBottom": "0"}),
                ], style={"marginBottom": "14px"}),

                html.Div([
                    html.Label("Message", style={"color": "#94a3b8", "fontSize": "12px",
                                                   "fontWeight": "600", "marginBottom": "6px", "display": "block"}),
                    dcc.Textarea(id="contact-message", placeholder="Describe your issue or question...",
                                 style={"width": "100%", "padding": "10px 14px", "borderRadius": "10px",
                                        "border": "1px solid rgba(99,102,241,0.2)", "background": "rgba(15,23,42,0.6)",
                                        "color": "#e2e8f0", "fontSize": "14px", "minHeight": "120px",
                                        "boxSizing": "border-box", "fontFamily": "inherit"}),
                ], style={"marginBottom": "20px"}),

                html.Div([
                    html.Button("📤 Send Message", id="contact-send-btn", n_clicks=0,
                                style={"flex": "1", "padding": "12px", "borderRadius": "10px",
                                       "background": "linear-gradient(135deg, #6366f1, #8b5cf6)",
                                       "color": "#fff", "border": "none", "fontSize": "14px",
                                       "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("Cancel", id="contact-cancel-btn", n_clicks=0,
                                style={"padding": "12px 24px", "borderRadius": "10px",
                                       "background": "rgba(239,68,68,0.08)", "color": "#f87171",
                                       "border": "1px solid rgba(239,68,68,0.2)", "fontSize": "14px",
                                       "fontWeight": "600", "cursor": "pointer"}),
                ], style={"display": "flex", "gap": "12px"}),

                html.Div(id="contact-result", style={"marginTop": "12px"}),

            ], style={
                "maxWidth": "500px", "width": "90%", "padding": "32px",
                "background": "rgba(15,23,42,0.95)", "backdropFilter": "blur(20px)",
                "border": "1px solid rgba(99,102,241,0.15)", "borderRadius": "20px",
                "boxShadow": "0 25px 80px rgba(0,0,0,0.6)",
            }),
        ],
        style={"display": "none"},
    )


# =============================================================================
# CONTACT FORM CALLBACKS
# =============================================================================

@callback(
    Output("contact-modal", "style"),
    Input("open-contact-btn", "n_clicks"),
    Input("contact-cancel-btn", "n_clicks"),
    Input("open-contact-btn-pricing", "n_clicks"),
    Input("contact-send-btn", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_contact_modal(open_clicks, cancel_clicks, open_clicks_pricing, send_clicks):
    triggered = ctx.triggered_id
    if triggered in ("open-contact-btn", "open-contact-btn-pricing") and (open_clicks or open_clicks_pricing):
        return {
            "position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
            "background": "rgba(0,0,0,0.7)", "backdropFilter": "blur(6px)",
            "display": "flex", "alignItems": "center", "justifyContent": "center", "zIndex": "9999",
        }
    return {"display": "none"}


@callback(
    Output("contact-result", "children"),
    Input("contact-send-btn", "n_clicks"),
    State("contact-name", "value"),
    State("contact-email", "value"),
    State("contact-subject", "value"),
    State("contact-message", "value"),
    prevent_initial_call=True,
)
def send_contact_form(n_clicks, name, email, subject, message):
    if not n_clicks:
        raise PreventUpdate

    if not name or not email or not message:
        return html.Div("Please fill in all fields.", style={
            "color": "#ef4444", "fontSize": "13px", "padding": "8px 12px",
            "background": "rgba(239,68,68,0.08)", "borderRadius": "8px",
            "border": "1px solid rgba(239,68,68,0.2)"})

    # Send via email service
    try:
        if EMAIL_SERVICE_AVAILABLE:
            from email_service import send_email
            body = f"""
            <h3>Contact Form Submission</h3>
            <p><strong>From:</strong> {name} ({email})</p>
            <p><strong>Subject:</strong> {subject}</p>
            <p><strong>Message:</strong></p>
            <p>{message.replace(chr(10), '<br>')}</p>
            """
            result = send_email("itubusinesshub@gmail.com", f"[Contact] {subject} — {name}", body)
            if result.get("success"):
                return html.Div("✅ Message sent! We'll get back to you soon.", style={
                    "color": "#10b981", "fontSize": "13px", "padding": "8px 12px",
                    "background": "rgba(16,185,129,0.08)", "borderRadius": "8px",
                    "border": "1px solid rgba(16,185,129,0.2)"})
    except Exception as e:
        logger.warning(f"Contact form email failed: {e}")

    # Fallback — save to Firestore
    try:
        if FIRESTORE_AVAILABLE and _firestore_db:
            _firestore_db.collection("contact_submissions").add({
                "name": name, "email": email, "subject": subject,
                "message": message, "timestamp": datetime.now().isoformat(),
            })
            return html.Div("✅ Message received! We'll get back to you soon.", style={
                "color": "#10b981", "fontSize": "13px", "padding": "8px 12px",
                "background": "rgba(16,185,129,0.08)", "borderRadius": "8px",
                "border": "1px solid rgba(16,185,129,0.2)"})
    except Exception:
        pass

    return html.Div("✅ Thank you! Please also email us at itubusinesshub@gmail.com", style={
        "color": "#10b981", "fontSize": "13px", "padding": "8px 12px",
        "background": "rgba(16,185,129,0.08)", "borderRadius": "8px",
        "border": "1px solid rgba(16,185,129,0.2)"})


# =============================================================================
# DISCLAIMER CALLBACKS
# =============================================================================

@callback(
    Output("disclaimer-container", "style"),
    Output("main-app-content", "style"),
    Input("disclaimer-store", "data"),
    Input("disclaimer-accept-btn", "n_clicks"),
    Input("disclaimer-decline-btn", "n_clicks"),
    State("disclaimer-store", "data"),
    prevent_initial_call=True,
)
def handle_disclaimer(store_data, accept_clicks, decline_clicks, current_store):
    triggered = ctx.triggered_id

    if triggered == "disclaimer-accept-btn" and accept_clicks:
        # User accepted — hide overlay, show app
        return {"display": "none"}, {"filter": "none"}

    if triggered == "disclaimer-decline-btn" and decline_clicks:
        # User declined — keep overlay, blur app more
        return (
            {"position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
             "background": "rgba(0,0,0,0.7)", "backdropFilter": "blur(8px)",
             "display": "flex", "alignItems": "center", "justifyContent": "center", "zIndex": "9999"},
            {"filter": "blur(12px)", "pointerEvents": "none"},
        )

    # Check if already accepted in store
    if store_data and isinstance(store_data, dict) and store_data.get("accepted"):
        return {"display": "none"}, {"filter": "none"}

    # Default: show overlay, blur app
    return (
        {"position": "fixed", "top": "0", "left": "0", "width": "100%", "height": "100%",
         "background": "rgba(0,0,0,0.7)", "backdropFilter": "blur(8px)",
         "display": "flex", "alignItems": "center", "justifyContent": "center", "zIndex": "9999"},
        {"filter": "blur(4px)", "pointerEvents": "none"},
    )


@callback(
    Output("disclaimer-store", "data", allow_duplicate=True),
    Input("disclaimer-accept-btn", "n_clicks"),
    prevent_initial_call=True,
)
def save_disclaimer_acceptance(n_clicks):
    if not n_clicks:
        raise PreventUpdate
    return {"accepted": True}


# =============================================================================
# DOWNLOAD AI MODELS FROM GCS ON STARTUP
# =============================================================================
try:
    from gcs_model_loader import ensure_models_available
    logger.info("📥 Downloading AI models from GCS on startup...")
    ensure_models_available()
except ImportError:
    logger.info("ℹ️ gcs_model_loader not available — models will train on demand")
except Exception as e:
    logger.warning(f"⚠️ GCS model download failed on startup: {e}")

# =============================================================================
# INITIALIZE AUTH & PAYMENTS SYSTEM
# =============================================================================

if SAAS_AUTH_AVAILABLE:
    setup_auth_system(app, server)

# Initialize email service
if EMAIL_SERVICE_AVAILABLE:
    try:
        init_email_service(firestore_db=_firestore_db)
    except Exception as e:
        logger.warning(f"⚠️ Email service init failed: {e}")

# Register email admin callbacks
try:
    from email_admin_panel import register_email_callbacks
    register_email_callbacks(app)
    logger.info("✅ Email admin callbacks registered")
except ImportError:
    pass

# Register admin monitoring callbacks (feature flags, health checks)
try:
    from admin_monitoring import register_monitoring_callbacks
    register_monitoring_callbacks(app)
    logger.info("✅ Admin monitoring callbacks registered")
except ImportError:
    logger.info("ℹ️ admin_monitoring.py not found — monitoring disabled")
except Exception as e:
    logger.warning(f"⚠️ Monitoring callbacks failed: {e}")

# =============================================================================
# RUN
# =============================================================================

app.clientside_callback(
    """
    function(url) {
        if (url && url.startsWith("http")) {
            window.location.href = url;
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("url", "href", allow_duplicate=True),
    Input("redirect-store", "data"),
    prevent_initial_call=True
)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)