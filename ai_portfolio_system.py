# =============================================================================
# AI PORTFOLIO SYSTEM — ai_portfolio_system.py
# =============================================================================
# Production-grade portfolio management layer for AI Trading Professional.
#
# Responsibilities:
#   • Multi-asset allocation driven by AI ensemble predictions
#   • Mean-variance + Black-Litterman optimization using model views
#   • Regime-aware risk budgeting (GMM regimes → dynamic weights)
#   • Real-time P&L, drawdown, VaR monitoring
#   • Correlation-aware position limits
#   • Portfolio-level performance attribution
#
# 💼 ENHANCED PORTFOLIO FEATURES:
#   ✅ 1. Strict Risk Controls (max drawdown limit, daily loss limit — FTMO/prop)
#   ✅ 2. Better Diversification Control (per-asset & correlation limits)
#   ✅ 3. Stress Testing (2008, COVID crash simulations)
#   ✅ 4. Liquidity Awareness (volume-based position sizing)
#   ✅ 5. Performance Attribution (by asset, strategy, model)
#   ✅ 6. Real-Time Risk Monitoring (live drawdown, VaR, dashboard)
#
# Import into tradingprofessional.py:
#   from ai_portfolio_system import AIPortfolioManager, PortfolioSnapshot,
#                                    PortfolioOptimizer, RealTimeRiskMonitor,
#                                    StrictRiskController, DiversificationController,
#                                    StressTester, LiquidityManager,
#                                    PerformanceAttributor, EnhancedRiskDashboard
# =============================================================================

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# optional GMM
try:
    from sklearn.mixture import GaussianMixture
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AssetView:
    """
    AI-derived view for a single asset used in Black-Litterman.
    One AssetView per ticker per rebalancing cycle.
    """
    ticker: str
    predicted_return: float       # annualised, from ensemble
    confidence: float             # 0–100
    predicted_volatility: float   # annualised
    regime: str
    model_std: float              # ensemble disagreement
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioPosition:
    """A live position held in the portfolio."""
    ticker: str
    shares: float
    entry_price: float
    entry_time: datetime
    current_price: float = 0.0
    weight: float = 0.0           # target allocation weight
    unrealised_pnl: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    regime: str = "unknown"


@dataclass
class PortfolioSnapshot:
    """Point-in-time state of the entire portfolio."""
    timestamp: datetime
    total_value: float
    cash: float
    invested_value: float
    positions: Dict[str, PortfolioPosition]

    # Allocation
    weights: Dict[str, float]
    target_weights: Dict[str, float]

    # Performance
    daily_return: float
    total_return: float
    drawdown: float
    peak_value: float

    # Risk
    portfolio_volatility: float
    sharpe_ratio: float
    var_95: float
    correlation_matrix: Optional[pd.DataFrame] = None


@dataclass
class RebalanceAction:
    """Instruction emitted by the optimizer to the execution layer."""
    ticker: str
    action: str          # 'buy' | 'sell' | 'hold'
    current_weight: float
    target_weight: float
    delta_weight: float
    estimated_trade_value: float
    reason: str


# =============================================================================
# MARKET REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """
    Lightweight GMM-based regime detector.
    Fits on recent returns and classifies current period.
    Falls back to volatility-based heuristic if GMM unavailable.
    """

    REGIMES = {0: "bull", 1: "bear", 2: "high_vol", 3: "ranging"}

    def __init__(self, n_regimes: int = 3, lookback: int = 252):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self._gmm: Optional[Any] = None
        self._fitted = False

    def fit(self, returns: pd.Series) -> None:
        if not GMM_AVAILABLE:
            return
        data = returns.dropna().values.reshape(-1, 1)
        if len(data) < 30:
            return
        try:
            self._gmm = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            self._gmm.fit(data[-self.lookback:])
            self._fitted = True
        except Exception as exc:
            logger.warning(f"[Regime] GMM fit failed: {exc}")

    def predict(self, recent_returns: pd.Series) -> str:
        if self._fitted and self._gmm is not None:
            try:
                last_return = float(recent_returns.dropna().iloc[-1])
                label = int(self._gmm.predict([[last_return]])[0])
                # Map by mean return of that component
                means = self._gmm.means_.flatten()
                vol = np.sqrt(self._gmm.covariances_.flatten())
                idx_sorted = np.argsort(means)

                if label == idx_sorted[-1]:
                    return "bull"
                if label == idx_sorted[0]:
                    return "bear"
                if vol[label] == vol.max():
                    return "high_vol"
                return "ranging"
            except Exception:
                pass

        # Heuristic fallback
        try:
            r = recent_returns.dropna()
            if len(r) < 5:
                return "unknown"
            recent_vol = float(r.iloc[-20:].std() * np.sqrt(252)) if len(r) >= 20 else 0.20
            recent_trend = float(r.iloc[-20:].mean() * 252) if len(r) >= 20 else 0.0
            if recent_vol > 0.40:
                return "high_vol"
            if recent_trend > 0.10:
                return "bull"
            if recent_trend < -0.10:
                return "bear"
            return "ranging"
        except Exception:
            return "unknown"


# =============================================================================
# PORTFOLIO OPTIMIZER
# =============================================================================

class PortfolioOptimizer:
    """
    AI-augmented mean-variance optimizer with Black-Litterman views.

    Combines:
      1. Historical covariance (stability)
      2. AI model views on expected returns (forward-looking)
      3. Regime-aware risk scaling
    """

    def __init__(
        self,
        risk_aversion: float = 2.0,
        max_weight: float = 0.40,
        min_weight: float = 0.02,
        leverage_limit: float = 1.0,   # 1.0 = long-only, fully invested
        transaction_cost: float = 0.001,
    ):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.leverage_limit = leverage_limit
        self.transaction_cost = transaction_cost

    # ------------------------------------------------------------------
    # Black-Litterman
    # ------------------------------------------------------------------

    def black_litterman_views(
        self,
        tickers: List[str],
        historical_returns: pd.DataFrame,
        asset_views: List[AssetView],
        tau: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Blend market equilibrium returns with AI model views.

        Returns
        -------
        bl_returns  : (n_assets,) posterior expected returns
        bl_cov      : (n_assets, n_assets) posterior covariance
        """
        n = len(tickers)
        ticker_idx = {t: i for i, t in enumerate(tickers)}

        # 1. Historical covariance Σ
        if historical_returns.empty or historical_returns.shape[0] < 30:
            sigma = np.eye(n) * 0.04
        else:
            ret_vals = historical_returns[tickers].dropna().values
            sigma = np.cov(ret_vals.T) * 252 + np.eye(n) * 1e-6
            if sigma.shape != (n, n):
                sigma = np.eye(n) * 0.04

        # 2. Equilibrium (market cap weighted) implied returns
        mkt_weights = np.ones(n) / n
        pi = self.risk_aversion * sigma @ mkt_weights

        # 3. AI views  (one view per asset)
        P = np.zeros((len(asset_views), n))
        q = np.zeros(len(asset_views))
        omega_diag = np.zeros(len(asset_views))

        for k, view in enumerate(asset_views):
            if view.ticker in ticker_idx:
                j = ticker_idx[view.ticker]
                P[k, j] = 1.0
                q[k] = view.predicted_return
                # Uncertainty proportional to model disagreement + (1 - confidence)
                uncertainty = (1 - view.confidence / 100) + view.model_std / max(abs(view.predicted_return), 0.01)
                omega_diag[k] = max(uncertainty * tau * sigma[j, j], 1e-6)

        omega = np.diag(omega_diag)

        # 4. Black-Litterman posterior
        try:
            tau_sigma = tau * sigma
            M = np.linalg.inv(
                np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(omega) @ P
            )
            bl_returns = M @ (
                np.linalg.inv(tau_sigma) @ pi + P.T @ np.linalg.inv(omega) @ q
            )
            bl_cov = sigma + M
        except np.linalg.LinAlgError:
            logger.warning("[Optimizer] BL matrix inversion failed; using MV only")
            bl_returns = pi
            bl_cov = sigma

        return bl_returns, bl_cov

    # ------------------------------------------------------------------
    # Mean-Variance Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        tickers: List[str],
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: Optional[np.ndarray] = None,
        regime: str = "unknown",
    ) -> np.ndarray:
        """
        Solve the mean-variance problem.
        Applies regime-aware risk scaling before optimisation.
        """
        n = len(tickers)
        if n == 0:
            return np.array([])

        # Regime scaling: reduce risk in bear / high-vol regimes
        regime_scale = {
            "bear": 0.60,
            "high_vol": 0.50,
            "ranging": 0.90,
            "bull": 1.10,
        }.get(regime, 1.0)

        adj_returns = expected_returns * regime_scale

        # Transaction cost penalty
        if current_weights is not None:
            tc_penalty = self.transaction_cost * np.ones(n)
        else:
            tc_penalty = np.zeros(n)

        def objective(w: np.ndarray) -> float:
            port_ret = np.dot(w, adj_returns)
            port_var = np.dot(w.T, np.dot(cov_matrix, w))
            tc = np.dot(np.abs(w - (current_weights if current_weights is not None else w)), tc_penalty)
            return -(port_ret - self.risk_aversion * port_var - tc)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - self.leverage_limit}
        ]
        bounds = [(self.min_weight, self.max_weight)] * n

        # Initial guess: equal weight
        w0 = np.ones(n) / n

        try:
            result = minimize(
                objective,
                w0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
            if result.success:
                weights = np.clip(result.x, 0.0, 1.0)
                return weights / weights.sum()
        except Exception as exc:
            logger.warning(f"[Optimizer] Optimisation failed: {exc}")

        # Fallback: confidence-proportional weights from views
        return np.ones(n) / n

    # ------------------------------------------------------------------
    # Risk-Parity fallback
    # ------------------------------------------------------------------

    def risk_parity_weights(
        self, cov_matrix: np.ndarray, tickers: List[str]
    ) -> np.ndarray:
        """Equal risk contribution weights."""
        n = len(tickers)
        if n == 0:
            return np.array([])

        def objective(w):
            port_var = np.dot(w, np.dot(cov_matrix, w))
            marginal = np.dot(cov_matrix, w)
            risk_contrib = w * marginal / max(port_var, 1e-10)
            target = np.ones(n) / n
            return np.sum((risk_contrib - target) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.01, 0.40)] * n

        try:
            result = minimize(
                objective,
                np.ones(n) / n,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            )
            if result.success:
                w = np.clip(result.x, 0.0, 1.0)
                return w / w.sum()
        except Exception:
            pass

        return np.ones(n) / n


# =============================================================================
# REAL-TIME RISK MONITOR
# =============================================================================

class RealTimeRiskMonitor:
    """
    Continuously tracks portfolio risk metrics and fires alerts
    when thresholds are breached.

    Attach to AIPortfolioManager; called every bar.
    """

    DEFAULT_LIMITS = {
        "max_drawdown": -0.15,         # -15 %
        "daily_var_95": -0.03,         # -3 % in a day
        "max_position_weight": 0.40,   # single asset ≤ 40 %
        "max_portfolio_vol": 0.30,     # annualised vol ≤ 30 %
        "min_sharpe_rolling": -0.50,   # rolling 30-day Sharpe
        "max_correlation": 0.90,       # any pair correlation
    }

    def __init__(self, limits: Optional[Dict[str, float]] = None):
        self.limits = {**self.DEFAULT_LIMITS, **(limits or {})}
        self.alerts: List[Dict[str, Any]] = []
        self._return_history: List[float] = []
        self._value_history: List[float] = []

    def update(self, snapshot: PortfolioSnapshot) -> List[Dict[str, Any]]:
        """
        Process a new portfolio snapshot.
        Returns list of any new alerts triggered this bar.
        """
        new_alerts: List[Dict[str, Any]] = []

        self._return_history.append(snapshot.daily_return)
        self._value_history.append(snapshot.total_value)

        # ---- Drawdown alert ----
        if snapshot.drawdown < self.limits["max_drawdown"]:
            new_alerts.append(self._alert(
                "MAX_DRAWDOWN_BREACH",
                f"Drawdown {snapshot.drawdown*100:.1f}% below limit "
                f"{self.limits['max_drawdown']*100:.1f}%",
                severity="critical",
            ))

        # ---- VaR alert ----
        if len(self._return_history) >= 20:
            r = np.array(self._return_history[-20:])
            var_95 = float(np.percentile(r, 5))
            if var_95 < self.limits["daily_var_95"]:
                new_alerts.append(self._alert(
                    "VAR_BREACH",
                    f"20-day rolling VaR(95%) = {var_95*100:.2f}% below limit",
                    severity="warning",
                ))

        # ---- Concentration alert ----
        for ticker, w in snapshot.weights.items():
            if w > self.limits["max_position_weight"]:
                new_alerts.append(self._alert(
                    "CONCENTRATION",
                    f"{ticker} weight {w*100:.1f}% exceeds limit "
                    f"{self.limits['max_position_weight']*100:.1f}%",
                    severity="warning",
                ))

        # ---- Volatility alert ----
        if snapshot.portfolio_volatility > self.limits["max_portfolio_vol"]:
            new_alerts.append(self._alert(
                "HIGH_VOLATILITY",
                f"Portfolio vol {snapshot.portfolio_volatility*100:.1f}% above limit",
                severity="warning",
            ))

        # ---- Rolling Sharpe ----
        if len(self._return_history) >= 30:
            r30 = np.array(self._return_history[-30:])
            rol_sharpe = (r30.mean() / r30.std()) * np.sqrt(252) if r30.std() > 0 else 0
            if rol_sharpe < self.limits["min_sharpe_rolling"]:
                new_alerts.append(self._alert(
                    "LOW_SHARPE",
                    f"Rolling 30-day Sharpe = {rol_sharpe:.2f} below limit",
                    severity="info",
                ))

        self.alerts.extend(new_alerts)
        return new_alerts

    def compute_portfolio_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        portfolio_value: float,
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> float:
        """Parametric VaR for the full portfolio."""
        port_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
        port_std = np.sqrt(port_var) * np.sqrt(horizon_days / 252)
        z = stats.norm.ppf(1 - confidence)
        return float(z * port_std * portfolio_value)

    def component_var(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        tickers: List[str],
        portfolio_value: float,
    ) -> Dict[str, float]:
        """Each asset's contribution to portfolio VaR."""
        port_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
        if port_var <= 0:
            return {t: 0.0 for t in tickers}
        marginal = np.dot(cov_matrix, weights) / np.sqrt(port_var)
        component = weights * marginal
        component_value = component / component.sum() * portfolio_value
        return {tickers[i]: float(component_value[i]) for i in range(len(tickers))}

    def performance_attribution(
        self,
        snapshots: List[PortfolioSnapshot],
    ) -> pd.DataFrame:
        """Brinson attribution: how much each asset contributed to total return."""
        records = []
        for snap in snapshots:
            for ticker, pos in snap.positions.items():
                records.append({
                    "timestamp": snap.timestamp,
                    "ticker": ticker,
                    "weight": snap.weights.get(ticker, 0.0),
                    "unrealised_pnl": pos.unrealised_pnl,
                })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        return df.groupby("ticker")[["unrealised_pnl"]].sum().sort_values(
            "unrealised_pnl", ascending=False
        )

    @staticmethod
    def _alert(code: str, message: str, severity: str = "info") -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "severity": severity,
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# 💼 ENHANCEMENT 1: STRICT RISK CONTROLS
# =============================================================================
# Max drawdown limit, daily loss limit — matches FTMO / prop firm rules.
# Automatically halts trading when limits are breached.
# =============================================================================

class StrictRiskController:
    """
    Prop-firm-grade risk controls.

    Enforces hard limits that, when breached, either reduce position sizes
    or halt trading entirely — matching rules from FTMO, The5ers, etc.

    Parameters
    ----------
    max_drawdown_pct       : max total drawdown before trading halt (e.g. -0.10 = -10%)
    daily_loss_limit_pct   : max daily loss before trading halt (e.g. -0.05 = -5%)
    trailing_dd_pct        : trailing drawdown limit from equity peak
    max_consecutive_losses : halt after N consecutive losing trades
    cooldown_bars          : bars to wait after a limit breach before resuming
    """

    def __init__(
        self,
        max_drawdown_pct: float = -0.10,
        daily_loss_limit_pct: float = -0.05,
        trailing_dd_pct: float = -0.08,
        max_consecutive_losses: int = 10,
        cooldown_bars: int = 5,
    ):
        self.max_drawdown_pct = max_drawdown_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.trailing_dd_pct = trailing_dd_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.cooldown_bars = cooldown_bars

        # State
        self._peak_value: float = 0.0
        self._daily_start_value: float = 0.0
        self._consecutive_losses: int = 0
        self._trading_halted: bool = False
        self._halt_reason: str = ""
        self._cooldown_remaining: int = 0
        self._breaches: List[Dict[str, Any]] = []
        self._current_date: Optional[datetime] = None

    def check(
        self,
        current_value: float,
        initial_capital: float,
        timestamp: datetime,
        last_trade_pnl: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Check all risk limits. Called every bar.

        Returns
        -------
        Dict with 'trading_allowed', 'breaches', 'status'
        """
        # Reset daily tracking on new day
        if self._current_date is None or timestamp.date() != self._current_date:
            self._current_date = timestamp.date()
            self._daily_start_value = current_value

        # Track peak
        if current_value > self._peak_value:
            self._peak_value = current_value

        # Cooldown
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining == 0:
                self._trading_halted = False
                self._halt_reason = ""
                logger.info("[RiskControl] Cooldown expired, trading resumed")

        new_breaches = []

        # 1. Max total drawdown
        total_dd = (current_value - initial_capital) / initial_capital
        if total_dd < self.max_drawdown_pct:
            breach = self._create_breach(
                "MAX_DRAWDOWN", f"Total drawdown {total_dd:.1%} below limit {self.max_drawdown_pct:.1%}",
                "critical", timestamp
            )
            new_breaches.append(breach)
            self._halt_trading("Max drawdown breached")

        # 2. Daily loss limit
        daily_pnl = (current_value - self._daily_start_value) / max(self._daily_start_value, 1e-8)
        if daily_pnl < self.daily_loss_limit_pct:
            breach = self._create_breach(
                "DAILY_LOSS_LIMIT", f"Daily loss {daily_pnl:.1%} below limit {self.daily_loss_limit_pct:.1%}",
                "critical", timestamp
            )
            new_breaches.append(breach)
            self._halt_trading("Daily loss limit breached")

        # 3. Trailing drawdown
        trailing_dd = (current_value - self._peak_value) / max(self._peak_value, 1e-8)
        if trailing_dd < self.trailing_dd_pct:
            breach = self._create_breach(
                "TRAILING_DRAWDOWN", f"Trailing DD {trailing_dd:.1%} below limit {self.trailing_dd_pct:.1%}",
                "critical", timestamp
            )
            new_breaches.append(breach)
            self._halt_trading("Trailing drawdown breached")

        # 4. Consecutive losses
        if last_trade_pnl is not None:
            if last_trade_pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

            if self._consecutive_losses >= self.max_consecutive_losses:
                breach = self._create_breach(
                    "CONSECUTIVE_LOSSES",
                    f"{self._consecutive_losses} consecutive losses (limit={self.max_consecutive_losses})",
                    "warning", timestamp
                )
                new_breaches.append(breach)
                self._halt_trading("Consecutive loss limit")

        self._breaches.extend(new_breaches)

        # Position size scaling based on proximity to limits
        risk_scale = self._compute_risk_scale(total_dd, daily_pnl, trailing_dd)

        return {
            "trading_allowed": not self._trading_halted,
            "halt_reason": self._halt_reason,
            "risk_scale": risk_scale,
            "total_drawdown": total_dd,
            "daily_pnl": daily_pnl,
            "trailing_drawdown": trailing_dd,
            "consecutive_losses": self._consecutive_losses,
            "cooldown_remaining": self._cooldown_remaining,
            "new_breaches": new_breaches,
            "total_breaches": len(self._breaches),
        }

    def _halt_trading(self, reason: str) -> None:
        if not self._trading_halted:
            self._trading_halted = True
            self._halt_reason = reason
            self._cooldown_remaining = self.cooldown_bars
            logger.warning(f"[RiskControl] TRADING HALTED: {reason}")

    def _compute_risk_scale(
        self, total_dd: float, daily_pnl: float, trailing_dd: float
    ) -> float:
        """Scale down position sizes as we approach limits (0.0 to 1.0)."""
        # How close are we to each limit? (1.0 = safe, 0.0 = at limit)
        dd_ratio = max(0.0, 1.0 - abs(total_dd) / max(abs(self.max_drawdown_pct), 1e-8))
        daily_ratio = max(0.0, 1.0 - abs(daily_pnl) / max(abs(self.daily_loss_limit_pct), 1e-8))
        trail_ratio = max(0.0, 1.0 - abs(trailing_dd) / max(abs(self.trailing_dd_pct), 1e-8))

        # Use the most constrained ratio
        return float(min(dd_ratio, daily_ratio, trail_ratio))

    @staticmethod
    def _create_breach(code: str, message: str, severity: str, timestamp: datetime) -> Dict[str, Any]:
        return {
            "code": code,
            "message": message,
            "severity": severity,
            "timestamp": timestamp.isoformat(),
        }

    @property
    def breach_history(self) -> List[Dict[str, Any]]:
        return list(self._breaches)

    def reset(self) -> None:
        """Reset all state (for new session)."""
        self._peak_value = 0.0
        self._daily_start_value = 0.0
        self._consecutive_losses = 0
        self._trading_halted = False
        self._halt_reason = ""
        self._cooldown_remaining = 0
        self._breaches = []


# =============================================================================
# 💼 ENHANCEMENT 2: DIVERSIFICATION CONTROLLER
# =============================================================================
# Limit per-asset exposure and correlated-asset concentration.
# Prevents hidden risk from correlated positions.
# =============================================================================

class DiversificationController:
    """
    Enforces diversification constraints on portfolio weights.

    Rules
    -----
    1. Max single-asset weight (e.g. 25%)
    2. Max correlated-group weight (e.g. 40% for assets with ρ > 0.7)
    3. Minimum number of assets held
    4. Sector/asset-class concentration limits
    """

    def __init__(
        self,
        max_single_weight: float = 0.25,
        max_correlated_group_weight: float = 0.40,
        correlation_threshold: float = 0.70,
        min_positions: int = 3,
        max_positions: int = 20,
    ):
        self.max_single_weight = max_single_weight
        self.max_correlated_group_weight = max_correlated_group_weight
        self.correlation_threshold = correlation_threshold
        self.min_positions = min_positions
        self.max_positions = max_positions

    def enforce(
        self,
        target_weights: np.ndarray,
        tickers: List[str],
        correlation_matrix: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Adjust target weights to satisfy diversification constraints.

        Returns
        -------
        (adjusted_weights, adjustment_log)
        """
        n = len(tickers)
        w = target_weights.copy()
        adjustments = []

        # 1. Cap individual weights
        for i in range(n):
            if w[i] > self.max_single_weight:
                excess = w[i] - self.max_single_weight
                adjustments.append({
                    "type": "single_cap",
                    "ticker": tickers[i],
                    "original": float(target_weights[i]),
                    "capped_to": self.max_single_weight,
                    "excess_redistributed": float(excess),
                })
                w[i] = self.max_single_weight

        # 2. Correlation group limits
        if correlation_matrix is not None and correlation_matrix.shape == (n, n):
            groups = self._find_correlated_groups(correlation_matrix, tickers)
            for group_tickers, group_indices in groups:
                group_weight = sum(w[i] for i in group_indices)
                if group_weight > self.max_correlated_group_weight:
                    # Scale down proportionally within group
                    scale = self.max_correlated_group_weight / max(group_weight, 1e-8)
                    for idx in group_indices:
                        old_w = w[idx]
                        w[idx] *= scale
                        if old_w != w[idx]:
                            adjustments.append({
                                "type": "correlation_cap",
                                "ticker": tickers[idx],
                                "group": group_tickers,
                                "original": float(old_w),
                                "adjusted_to": float(w[idx]),
                                "group_correlation": f"> {self.correlation_threshold}",
                            })

        # 3. Re-normalize to sum to 1
        if w.sum() > 0:
            w = w / w.sum()

        return w, adjustments

    def _find_correlated_groups(
        self,
        corr_matrix: np.ndarray,
        tickers: List[str],
    ) -> List[Tuple[List[str], List[int]]]:
        """Find groups of assets with pairwise correlation above threshold."""
        n = len(tickers)
        visited = set()
        groups = []

        for i in range(n):
            if i in visited:
                continue
            group_idx = [i]
            for j in range(i + 1, n):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    group_idx.append(j)
                    visited.add(j)
            if len(group_idx) > 1:
                group_tickers = [tickers[k] for k in group_idx]
                groups.append((group_tickers, group_idx))
            visited.add(i)

        return groups

    def check_constraints(
        self,
        weights: Dict[str, float],
        correlation_matrix: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Check current portfolio against diversification rules."""
        violations = []
        n_positions = sum(1 for w in weights.values() if w > 0.01)

        for ticker, w in weights.items():
            if w > self.max_single_weight:
                violations.append({
                    "type": "over_concentrated",
                    "ticker": ticker,
                    "weight": w,
                    "limit": self.max_single_weight,
                })

        if n_positions < self.min_positions:
            violations.append({
                "type": "under_diversified",
                "n_positions": n_positions,
                "min_required": self.min_positions,
            })

        return {
            "n_positions": n_positions,
            "violations": violations,
            "is_compliant": len(violations) == 0,
            "max_weight": max(weights.values()) if weights else 0,
            "hhi": sum(w ** 2 for w in weights.values()),  # Herfindahl index
        }


# =============================================================================
# 💼 ENHANCEMENT 3: STRESS TESTING
# =============================================================================
# Simulate historical crashes (2008, COVID, etc.) on current portfolio.
# Shows survival probability in extreme market conditions.
# =============================================================================

class StressTester:
    """
    Historical stress test simulator.

    Applies historical crash scenarios to the current portfolio
    to estimate P&L impact under extreme conditions.
    """

    # Pre-defined crash scenarios (daily return shocks over period)
    SCENARIOS = {
        "2008_financial_crisis": {
            "description": "Global Financial Crisis (Sep-Nov 2008)",
            "duration_days": 60,
            "equity_shock": -0.45,    # S&P 500 dropped ~45%
            "crypto_shock": -0.80,    # (BTC didn't exist, but proxy)
            "commodity_shock": -0.35,
            "forex_shock": -0.15,
            "vol_multiplier": 3.5,
        },
        "covid_crash_2020": {
            "description": "COVID-19 Market Crash (Feb-Mar 2020)",
            "duration_days": 30,
            "equity_shock": -0.34,
            "crypto_shock": -0.50,
            "commodity_shock": -0.30,
            "forex_shock": -0.10,
            "vol_multiplier": 4.0,
        },
        "dot_com_2000": {
            "description": "Dot-Com Bubble Burst (2000-2001)",
            "duration_days": 180,
            "equity_shock": -0.50,
            "crypto_shock": -0.70,
            "commodity_shock": -0.10,
            "forex_shock": -0.05,
            "vol_multiplier": 2.0,
        },
        "flash_crash_2010": {
            "description": "Flash Crash (May 6, 2010)",
            "duration_days": 1,
            "equity_shock": -0.09,
            "crypto_shock": -0.15,
            "commodity_shock": -0.05,
            "forex_shock": -0.03,
            "vol_multiplier": 10.0,
        },
        "crypto_winter_2022": {
            "description": "Crypto Winter / LUNA-FTX Collapse (2022)",
            "duration_days": 90,
            "equity_shock": -0.15,
            "crypto_shock": -0.75,
            "commodity_shock": -0.05,
            "forex_shock": -0.08,
            "vol_multiplier": 3.0,
        },
        "rate_shock": {
            "description": "Hypothetical rapid rate hike (+300bps)",
            "duration_days": 30,
            "equity_shock": -0.20,
            "crypto_shock": -0.40,
            "commodity_shock": -0.15,
            "forex_shock": -0.12,
            "vol_multiplier": 2.5,
        },
    }

    def run_all(
        self,
        positions: Dict[str, float],  # ticker → current value
        total_value: float,
        asset_types: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run all stress scenarios.

        Parameters
        ----------
        positions    : {ticker: market_value} of current holdings
        total_value  : total portfolio value
        asset_types  : {ticker: 'equity'|'crypto'|'commodity'|'forex'}
        """
        if asset_types is None:
            asset_types = {t: self._infer_type(t) for t in positions}

        results = {}
        for name, scenario in self.SCENARIOS.items():
            results[name] = self._run_scenario(
                scenario, positions, total_value, asset_types
            )

        # Summary
        worst_scenario = min(results.items(), key=lambda x: x[1]["portfolio_impact_pct"])
        survives_all = all(
            r["portfolio_survives"] for r in results.values()
        )

        return {
            "scenarios": results,
            "worst_scenario": worst_scenario[0],
            "worst_impact_pct": worst_scenario[1]["portfolio_impact_pct"],
            "survives_all": survives_all,
            "survival_rate": sum(
                1 for r in results.values() if r["portfolio_survives"]
            ) / len(results),
        }

    def _run_scenario(
        self,
        scenario: Dict[str, Any],
        positions: Dict[str, float],
        total_value: float,
        asset_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Apply a single crash scenario to the portfolio."""
        impact_by_asset = {}
        total_impact = 0.0

        for ticker, value in positions.items():
            atype = asset_types.get(ticker, "equity")
            shock_key = f"{atype}_shock"
            shock = scenario.get(shock_key, scenario.get("equity_shock", -0.30))

            asset_impact = value * shock
            impact_by_asset[ticker] = {
                "value_before": value,
                "shock_pct": shock,
                "impact": float(asset_impact),
                "value_after": float(value + asset_impact),
            }
            total_impact += asset_impact

        portfolio_impact_pct = total_impact / max(total_value, 1e-8)

        return {
            "description": scenario["description"],
            "duration_days": scenario["duration_days"],
            "vol_multiplier": scenario["vol_multiplier"],
            "portfolio_impact_pct": float(portfolio_impact_pct),
            "portfolio_impact_value": float(total_impact),
            "value_after_stress": float(total_value + total_impact),
            "portfolio_survives": (total_value + total_impact) > 0,
            "per_asset": impact_by_asset,
        }

    @staticmethod
    def _infer_type(ticker: str) -> str:
        if ticker in ("BTCUSD", "ETHUSD", "SOLUSD", "BNBUSD", "XRPUSD"):
            return "crypto"
        if "=F" in ticker:
            return "commodity"
        if "USD" in ticker and len(ticker) <= 7:
            return "forex"
        return "equity"


# =============================================================================
# 💼 ENHANCEMENT 4: LIQUIDITY MANAGER
# =============================================================================
# Volume-based position sizing — avoids unrealistic trades.
# =============================================================================

class LiquidityManager:
    """
    Adjusts position sizes based on market liquidity.

    Prevents:
    - Unrealistically large positions relative to volume
    - Orders that would move the market
    - Illiquid asset over-allocation
    """

    def __init__(
        self,
        max_volume_pct: float = 0.02,     # max 2% of daily volume per trade
        min_dollar_volume: float = 50_000, # minimum daily $ volume to trade
        impact_threshold: float = 0.05,     # warn if order > 5% of volume
    ):
        self.max_volume_pct = max_volume_pct
        self.min_dollar_volume = min_dollar_volume
        self.impact_threshold = impact_threshold

    def adjust_position_size(
        self,
        target_value: float,
        ticker: str,
        price: float,
        avg_daily_volume: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Adjust position size based on liquidity.

        Returns
        -------
        (adjusted_value, liquidity_info)
        """
        dollar_volume = avg_daily_volume * price
        max_order = dollar_volume * self.max_volume_pct

        adjusted = target_value
        flags = []

        # Check minimum liquidity
        if dollar_volume < self.min_dollar_volume:
            adjusted = 0.0
            flags.append(f"Illiquid: ${dollar_volume:,.0f} daily vol < ${self.min_dollar_volume:,.0f} min")

        # Cap to max volume participation
        if adjusted > max_order and max_order > 0:
            flags.append(
                f"Capped: ${target_value:,.0f} → ${max_order:,.0f} "
                f"({self.max_volume_pct:.0%} of daily volume)"
            )
            adjusted = max_order

        # Market impact warning
        participation_rate = target_value / max(dollar_volume, 1e-8)
        if participation_rate > self.impact_threshold:
            flags.append(
                f"High impact: order is {participation_rate:.1%} of daily volume"
            )

        return adjusted, {
            "target_value": target_value,
            "adjusted_value": adjusted,
            "dollar_volume": dollar_volume,
            "participation_rate": participation_rate,
            "max_order_value": max_order,
            "flags": flags,
            "is_liquid": dollar_volume >= self.min_dollar_volume,
        }

    def screen_universe(
        self,
        tickers: List[str],
        volumes: Dict[str, float],
        prices: Dict[str, float],
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
        """
        Screen tickers for minimum liquidity.

        Returns
        -------
        (liquid_tickers, screening_report)
        """
        liquid = []
        report = {}

        for ticker in tickers:
            vol = volumes.get(ticker, 0)
            price = prices.get(ticker, 0)
            dollar_vol = vol * price

            info = {
                "volume": vol,
                "price": price,
                "dollar_volume": dollar_vol,
                "passes": dollar_vol >= self.min_dollar_volume,
            }
            report[ticker] = info
            if info["passes"]:
                liquid.append(ticker)

        return liquid, report


# =============================================================================
# 💼 ENHANCEMENT 5: PERFORMANCE ATTRIBUTOR
# =============================================================================
# Break P&L into asset, strategy, and model contributions.
# Shows what actually works in the portfolio.
# =============================================================================

class PerformanceAttributor:
    """
    Multi-dimensional performance attribution.

    Decomposes portfolio returns into:
    1. Asset attribution — which tickers contributed most
    2. Strategy attribution — which signals (buy/sell/regime) worked
    3. Model attribution — which models drove the best trades
    4. Timing attribution — when did trades perform best
    """

    def attribute(
        self,
        snapshots: List[PortfolioSnapshot],
        trades: Optional[List] = None,  # List of trade objects with model info
    ) -> Dict[str, Any]:
        """
        Run full attribution analysis.

        Parameters
        ----------
        snapshots : list of PortfolioSnapshot objects
        trades    : optional list of trade-like objects with
                    .decision.model_predictions, .decision.regime, etc.
        """
        results = {}

        # 1. Asset attribution
        results["by_asset"] = self._asset_attribution(snapshots)

        # 2. Time-based attribution
        results["by_time"] = self._time_attribution(snapshots)

        # 3. Trade-level attribution (if trades provided)
        if trades:
            results["by_regime"] = self._regime_attribution(trades)
            results["by_model"] = self._model_attribution(trades)
            results["by_exit_reason"] = self._exit_reason_attribution(trades)

        # 4. Risk-adjusted attribution
        results["risk_adjusted"] = self._risk_adjusted_attribution(snapshots)

        return results

    def _asset_attribution(
        self, snapshots: List[PortfolioSnapshot]
    ) -> Dict[str, Any]:
        """Attribute returns to individual assets."""
        if len(snapshots) < 2:
            return {}

        asset_pnl: Dict[str, float] = defaultdict(float)
        asset_weight_sum: Dict[str, float] = defaultdict(float)
        asset_count: Dict[str, int] = defaultdict(int)

        for i in range(1, len(snapshots)):
            snap = snapshots[i]
            for ticker, pos in snap.positions.items():
                asset_pnl[ticker] += pos.unrealised_pnl
                asset_weight_sum[ticker] += snap.weights.get(ticker, 0)
                asset_count[ticker] += 1

        total_pnl = sum(asset_pnl.values())
        attribution = {}
        for ticker in asset_pnl:
            attribution[ticker] = {
                "total_pnl": float(asset_pnl[ticker]),
                "contribution_pct": float(
                    asset_pnl[ticker] / max(abs(total_pnl), 1e-8) * 100
                ),
                "avg_weight": float(
                    asset_weight_sum[ticker] / max(asset_count[ticker], 1)
                ),
                "periods_held": asset_count[ticker],
            }

        sorted_attr = dict(
            sorted(attribution.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
        )

        return {
            "by_asset": sorted_attr,
            "total_pnl": float(total_pnl),
            "top_contributor": list(sorted_attr.keys())[0] if sorted_attr else "N/A",
            "bottom_contributor": list(sorted_attr.keys())[-1] if sorted_attr else "N/A",
        }

    def _time_attribution(
        self, snapshots: List[PortfolioSnapshot]
    ) -> Dict[str, Any]:
        """Attribute returns by time period (monthly)."""
        if len(snapshots) < 2:
            return {}

        monthly: Dict[str, float] = defaultdict(float)
        for i in range(1, len(snapshots)):
            month_key = snapshots[i].timestamp.strftime("%Y-%m")
            monthly[month_key] += snapshots[i].daily_return

        best_month = max(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0)
        worst_month = min(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0)

        return {
            "monthly_returns": {k: float(v) for k, v in monthly.items()},
            "best_month": {"period": best_month[0], "return": float(best_month[1])},
            "worst_month": {"period": worst_month[0], "return": float(worst_month[1])},
            "positive_months": sum(1 for v in monthly.values() if v > 0),
            "negative_months": sum(1 for v in monthly.values() if v <= 0),
        }

    def _regime_attribution(self, trades: List) -> Dict[str, Any]:
        """Attribute returns by market regime."""
        regime_pnl: Dict[str, List[float]] = defaultdict(list)

        for t in trades:
            regime = getattr(getattr(t, 'decision', None), 'regime', 'unknown')
            pnl = getattr(t, 'realized_pnl', 0.0)
            regime_pnl[regime].append(pnl)

        result = {}
        for regime, pnls in regime_pnl.items():
            arr = np.array(pnls)
            result[regime] = {
                "n_trades": len(pnls),
                "total_pnl": float(arr.sum()),
                "avg_pnl": float(arr.mean()),
                "win_rate": float(np.mean(arr > 0)),
            }

        return result

    def _model_attribution(self, trades: List) -> Dict[str, Any]:
        """Attribute returns by which model drove the decision."""
        model_pnl: Dict[str, List[float]] = defaultdict(list)

        for t in trades:
            preds = getattr(getattr(t, 'decision', None), 'model_predictions', {})
            pnl = getattr(t, 'realized_pnl', 0.0)

            if preds:
                # Attribute to model with highest absolute prediction
                best_model = max(preds.items(), key=lambda x: abs(x[1]))[0]
                model_pnl[best_model].append(pnl)

        result = {}
        for model, pnls in model_pnl.items():
            arr = np.array(pnls)
            result[model] = {
                "n_trades": len(pnls),
                "total_pnl": float(arr.sum()),
                "avg_pnl": float(arr.mean()),
                "win_rate": float(np.mean(arr > 0)),
            }

        return result

    def _exit_reason_attribution(self, trades: List) -> Dict[str, Any]:
        """Attribute returns by exit reason."""
        reason_pnl: Dict[str, List[float]] = defaultdict(list)

        for t in trades:
            reason = getattr(t, 'exit_reason', 'unknown')
            pnl = getattr(t, 'realized_pnl', 0.0)
            reason_pnl[reason].append(pnl)

        result = {}
        for reason, pnls in reason_pnl.items():
            arr = np.array(pnls)
            result[reason] = {
                "n_trades": len(pnls),
                "total_pnl": float(arr.sum()),
                "avg_pnl": float(arr.mean()),
            }

        return result

    def _risk_adjusted_attribution(
        self, snapshots: List[PortfolioSnapshot]
    ) -> Dict[str, Any]:
        """Risk-adjusted return attribution per asset."""
        if len(snapshots) < 10:
            return {}

        asset_returns: Dict[str, List[float]] = defaultdict(list)

        for i in range(1, len(snapshots)):
            for ticker in snapshots[i].weights:
                w = snapshots[i].weights.get(ticker, 0)
                r = snapshots[i].daily_return
                asset_returns[ticker].append(w * r)

        result = {}
        for ticker, rets in asset_returns.items():
            arr = np.array(rets)
            if len(arr) > 5 and arr.std() > 0:
                result[ticker] = {
                    "contribution_sharpe": float(
                        arr.mean() / arr.std() * np.sqrt(252)
                    ),
                    "avg_contribution": float(arr.mean()),
                    "vol_contribution": float(arr.std() * np.sqrt(252)),
                }

        return result


# =============================================================================
# 💼 ENHANCEMENT 6: ENHANCED REAL-TIME RISK DASHBOARD
# =============================================================================
# Live drawdown tracking, VaR, and comprehensive risk monitoring.
# Extends existing RealTimeRiskMonitor with dashboard-ready output.
# =============================================================================

class EnhancedRiskDashboard:
    """
    Dashboard-ready risk monitoring layer.

    Wraps RealTimeRiskMonitor and adds:
    1. Live drawdown waterfall
    2. Rolling VaR with confidence bands
    3. Concentration heatmap data
    4. Risk budget utilization
    5. Alert timeline
    """

    def __init__(
        self,
        risk_monitor: RealTimeRiskMonitor,
        var_window: int = 20,
        vol_window: int = 60,
    ):
        self.risk_monitor = risk_monitor
        self.var_window = var_window
        self.vol_window = vol_window

        # Historical tracking
        self._var_history: List[Dict[str, float]] = []
        self._drawdown_history: List[Dict[str, float]] = []
        self._vol_history: List[Dict[str, float]] = []
        self._alert_timeline: List[Dict[str, Any]] = []

    def update(
        self,
        snapshot: PortfolioSnapshot,
        weights: Optional[np.ndarray] = None,
        cov_matrix: Optional[np.ndarray] = None,
        tickers: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Process snapshot and return dashboard-ready data.

        Returns
        -------
        Dict with all risk metrics formatted for UI display
        """
        # Run base monitor
        new_alerts = self.risk_monitor.update(snapshot)
        for alert in new_alerts:
            self._alert_timeline.append(alert)

        # Track VaR
        if len(self.risk_monitor._return_history) >= self.var_window:
            r = np.array(self.risk_monitor._return_history[-self.var_window:])
            var_95 = float(np.percentile(r, 5))
            var_99 = float(np.percentile(r, 1))
            cvar_95 = float(np.mean(r[r <= var_95])) if len(r[r <= var_95]) > 0 else var_95

            self._var_history.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "var_95": var_95,
                "var_99": var_99,
                "cvar_95": cvar_95,
            })

        # Track drawdown
        self._drawdown_history.append({
            "timestamp": snapshot.timestamp.isoformat(),
            "drawdown": snapshot.drawdown,
            "value": snapshot.total_value,
            "peak": snapshot.peak_value,
        })

        # Track volatility
        if len(self.risk_monitor._return_history) >= self.vol_window:
            r = np.array(self.risk_monitor._return_history[-self.vol_window:])
            realized_vol = float(r.std() * np.sqrt(252))
            self._vol_history.append({
                "timestamp": snapshot.timestamp.isoformat(),
                "realized_vol": realized_vol,
            })

        # Component VaR
        component_var = {}
        if weights is not None and cov_matrix is not None and tickers:
            component_var = self.risk_monitor.component_var(
                weights, cov_matrix, tickers, snapshot.total_value
            )

        # Build dashboard output
        return {
            "live_metrics": {
                "total_value": snapshot.total_value,
                "drawdown_pct": snapshot.drawdown * 100,
                "daily_return_pct": snapshot.daily_return * 100,
                "portfolio_vol_pct": snapshot.portfolio_volatility * 100,
                "sharpe_ratio": snapshot.sharpe_ratio,
                "var_95_pct": snapshot.var_95 * 100,
                "n_positions": len(snapshot.positions),
            },
            "risk_budget": {
                "drawdown_used_pct": float(
                    abs(snapshot.drawdown)
                    / abs(self.risk_monitor.limits.get("max_drawdown", -0.15))
                    * 100
                ),
                "vol_used_pct": float(
                    snapshot.portfolio_volatility
                    / self.risk_monitor.limits.get("max_portfolio_vol", 0.30)
                    * 100
                ),
                "max_weight_used_pct": float(
                    max(snapshot.weights.values(), default=0)
                    / self.risk_monitor.limits.get("max_position_weight", 0.40)
                    * 100
                ),
            },
            "component_var": component_var,
            "recent_alerts": self._alert_timeline[-10:],
            "var_history_count": len(self._var_history),
            "drawdown_history_count": len(self._drawdown_history),
        }

    def get_drawdown_waterfall(self, last_n: int = 100) -> List[Dict[str, float]]:
        """Return recent drawdown history for waterfall chart."""
        return self._drawdown_history[-last_n:]

    def get_var_timeseries(self, last_n: int = 100) -> List[Dict[str, float]]:
        """Return recent VaR history for line chart."""
        return self._var_history[-last_n:]

    def get_vol_timeseries(self, last_n: int = 100) -> List[Dict[str, float]]:
        """Return recent volatility history for line chart."""
        return self._vol_history[-last_n:]

    def get_alert_summary(self) -> Dict[str, int]:
        """Count alerts by severity."""
        summary: Dict[str, int] = defaultdict(int)
        for alert in self._alert_timeline:
            summary[alert.get("severity", "info")] += 1
        return dict(summary)


# =============================================================================
# AI PORTFOLIO MANAGER — top-level orchestrator
# =============================================================================

class AIPortfolioManager:
    """
    Manages a multi-asset AI-driven portfolio end-to-end.

    Responsibilities
    ----------------
    1. Receives AssetViews from AI models each bar
    2. Runs Black-Litterman + mean-variance optimisation
    3. Generates RebalanceActions for the execution layer
    4. Tracks positions, P&L, and risk metrics in real time
    5. Maintains a full history of PortfolioSnapshots

    Usage in tradingprofessional.py
    --------------------------------
    from ai_portfolio_system import AIPortfolioManager, AssetView

    mgr = AIPortfolioManager(
        tickers=['BTCUSD', 'ETHUSD', 'GC=F'],
        initial_capital=500_000,
    )
    # Each bar / rebalancing cycle:
    views = [
        AssetView('BTCUSD', predicted_return=0.25, confidence=72, ...),
        ...
    ]
    actions = mgr.rebalance(views, current_prices, historical_returns_df)
    snapshot = mgr.latest_snapshot
    """

    def __init__(
        self,
        tickers: List[str],
        initial_capital: float = 100_000.0,
        rebalance_frequency: str = "weekly",   # 'daily' | 'weekly' | 'monthly'
        risk_limits: Optional[Dict[str, float]] = None,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        regime_lookback: int = 252,
        # ---- 💼 Enhancement params ----
        strict_risk_kwargs: Optional[Dict[str, float]] = None,
        diversification_kwargs: Optional[Dict[str, float]] = None,
        liquidity_kwargs: Optional[Dict[str, float]] = None,
        enable_strict_risk: bool = True,
        enable_diversification: bool = True,
        enable_stress_testing: bool = True,
        enable_liquidity: bool = True,
        enable_attribution: bool = True,
        enable_risk_dashboard: bool = True,
    ):
        self.tickers = tickers
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

        # Sub-components
        self.optimizer = PortfolioOptimizer(**(optimizer_kwargs or {}))
        self.risk_monitor = RealTimeRiskMonitor(limits=risk_limits)
        self.regime_detectors: Dict[str, RegimeDetector] = {
            t: RegimeDetector(lookback=regime_lookback) for t in tickers
        }

        # State
        self.cash = initial_capital
        self.positions: Dict[str, PortfolioPosition] = {}
        self.snapshots: List[PortfolioSnapshot] = []
        self.rebalance_history: List[Dict[str, Any]] = []
        self._peak_value = initial_capital
        self._last_rebalance: Optional[datetime] = None
        self._return_history: List[float] = []

        # ---- 💼 Enhancement modules ----
        self.strict_risk = StrictRiskController(
            **(strict_risk_kwargs or {})
        ) if enable_strict_risk else None

        self.diversification = DiversificationController(
            **(diversification_kwargs or {})
        ) if enable_diversification else None

        self.stress_tester = StressTester() if enable_stress_testing else None

        self.liquidity_manager = LiquidityManager(
            **(liquidity_kwargs or {})
        ) if enable_liquidity else None

        self.attributor = PerformanceAttributor() if enable_attribution else None

        self.risk_dashboard = EnhancedRiskDashboard(
            risk_monitor=self.risk_monitor,
        ) if enable_risk_dashboard else None

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def update_prices(
        self,
        current_prices: Dict[str, float],
        timestamp: datetime,
    ) -> PortfolioSnapshot:
        """
        Mark positions to market; compute snapshot; fire risk alerts.
        Call every bar even when not rebalancing.
        """
        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos.entry_price)
            pos.current_price = price
            pos.unrealised_pnl = (price - pos.entry_price) * pos.shares

        invested = sum(
            p.shares * p.current_price for p in self.positions.values()
        )
        total_value = self.cash + invested

        # Weights
        weights = {}
        for t, pos in self.positions.items():
            val = pos.shares * pos.current_price
            weights[t] = val / max(total_value, 1e-8)

        # Returns
        prev_value = self.snapshots[-1].total_value if self.snapshots else self.initial_capital
        daily_ret = (total_value - prev_value) / max(prev_value, 1e-8)
        total_ret = (total_value - self.initial_capital) / self.initial_capital

        self._peak_value = max(self._peak_value, total_value)
        drawdown = (total_value - self._peak_value) / max(self._peak_value, 1e-8)

        self._return_history.append(daily_ret)
        r = np.array(self._return_history[-60:])
        vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0
        sharpe = float(
            (r.mean() / r.std()) * np.sqrt(252)
        ) if (len(r) > 5 and r.std() > 0) else 0.0
        var_95 = float(np.percentile(r, 5)) if len(r) >= 10 else 0.0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash=self.cash,
            invested_value=invested,
            positions=dict(self.positions),
            weights=weights,
            target_weights={},           # filled during rebalance
            daily_return=daily_ret,
            total_return=total_ret,
            drawdown=drawdown,
            peak_value=self._peak_value,
            portfolio_volatility=vol,
            sharpe_ratio=sharpe,
            var_95=var_95,
        )

        self.snapshots.append(snapshot)
        alerts = self.risk_monitor.update(snapshot)
        if alerts:
            logger.warning(
                f"[Portfolio] {len(alerts)} risk alert(s): "
                + " | ".join(a["code"] for a in alerts)
            )

        # 💼 Enhancement 1: Strict Risk Controls check
        if self.strict_risk is not None:
            risk_check = self.strict_risk.check(
                total_value, self.initial_capital, timestamp
            )
            if not risk_check["trading_allowed"]:
                logger.warning(
                    f"[StrictRisk] Trading HALTED: {risk_check['halt_reason']} | "
                    f"DD={risk_check['total_drawdown']:.1%}"
                )

        # 💼 Enhancement 6: Enhanced Risk Dashboard update
        if self.risk_dashboard is not None:
            self.risk_dashboard.update(snapshot)

        return snapshot

    def rebalance(
        self,
        asset_views: List[AssetView],
        current_prices: Dict[str, float],
        historical_returns: pd.DataFrame,
        timestamp: Optional[datetime] = None,
        force: bool = False,
        volumes: Optional[Dict[str, float]] = None,  # 💼 Enhancement 4: daily volumes
    ) -> List[RebalanceAction]:
        """
        Run full rebalancing cycle with 💼 enhancement integration.

        Steps
        -----
        1. Fit regime detectors
        2. 💼 Check strict risk controls (halt if breached)
        3. Black-Litterman blending of views + history
        4. Mean-variance optimisation → target weights
        5. 💼 Apply diversification constraints
        6. 💼 Apply liquidity constraints
        7. Generate buy/sell instructions
        8. Execute (update positions and cash)

        Returns list of RebalanceAction for display in the Streamlit UI.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        if not force and not self._should_rebalance(timestamp):
            return []

        # 💼 Enhancement 1: Strict Risk Controls — block rebalance if halted
        if self.strict_risk is not None:
            total_value = self.cash + sum(
                pos.shares * current_prices.get(t, pos.current_price)
                for t, pos in self.positions.items()
            )
            risk_status = self.strict_risk.check(
                total_value, self.initial_capital, timestamp
            )
            if not risk_status["trading_allowed"]:
                logger.warning(
                    f"[Portfolio] Rebalance BLOCKED by strict risk: {risk_status['halt_reason']}"
                )
                return []

        logger.info(f"[Portfolio] Rebalancing {len(asset_views)} assets at {timestamp}")

        # ---- 1. Regime per asset ----
        regimes: Dict[str, str] = {}
        for view in asset_views:
            ticker = view.ticker
            if ticker in historical_returns.columns:
                r = historical_returns[ticker].dropna()
                det = self.regime_detectors.get(ticker, RegimeDetector())
                det.fit(r)
                regimes[ticker] = det.predict(r)
            else:
                regimes[ticker] = view.regime

        portfolio_regime = self._dominant_regime(regimes)
        logger.info(f"[Portfolio] Portfolio regime: {portfolio_regime}")

        # Active tickers (those with both a view and a current price)
        active = [v.ticker for v in asset_views if v.ticker in current_prices]
        if not active:
            return []

        # 💼 Enhancement 4: Liquidity screening
        if self.liquidity_manager is not None and volumes:
            liquid_tickers, screening = self.liquidity_manager.screen_universe(
                active, volumes, current_prices
            )
            illiquid = set(active) - set(liquid_tickers)
            if illiquid:
                logger.info(f"[Liquidity] Excluded illiquid: {illiquid}")
            active = liquid_tickers
            if not active:
                return []

        active_views = [v for v in asset_views if v.ticker in active]

        # ---- 2. Black-Litterman ----
        bl_returns, bl_cov = self.optimizer.black_litterman_views(
            active, historical_returns, active_views
        )

        # ---- 3. Optimise ----
        n = len(active)
        current_total = self.cash + sum(
            self.positions.get(t, PortfolioPosition(t, 0, 0, datetime.utcnow())).shares
            * current_prices.get(t, 0)
            for t in active
        )

        current_weights_arr = np.array([
            (
                self.positions[t].shares * current_prices[t] / max(current_total, 1e-8)
                if t in self.positions else 0.0
            )
            for t in active
        ])

        target_weights = self.optimizer.optimize(
            active, bl_returns, bl_cov,
            current_weights=current_weights_arr,
            regime=portfolio_regime,
        )

        # 💼 Enhancement 2: Apply diversification constraints
        if self.diversification is not None:
            corr_matrix = None
            if not historical_returns.empty:
                try:
                    valid_cols = [c for c in active if c in historical_returns.columns]
                    if valid_cols:
                        corr_matrix = historical_returns[valid_cols].corr().values
                except Exception:
                    pass

            target_weights, div_adjustments = self.diversification.enforce(
                target_weights, active, corr_matrix
            )
            if div_adjustments:
                logger.info(
                    f"[Diversification] Applied {len(div_adjustments)} weight adjustments"
                )

        # 💼 Enhancement 1: Scale positions by risk proximity
        if self.strict_risk is not None:
            risk_scale = risk_status.get("risk_scale", 1.0)
            if risk_scale < 1.0:
                target_weights *= risk_scale
                # Increase cash allocation
                cash_weight = 1.0 - target_weights.sum()
                logger.info(
                    f"[StrictRisk] Scaled weights by {risk_scale:.2f} | "
                    f"Cash allocation: {cash_weight:.1%}"
                )

        # ---- 4. Generate actions ----
        actions: List[RebalanceAction] = []
        for i, ticker in enumerate(active):
            cw = float(current_weights_arr[i])
            tw = float(target_weights[i])
            delta = tw - cw

            if abs(delta) < 0.005:     # ignore tiny rebalances
                continue

            act = "buy" if delta > 0 else "sell"
            trade_val = abs(delta) * current_total

            # 💼 Enhancement 4: Liquidity-adjusted trade value
            if self.liquidity_manager is not None and volumes:
                vol = volumes.get(ticker, 0)
                price = current_prices.get(ticker, 0)
                adj_val, liq_info = self.liquidity_manager.adjust_position_size(
                    trade_val, ticker, price, vol
                )
                if adj_val < trade_val:
                    logger.info(
                        f"[Liquidity] {ticker}: ${trade_val:,.0f} → ${adj_val:,.0f}"
                    )
                trade_val = adj_val

            view = next((v for v in active_views if v.ticker == ticker), None)
            reason = (
                f"BL-predicted return={view.predicted_return*100:+.1f}% "
                f"| confidence={view.confidence:.0f}% "
                f"| regime={regimes.get(ticker, '?')}"
                if view else "Rebalance"
            )

            actions.append(RebalanceAction(
                ticker=ticker,
                action=act,
                current_weight=cw,
                target_weight=tw,
                delta_weight=delta,
                estimated_trade_value=trade_val,
                reason=reason,
            ))

        # ---- 5. Execute ----
        self._execute_rebalance(actions, current_prices, target_weights, active, timestamp)

        # Store target weights in latest snapshot
        if self.snapshots:
            self.snapshots[-1].target_weights = {
                active[i]: float(target_weights[i]) for i in range(n)
            }

        self._last_rebalance = timestamp
        self.rebalance_history.append({
            "timestamp": str(timestamp),
            "n_actions": len(actions),
            "regime": portfolio_regime,
            "target_weights": {active[i]: float(target_weights[i]) for i in range(n)},
        })

        logger.info(f"[Portfolio] Rebalanced: {len(actions)} trade instructions generated")
        return actions

    # ------------------------------------------------------------------
    # Portfolio analytics
    # ------------------------------------------------------------------

    def performance_summary(self) -> Dict[str, Any]:
        """Return a dict suitable for display in Streamlit metrics."""
        if not self.snapshots:
            return {}

        latest = self.snapshots[-1]
        r = np.array(self._return_history)
        ann_ret = float(r.mean() * 252) if len(r) > 0 else 0.0
        ann_vol = float(r.std() * np.sqrt(252)) if len(r) > 1 else 0.0
        sharpe = ann_ret / max(ann_vol, 1e-8) - 0.02 / max(ann_vol, 1e-8)

        rolling_max = max(s.total_value for s in self.snapshots)
        max_dd = min(
            (s.total_value - rolling_max) / rolling_max
            for s in self.snapshots
        ) if self.snapshots else 0.0

        wins = [r for r in self._return_history if r > 0]
        losses = [r for r in self._return_history if r <= 0]

        return {
            "total_value": latest.total_value,
            "total_return_pct": latest.total_return * 100,
            "annualized_return_pct": ann_ret * 100,
            "annualized_volatility_pct": ann_vol * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd * 100,
            "current_drawdown_pct": latest.drawdown * 100,
            "n_positions": len(latest.positions),
            "cash_pct": latest.cash / max(latest.total_value, 1e-8) * 100,
            "win_rate_daily": len(wins) / max(len(self._return_history), 1) * 100,
            "avg_daily_return_pct": float(np.mean(self._return_history)) * 100 if self._return_history else 0.0,
            "var_95_daily_pct": latest.var_95 * 100,
            "portfolio_volatility_pct": latest.portfolio_volatility * 100,
            "n_rebalances": len(self.rebalance_history),
            "n_alerts": len(self.risk_monitor.alerts),
        }

    def value_series(self) -> pd.Series:
        if not self.snapshots:
            return pd.Series(dtype=float)
        idx = [s.timestamp for s in self.snapshots]
        vals = [s.total_value for s in self.snapshots]
        return pd.Series(vals, index=pd.to_datetime(idx))

    def drawdown_series(self) -> pd.Series:
        if not self.snapshots:
            return pd.Series(dtype=float)
        idx = [s.timestamp for s in self.snapshots]
        dd = [s.drawdown for s in self.snapshots]
        return pd.Series(dd, index=pd.to_datetime(idx))

    def weights_history(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame()
        records = [
            {"timestamp": s.timestamp, **s.weights}
            for s in self.snapshots
        ]
        return pd.DataFrame(records).set_index("timestamp").fillna(0.0)

    @property
    def latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        return self.snapshots[-1] if self.snapshots else None

    # ------------------------------------------------------------------
    # 💼 Enhancement convenience methods
    # ------------------------------------------------------------------

    def run_stress_test(
        self, current_prices: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Run stress testing on current portfolio (Enhancement 3).

        Returns dict with per-scenario impact and survival analysis.
        """
        if self.stress_tester is None:
            return {"error": "Stress testing not enabled"}

        positions_value = {
            t: pos.shares * current_prices.get(t, pos.current_price)
            for t, pos in self.positions.items()
        }
        total_value = self.cash + sum(positions_value.values())

        return self.stress_tester.run_all(positions_value, total_value)

    def run_attribution(
        self, trades: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run performance attribution (Enhancement 5).

        Decomposes returns by asset, time, regime, model, and exit reason.
        """
        if self.attributor is None:
            return {"error": "Performance attribution not enabled"}

        return self.attributor.attribute(self.snapshots, trades)

    def check_diversification(self) -> Dict[str, Any]:
        """
        Check current diversification compliance (Enhancement 2).
        """
        if self.diversification is None:
            return {"error": "Diversification controller not enabled"}

        if not self.snapshots:
            return {"error": "No snapshots available"}

        return self.diversification.check_constraints(
            self.snapshots[-1].weights
        )

    def get_risk_dashboard_data(self) -> Dict[str, Any]:
        """
        Get comprehensive risk dashboard data (Enhancement 6).
        """
        if self.risk_dashboard is None:
            return {"error": "Risk dashboard not enabled"}

        return {
            "drawdown_waterfall": self.risk_dashboard.get_drawdown_waterfall(),
            "var_timeseries": self.risk_dashboard.get_var_timeseries(),
            "vol_timeseries": self.risk_dashboard.get_vol_timeseries(),
            "alert_summary": self.risk_dashboard.get_alert_summary(),
        }

    def get_risk_control_status(self) -> Dict[str, Any]:
        """
        Get current strict risk control status (Enhancement 1).
        """
        if self.strict_risk is None:
            return {"error": "Strict risk controls not enabled"}

        return {
            "trading_halted": self.strict_risk._trading_halted,
            "halt_reason": self.strict_risk._halt_reason,
            "consecutive_losses": self.strict_risk._consecutive_losses,
            "cooldown_remaining": self.strict_risk._cooldown_remaining,
            "breach_count": len(self.strict_risk._breaches),
            "recent_breaches": self.strict_risk._breaches[-5:],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_rebalance(self, timestamp: datetime) -> bool:
        if self._last_rebalance is None:
            return True
        delta = timestamp - self._last_rebalance
        freq_map = {
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30),
        }
        return delta >= freq_map.get(self.rebalance_frequency, timedelta(days=7))

    def _execute_rebalance(
        self,
        actions: List[RebalanceAction],
        prices: Dict[str, float],
        target_weights: np.ndarray,
        active: List[str],
        timestamp: datetime,
    ) -> None:
        """Apply the rebalance instructions to positions and cash."""
        total_value = self.cash + sum(
            self.positions.get(t, PortfolioPosition(t, 0, 0, timestamp)).shares
            * prices.get(t, 0)
            for t in active
        )

        target_map = {active[i]: float(target_weights[i]) for i in range(len(active))}

        for action in actions:
            ticker = action.ticker
            price = prices.get(ticker, 0.0)
            if price <= 0:
                continue

            target_val = target_map.get(ticker, 0.0) * total_value
            current_shares = self.positions.get(ticker, PortfolioPosition(ticker, 0, 0, timestamp)).shares
            current_val = current_shares * price

            delta_val = target_val - current_val
            delta_shares = delta_val / price

            if action.action == "buy" and delta_val > 0:
                cost = delta_val * (1 + 0.001)   # include commission
                if cost <= self.cash:
                    if ticker in self.positions:
                        self.positions[ticker].shares += delta_shares
                    else:
                        self.positions[ticker] = PortfolioPosition(
                            ticker=ticker,
                            shares=delta_shares,
                            entry_price=price,
                            entry_time=timestamp,
                            current_price=price,
                            weight=target_map.get(ticker, 0.0),
                        )
                    self.cash -= cost

            elif action.action == "sell" and ticker in self.positions:
                sell_shares = min(abs(delta_shares), self.positions[ticker].shares)
                proceeds = sell_shares * price * (1 - 0.001)
                self.positions[ticker].shares -= sell_shares
                self.cash += proceeds
                if self.positions[ticker].shares <= 0:
                    del self.positions[ticker]

    @staticmethod
    def _dominant_regime(regimes: Dict[str, str]) -> str:
        if not regimes:
            return "unknown"
        counts = defaultdict(int)
        for r in regimes.values():
            counts[r] += 1
        return max(counts, key=counts.get)


# =============================================================================
# CONVENIENCE CONSTRUCTORS  — called from tradingprofessional.py
# =============================================================================

def create_portfolio_manager(
    tickers: List[str],
    initial_capital: float = 100_000.0,
    rebalance_frequency: str = "weekly",
    risk_limits: Optional[Dict[str, float]] = None,
    # ---- 💼 Enhancement toggles ----
    strict_risk_kwargs: Optional[Dict[str, float]] = None,
    diversification_kwargs: Optional[Dict[str, float]] = None,
    liquidity_kwargs: Optional[Dict[str, float]] = None,
    enable_strict_risk: bool = True,
    enable_diversification: bool = True,
    enable_stress_testing: bool = True,
    enable_liquidity: bool = True,
    enable_attribution: bool = True,
    enable_risk_dashboard: bool = True,
) -> AIPortfolioManager:
    """
    Factory function: instantiate and return a fully configured manager
    with all 💼 enhancements enabled by default.

    Example
    -------
    from ai_portfolio_system import create_portfolio_manager

    mgr = create_portfolio_manager(
        tickers=['BTCUSD', 'ETHUSD', 'GC=F', '^GSPC'],
        initial_capital=200_000,
        rebalance_frequency='weekly',
    )

    # Access enhancement modules:
    stress_results = mgr.run_stress_test(current_prices)
    attribution = mgr.run_attribution(trades)
    div_check = mgr.check_diversification()
    dashboard = mgr.get_risk_dashboard_data()
    """
    return AIPortfolioManager(
        tickers=tickers,
        initial_capital=initial_capital,
        rebalance_frequency=rebalance_frequency,
        risk_limits=risk_limits,
        optimizer_kwargs={
            "risk_aversion": 2.0,
            "max_weight": 0.40,
            "min_weight": 0.02,
        },
        strict_risk_kwargs=strict_risk_kwargs,
        diversification_kwargs=diversification_kwargs,
        liquidity_kwargs=liquidity_kwargs,
        enable_strict_risk=enable_strict_risk,
        enable_diversification=enable_diversification,
        enable_stress_testing=enable_stress_testing,
        enable_liquidity=enable_liquidity,
        enable_attribution=enable_attribution,
        enable_risk_dashboard=enable_risk_dashboard,
    )


def build_asset_view(
    ticker: str,
    ensemble_prediction: float,
    current_price: float,
    confidence: float,
    ensemble_std: float,
    historical_vol: float,
    regime: str = "unknown",
) -> AssetView:
    """
    Helper to build an AssetView from the outputs of enhanced_ensemble_predict().

    Example
    -------
    from ai_portfolio_system import build_asset_view

    view = build_asset_view(
        ticker='BTCUSD',
        ensemble_prediction=pred_price,   # from enhanced_ensemble_predict
        current_price=current_price,
        confidence=confidence_score,
        ensemble_std=pred_std,
        historical_vol=0.60,
    )
    """
    predicted_return = (ensemble_prediction - current_price) / max(current_price, 1e-8)
    # Annualise (assume daily bars, ×252)
    annualised_return = predicted_return * 252

    return AssetView(
        ticker=ticker,
        predicted_return=annualised_return,
        confidence=confidence,
        predicted_volatility=max(historical_vol, 0.01),
        regime=regime,
        model_std=ensemble_std / max(current_price, 1e-8),  # normalise
    )
