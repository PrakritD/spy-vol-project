"""Live-readiness layer.

Production-thinking stubs that turn the offline backtest pipeline into
something a cron / orchestrator could call once per day at market close
to produce tomorrow's sized position. Not an actual trading bot — the
broker integration, risk limits, and live data ingestion are explicitly
out of scope.

Entry point: `live.predict_today.predict_today()`.
"""
