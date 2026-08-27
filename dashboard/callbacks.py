"""Callback wiring.

Two structural choices worth stating:

* **One render callback per view, driven by stores.**  A theme or selection
  change re-renders the tab from the current snapshot rather than mutating
  figures in place, so there is exactly one code path that produces each view.
* **The poll tick does not re-render the tabs.**  It refreshes the status line
  and bumps a snapshot identifier; the tab body only rebuilds when that
  identifier actually changes.  Otherwise every open page would rebuild a dozen
  figures every twenty seconds and reset whatever the reader was looking at.
"""

from __future__ import annotations

import logging
import threading

from dash import Input, Output, State, callback_context, html, no_update

from btcmodels.engine import Engine

from . import layout as L
from .components import callout

log = logging.getLogger(__name__)


def register(app, engine: Engine) -> None:

    # -- theme --------------------------------------------------------------
    @app.callback(
        Output("theme-store", "data"),
        Input("theme-button", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_theme(_clicks, current):
        return "light" if current == "dark" else "dark"

    @app.callback(
        Output("theme-button", "children"),
        Input("theme-store", "data"),
    )
    def theme_button_label(theme):
        return "Light" if (theme or "dark") == "dark" else "Dark"

    app.clientside_callback(
        """
        function(theme) {
            document.documentElement.setAttribute('data-theme', theme || 'dark');
            return '';
        }
        """,
        Output("theme-sink", "children"),
        Input("theme-store", "data"),
    )

    # -- horizon ------------------------------------------------------------
    @app.callback(
        Output("horizon-store", "data"),
        Input("horizon-toggle", "value"),
        prevent_initial_call=True,
    )
    def set_horizon(value):
        return value or "1w"

    @app.callback(
        Output("horizon-toggle", "value"),
        Input("horizon-store", "data"),
    )
    def sync_horizon(value):
        return value or "1w"

    # -- status line + snapshot identity ------------------------------------
    @app.callback(
        Output("masthead", "children"),
        Output("snapshot-id", "data"),
        Input("poll", "n_intervals"),
        Input("theme-store", "data"),
        Input("horizon-store", "data"),
        Input("refresh-button", "n_clicks"),
        State("snapshot-id", "data"),
    )
    def update_masthead(_ticks, theme, horizon, _clicks, current_id):
        triggered = [t["prop_id"] for t in callback_context.triggered]
        if any("refresh-button" in t for t in triggered):
            # Refreshing takes about a minute. Doing it inline would block the
            # request until gunicorn's timeout; instead it is kicked off in the
            # background and the poll tick picks up the new snapshot.
            _refresh_async(engine)

        snapshot = engine.snapshot
        new_id = snapshot.generated_at.isoformat() if snapshot else ""
        return (
            L.render_masthead(snapshot, theme or "dark", horizon or "1w", engine.status),
            new_id if new_id != current_id else no_update,
        )

    # -- selections ---------------------------------------------------------
    @app.callback(Output("model-store", "data"), Input("model-select", "value"),
                  prevent_initial_call=True)
    def set_model(value):
        return value or no_update

    @app.callback(Output("backtest-store", "data"), Input("backtest-model-select", "value"),
                  prevent_initial_call=True)
    def set_backtest_model(value):
        return value or no_update

    @app.callback(Output("options-store", "data"), Input("options-model-select", "value"),
                  prevent_initial_call=True)
    def set_options_model(value):
        return value or no_update

    @app.callback(Output("greek-store", "data"), Input("greek-select", "value"),
                  prevent_initial_call=True)
    def set_greek(value):
        return value or no_update

    # -- the single render callback -----------------------------------------
    @app.callback(
        Output("tab-content", "children"),
        Input("tabs", "value"),
        Input("theme-store", "data"),
        Input("horizon-store", "data"),
        Input("snapshot-id", "data"),
        Input("model-store", "data"),
        Input("backtest-store", "data"),
        Input("options-store", "data"),
        Input("greek-store", "data"),
    )
    def render_tab(tab, theme, horizon, _snapshot_id, model_key, backtest_key,
                   options_key, greek):
        theme = theme or "dark"
        horizon = horizon or "1w"
        snapshot = engine.snapshot

        if snapshot is None:
            message = ("Building the first snapshot — fetching prices, fitting eleven "
                       f"models and pulling the live option chain. Status: {engine.status}."
                       + (f" Last error: {engine.last_error}" if engine.last_error else ""))
            return html.Div(callout(message, "info", "Warming up"), className="panel")

        try:
            if tab == "forecast":
                return L.render_forecast(snapshot, theme, horizon)
            if tab == "models":
                return L.render_models(snapshot, theme, horizon, model_key or "gbm")
            if tab == "backtest":
                return L.render_backtest(snapshot, theme, horizon, backtest_key or "gbm")
            if tab == "options":
                return L.render_options(snapshot, theme, options_key or "gbm",
                                        greek or "theta")
            if tab == "method":
                return L.render_method(theme)
        except Exception as exc:
            log.exception("tab render failed")
            return html.Div(callout(f"{type(exc).__name__}: {exc}", "bad",
                                    "This view failed to render"), className="panel")
        return html.Div()


def _refresh_async(engine: Engine) -> None:
    """Kick a refresh onto a worker thread, ignoring double-clicks."""
    if getattr(engine, "_manual_thread", None) and engine._manual_thread.is_alive():
        return

    def run():
        try:
            engine.refresh(force_data=True)
        except Exception as exc:
            log.exception("manual refresh failed")
            engine.last_error = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=run, name="btc-manual-refresh", daemon=True)
    engine._manual_thread = thread
    thread.start()
