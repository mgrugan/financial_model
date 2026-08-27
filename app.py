#!/usr/bin/env python3
"""Dash entry point.

Run locally::

    python app.py

In production (Render) gunicorn imports the module-level ``server``::

    gunicorn app:server --workers 1 --threads 4 --timeout 180

One worker, several threads, is deliberate: the fitted models live in this
process's memory and a second worker would train its own copy of all eleven.
"""

from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings("ignore")

import dash
from dash import html

from btcmodels.engine import get_engine
from dashboard import callbacks
from dashboard.layout import shell

logging.basicConfig(
    level=os.environ.get("BTC_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = dash.Dash(
    __name__,
    title="Bitcoin Model Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description",
         "content": "Eleven stochastic, machine-learning and hybrid models forecasting "
                    "Bitcoin, with walk-forward backtests and live options analytics."},
    ],
)
server = app.server                      # gunicorn target

app.index_string = """<!DOCTYPE html>
<html data-theme="dark">
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <script>
      // Apply the stored theme before first paint so the page never flashes light.
      try {
        var stored = JSON.parse(window.localStorage.getItem('theme-store') || '"dark"');
        document.documentElement.setAttribute('data-theme', stored);
      } catch (e) {}
    </script>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""

app.layout = shell
engine = get_engine(start=True)
callbacks.register(app, engine)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("BTC_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
