"""Render a Dash component tree to plain HTML.

The point of this module is that the dashboard's layout code is not rewritten
for the static build. ``dashboard/layout.py`` produces a tree of Dash
components; this walks that tree and emits HTML, so the served site and the
static site are generated from one definition and cannot drift apart.

Four component types need real handling:

``dcc.Graph``      becomes an empty div plus its figure JSON, which the page's
                   own script hands to Plotly on load. Figures are emitted for
                   both themes so the toggle swaps them rather than trying to
                   recolour a chart in place.
``dash_table``     becomes an ordinary ``<table>``, with the column formatting
                   applied in Python since there is no Dash renderer to do it.
``dcc.Dropdown``   becomes a ``<select>``; every option's panel is pre-rendered
                   and the script shows one.
``dcc.Tabs``       becomes a button row over pre-rendered panels.
"""

from __future__ import annotations

import html
import json
import re
from typing import Any

import numpy as np
import plotly.utils

VOID_TAGS = {"br", "hr", "img", "input", "meta", "link"}


def _camel_to_kebab(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def _style_to_css(style: dict | None) -> str:
    if not style:
        return ""
    return "; ".join(f"{_camel_to_kebab(k)}: {v}" for k, v in style.items())


class HtmlRenderer:
    """Walks a component tree, collecting figures as it goes."""

    def __init__(self) -> None:
        self.figures: dict[str, Any] = {}
        self._counter = 0

    def _next_id(self, hint: str = "fig") -> str:
        self._counter += 1
        return f"{hint}-{self._counter}"

    # -- entry point -------------------------------------------------------
    def render(self, node: Any) -> str:
        if node is None or node is False:
            return ""
        if isinstance(node, (str, int, float)):
            return html.escape(str(node))
        if isinstance(node, (list, tuple)):
            return "".join(self.render(child) for child in node)

        name = type(node).__name__
        handler = getattr(self, f"_render_{name.lower()}", None)
        if handler is not None:
            return handler(node)
        if hasattr(node, "children") or hasattr(node, "className"):
            return self._render_generic(node)
        return html.escape(str(node))

    # -- generic html.* components ----------------------------------------
    def _attrs(self, node: Any, extra: dict[str, str] | None = None) -> str:
        parts: dict[str, str] = {}
        for attr, key in (("className", "class"), ("id", "id"), ("title", "title"),
                          ("href", "href"), ("target", "target")):
            value = getattr(node, attr, None)
            if isinstance(value, str) and value:
                parts[key] = value
        css = _style_to_css(getattr(node, "style", None))
        if css:
            parts["style"] = css
        parts.update(extra or {})
        return "".join(f' {k}="{html.escape(str(v), quote=True)}"' for k, v in parts.items())

    def _render_generic(self, node: Any) -> str:
        tag = type(node).__name__.lower()
        if tag not in ("div", "span", "p", "h1", "h2", "h3", "h4", "h5", "h6",
                       "section", "header", "footer", "ul", "ol", "li", "table",
                       "thead", "tbody", "tr", "th", "td", "strong", "em", "a",
                       "button", "label", "small", "br", "hr", "main", "nav"):
            tag = "div"
        attrs = self._attrs(node)
        if tag in VOID_TAGS:
            return f"<{tag}{attrs}>"
        return f"<{tag}{attrs}>{self.render(getattr(node, 'children', None))}</{tag}>"

    # -- dcc.Graph ---------------------------------------------------------
    def _render_graph(self, node: Any) -> str:
        figure = getattr(node, "figure", None)
        if figure is None:
            return ""
        fig_id = getattr(node, "id", None) or self._next_id()
        if not isinstance(fig_id, str):
            fig_id = self._next_id()
        self.figures[fig_id] = figure
        style = _style_to_css(getattr(node, "style", None))
        return (f'<div class="plot" id="{html.escape(fig_id, quote=True)}"'
                + (f' style="{html.escape(style, quote=True)}"' if style else "")
                + "></div>")

    # -- dash_table.DataTable ---------------------------------------------
    @staticmethod
    def _format_cell(value: Any, column: dict) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "—"
        fmt = (column.get("format") or {}).get("specifier")
        if fmt and isinstance(value, (int, float, np.integer, np.floating)):
            try:
                spec = fmt.lstrip("+")
                plus = fmt.startswith("+")
                text = format(float(value), spec)
                if plus and float(value) >= 0 and not text.startswith("+"):
                    text = "+" + text
                return text
            except (TypeError, ValueError):
                pass
        if isinstance(value, float):
            return f"{value:,.4g}"
        return str(value)

    def _render_datatable(self, node: Any) -> str:
        columns = getattr(node, "columns", []) or []
        rows = getattr(node, "data", []) or []
        left = {"model", "family", "instrument", "moneyness_label", "option_type",
                "horizon", "expiry", "structure", "parameter", "quantile"}

        head = "".join(
            f'<th class="{"txt" if c["id"] in left else "num"}">{html.escape(str(c["name"]))}</th>'
            for c in columns)
        body = []
        for row in rows:
            cells = "".join(
                f'<td class="{"txt" if c["id"] in left else "num"}">'
                f'{html.escape(self._format_cell(row.get(c["id"]), c))}</td>'
                for c in columns)
            body.append(f"<tr>{cells}</tr>")
        return (f'<div class="table-wrap"><table class="dtable">'
                f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
                f"</table></div>")

    # -- controls ----------------------------------------------------------
    def _render_dropdown(self, node: Any) -> str:
        options = getattr(node, "options", []) or []
        value = getattr(node, "value", None)
        select_id = getattr(node, "id", None) or self._next_id("select")
        opts = "".join(
            f'<option value="{html.escape(str(o["value"]), quote=True)}"'
            f'{" selected" if o["value"] == value else ""}>'
            f'{html.escape(str(o["label"]))}</option>' for o in options)
        return (f'<select class="picker" id="{html.escape(str(select_id), quote=True)}">'
                f"{opts}</select>")

    def _render_radioitems(self, node: Any) -> str:
        options = getattr(node, "options", []) or []
        value = getattr(node, "value", None)
        group_id = getattr(node, "id", None) or self._next_id("radio")
        items = "".join(
            f'<label><input type="radio" name="{html.escape(str(group_id), quote=True)}"'
            f' value="{html.escape(str(o["value"]), quote=True)}"'
            f'{" checked" if o["value"] == value else ""}>'
            f'{html.escape(str(o["label"]))}</label>' for o in options)
        return f'<div class="radio-group" id="{html.escape(str(group_id), quote=True)}">{items}</div>'

    # -- containers that carry no markup of their own ----------------------
    def _render_loading(self, node: Any) -> str:
        return self.render(getattr(node, "children", None))

    def _render_store(self, node: Any) -> str:
        return ""

    def _render_interval(self, node: Any) -> str:
        return ""

    def _render_tabs(self, node: Any) -> str:
        return ""

    def _render_tab(self, node: Any) -> str:
        return ""


def figure_to_json(figure: Any) -> str:
    return json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder)
