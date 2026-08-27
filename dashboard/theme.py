"""Colour and typography tokens.

The categorical hues, and the order they are assigned in, are a validated set
rather than a taste call: every adjacent pair clears the colour-vision-deficiency
and normal-vision separation floors on both the light and dark surfaces.

Two constraints fall out of that validation and are enforced in ``figures.py``:

* **Four family hues are safe only where same-family marks sit together** --
  bars and tables.  In a form where any series can land next to any other
  (overlaid lines, scatter) the safe set is three, so those charts either cap at
  three series or use small multiples.
* **Direction uses blue/red, not green/red.**  Red-green is the conventional
  finance pairing and the single worst choice for the ~8% of men with deuteranopia,
  for whom the two are nearly identical.  Direction is additionally always
  spelled out in text, so colour never carries the meaning alone.
"""

from __future__ import annotations

# --- categorical: model families -------------------------------------------
FAMILY_COLORS_LIGHT = {
    "Stochastic": "#2a78d6",        # blue
    "Machine Learning": "#eb6834",  # orange
    "Neural Network": "#1baf7a",    # aqua
    "Hybrid": "#4a3aa7",            # violet
}
FAMILY_COLORS_DARK = {
    "Stochastic": "#3987e5",
    "Machine Learning": "#d95926",
    "Neural Network": "#199e70",
    "Hybrid": "#9085e9",
}

# --- diverging: direction ---------------------------------------------------
UP_LIGHT, DOWN_LIGHT, NEUTRAL_LIGHT = "#2a78d6", "#e34948", "#8b8a85"
UP_DARK, DOWN_DARK, NEUTRAL_DARK = "#3987e5", "#e66767", "#9a998f"

# --- sequential blue ramp ---------------------------------------------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

LIGHT = {
    "surface": "#fcfcfb",
    "surface_raised": "#ffffff",
    "surface_sunken": "#f4f3f0",
    "border": "#dedcd6",
    "border_strong": "#c3c1b9",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#7a7973",
    "grid": "#e8e6e1",
    "up": UP_LIGHT, "down": DOWN_LIGHT, "neutral": NEUTRAL_LIGHT,
    "families": FAMILY_COLORS_LIGHT,
    "accent": "#2a78d6",
    "warn": "#eda100",
}

DARK = {
    "surface": "#1a1a19",
    "surface_raised": "#232322",
    "surface_sunken": "#141413",
    "border": "#383835",
    "border_strong": "#4d4d49",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#8e8d84",
    "grid": "#2e2e2c",
    "up": UP_DARK, "down": DOWN_DARK, "neutral": NEUTRAL_DARK,
    "families": FAMILY_COLORS_DARK,
    "accent": "#3987e5",
    "warn": "#c98500",
}


def tokens(theme: str) -> dict:
    return DARK if theme == "dark" else LIGHT


def family_color(family: str, theme: str) -> str:
    palette = tokens(theme)["families"]
    return palette.get(family, tokens(theme)["neutral"])


def direction_color(p_up: float, theme: str, deadband: float = 0.02) -> str:
    t = tokens(theme)
    if abs(p_up - 0.5) < deadband:
        return t["neutral"]
    return t["up"] if p_up >= 0.5 else t["down"]


FONT_STACK = ('ui-sans-serif, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", '
              'Arial, sans-serif')
MONO_STACK = ('ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
              '"Liberation Mono", monospace')
