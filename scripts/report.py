#!/usr/bin/env python3
"""Print the full model and options report to the terminal.

The same numbers the dashboard shows, without a browser -- useful for a quick
check, for piping into a log, or for running on a machine with no display.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

import logging

logging.basicConfig(level=logging.ERROR)

import numpy as np

from btcmodels.config import HORIZONS, HORIZON_LABELS
from btcmodels.engine import Engine
from btcmodels.registry import FAMILY_ORDER, MODEL_ORDER

RULE = "=" * 108
THIN = "-" * 108


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-options", action="store_true", help="skip the options section")
    parser.add_argument("--model", help="only show this model in the options section")
    args = parser.parse_args()

    print("Fetching data and fitting eleven models…", flush=True)
    snapshot = Engine().refresh()

    market = snapshot.market
    header("BITCOIN MODEL REPORT")
    print(f"  Price          ${market['price']:,.2f}   "
          f"24h {market.get('change_24h', 0):+.2f}%   "
          f"7d {market.get('change_7d', 0):+.2f}%   "
          f"30d {market.get('change_30d', 0):+.2f}%")
    print(f"  Realised vol   30d {market.get('realised_vol_30d', 0):.1f}%   "
          f"90d {market.get('realised_vol_90d', 0):.1f}%   (annualised)")
    print(f"  History        {market.get('n_bars', 0):,} daily bars from "
          f"{str(market.get('history_start'))[:10]} through {str(market.get('as_of'))[:10]}")
    if snapshot.errors:
        print(f"  Errors         {snapshot.errors}")

    backtest_available = snapshot.backtest.get("available")

    for horizon_key, horizon_days in HORIZONS.items():
        forecasts = snapshot.bundle.by_horizon(horizon_key)
        consensus = snapshot.bundle.consensus[horizon_key]

        header(f"DIRECTION — {HORIZON_LABELS[horizon_key].upper()}")
        print(f"  {'MODEL':<34}{'DIR':<6}{'P(UP)':>8}{'±':>7}   "
              f"{'E[RET]':>9}{'MEDIAN':>11}{'90% BAND':>23}{'VOL':>7}   TRACK RECORD")
        print(f"  {THIN}")

        last_family = None
        for f in sorted(forecasts, key=lambda x: (FAMILY_ORDER.index(x.family), -x.p_up)):
            if f.family != last_family:
                print(f"  {f.family}")
                last_family = f.family
            q = f.quantile_prices()
            rel = snapshot.reliability(f.model_key, horizon_key)
            if rel is None:
                verdict = "no backtest"
            elif rel["has_skill"]:
                verdict = f"SKILL  acc {rel['accuracy_pct']:.1f}% vs {rel['always_up_accuracy_pct']:.1f}%, AUC {rel['auc']:.3f}"
            else:
                verdict = f"none   acc {rel['accuracy_pct']:.1f}% vs {rel['always_up_accuracy_pct']:.1f}%, AUC {rel['auc']:.3f}"
            print(f"    {f.model_name:<32}{f.direction:<6}{f.p_up * 100:>7.2f}%"
                  f"{f.p_up_stderr * 100:>6.2f}   {f.expected_return:>+8.2f}%"
                  f"{q['q50']:>11,.0f}   ${q['q05']:>9,.0f}–${q['q95']:>9,.0f}"
                  f"{f.annualised_vol:>6.0f}%   {verdict}")

        print(f"  {THIN}")
        print(f"    {'CONSENSUS':<32}{consensus['direction']:<6}"
              f"{consensus['p_up'] * 100:>7.2f}%{'':>6}   {'':>9}"
              f"{consensus['median_price']:>11,.0f}   "
              f"${consensus['q05']:>9,.0f}–${consensus['q95']:>9,.0f}"
              f"{'':>7}   {consensus['n_up']} up / {consensus['n_down']} down")

    if not backtest_available:
        print(f"\n  NOTE: no walk-forward backtest is cached "
              f"({snapshot.backtest.get('reason', '')}). None of the probabilities "
              f"above has a track record attached.\n  Run: python scripts/run_backtest.py")
    else:
        header("WALK-FORWARD BACKTEST")
        summary = snapshot.backtest_summary()
        print(f"  {snapshot.backtest.get('backtest_days')} test days, re-estimated every "
              f"{snapshot.backtest.get('refit_every')} days, data through "
              f"{snapshot.backtest.get('data_end')}")
        for horizon_key in HORIZONS:
            block = summary[summary["horizon"] == horizon_key].sort_values(
                "brier_skill", ascending=False)
            if block.empty:
                continue
            print(f"\n  {HORIZON_LABELS[horizon_key]}   "
                  f"(always-up baseline = {block['always_up_accuracy_pct'].iloc[0]:.1f}% accuracy)")
            print(f"    {'MODEL':<34}{'ACC':>8}{'EDGE':>8}{'AUC':>8}"
                  f"{'BRIER SKILL':>14}{'LOGLOSS SKILL':>15}{'SHARPE':>9}{'RETURN':>10}{'MAXDD':>9}")
            print(f"    {THIN[:104]}")
            for _, row in block.iterrows():
                print(f"    {row['model']:<34}{row['accuracy_pct']:>7.1f}%"
                      f"{row['edge_vs_always_up_pp']:>+7.2f}{row['auc']:>8.3f}"
                      f"{row['brier_skill']:>+14.4f}{row['log_loss_skill']:>+15.4f}"
                      f"{row['sharpe']:>9.2f}{row['total_return_pct']:>+9.1f}%"
                      f"{row['max_drawdown_pct']:>8.1f}%")
            bh = block['buy_hold_return_pct'].iloc[0]
            print(f"    {'buy & hold over the same window':<34}{'':>8}{'':>8}{'':>8}"
                  f"{'':>14}{'':>15}{'':>9}{bh:>+9.1f}%")

    if args.no_options:
        return 0

    view = snapshot.option_view
    if not view.get("available"):
        header("OPTIONS")
        print(f"  unavailable: {view.get('error')}")
        return 0

    header(f"OPTIONS — EXPIRY {str(view['target_expiry'])[:10]} "
           f"({view['target_dte']:.1f} DAYS)")
    print(f"  Deribit index ${view['index_price']:,.0f}   "
          f"ATM implied vol {view['atm_iv_pct']:.1f}%   "
          f"carry {view['rate'] * 100:.2f}%   "
          f"{view['n_contracts']} tradeable contracts")
    print("  Values below are expected payoffs under each model's REAL-WORLD distribution,")
    print("  not arbitrage-free prices. A model that forecasts more volatility than the")
    print("  market is charging will find every contract cheap — check the vol gap.")

    keys = [args.model] if args.model else MODEL_ORDER
    for key in keys:
        entry = view["per_model"].get(key)
        if not entry or "error" in entry:
            continue
        vol = entry["vol_context"]
        rel = snapshot.reliability(key, view["horizon_key"])
        badge = ("beat the base rate out of sample" if rel and rel["has_skill"]
                 else "no out-of-sample edge" if rel else "no backtest")
        print(f"\n  {THIN}")
        print(f"  {entry['model_name']}   P(up) {entry['p_up'] * 100:.1f}% → "
              f"{entry['direction']}   |   model vol {vol['model_vol_pct']:.1f}% vs "
              f"market {vol['market_atm_iv_pct']:.1f}% ({vol['vol_gap_pts']:+.1f} pts)"
              f"   |   {badge}")
        if not entry["suggestions"]:
            print("    no liquid contracts matched the delta targets")
            continue
        print(f"    {'':4}{'INSTRUMENT':<22}{'PREM':>9}{'DELTA':>8}{'GAMMA':>9}"
              f"{'THETA/DAY':>12}{'VEGA':>8}{'BREAKEVEN':>12}{'P(ITM)':>9}{'EDGE':>9}{'TARGET':>11}")
        for s in entry["suggestions"]:
            print(f"    {s['moneyness_label']:<4}{s['instrument']:<22}"
                  f"{s['premium_usd']:>9,.0f}{s['delta']:>8.3f}"
                  f"{s['gamma'] * 1e6:>9.2f}"
                  f"{s['theta_usd_per_day']:>9,.0f}/d{s['vega']:>8,.0f}"
                  f"{s['breakeven']:>12,.0f}{s['model_prob_itm_pct']:>8.1f}%"
                  f"{s['model_edge_pct']:>+8.0f}%{s['target_price_median']:>11,.0f}")
        spread = entry.get("spread")
        if spread:
            print(f"    {'SPR':<4}{spread['structure']:<22}"
                  f"{spread['net_debit_usd']:>9,.0f}{spread['net_delta']:>8.3f}"
                  f"{spread['net_gamma'] * 1e6:>9.2f}"
                  f"{spread['net_theta_usd_per_day']:>9,.0f}/d{spread['net_vega']:>8,.0f}"
                  f"{spread['breakeven']:>12,.0f}"
                  f"{spread['model_prob_profit_pct']:>8.1f}%"
                  f"{'':>9}  max win ${spread['max_profit_usd']:,.0f}")

    print(f"\n  Gamma is per $1m of spot move. P(ITM) and EDGE are under the model's own")
    print(f"  distribution. TARGET is that model's median price at the expiry.")
    print(f"\n  Research tool, not financial advice. Long options can lose 100% of premium.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
