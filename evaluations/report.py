#!/usr/bin/env python3
"""
Usage:
    python report.py path/to/evaluation.csv

Importable API:
    from report import aggregate, build_delta_mrr_report
"""

import sys
import pandas as pd
import numpy as np


# ── core functions (importable) ───────────────────────────────────────────────

def aggregate(df: pd.DataFrame, n_iterations: int = None) -> pd.DataFrame:
    """
    Collapse multiple iteration rows into one row per (type, query, target).
      - rel* columns    → list of all values across iterations
      - time            → mean
      - best_* columns  → majority vote
      - rank_* columns  → mean; also adds _min, _max, _std variants
      - shapley* cols   → mean
      - model           → first value

    n_iterations: if given, only use the first n iterations (0 .. n_iterations-1).
    """
    if n_iterations is not None:
        df = df[df['iteration'] < n_iterations]

    group_cols = ['type', 'query', 'target']
    rel_cols   = [c for c in df.columns if c.startswith('rel')]
    rank_cols  = [c for c in df.columns if c.startswith('rank_')]
    best_cols  = [c for c in df.columns if c.startswith('best_')]
    shap_cols  = [c for c in df.columns if c.startswith('shapley')]

    def majority_vote(x):
        return x.mode().iloc[0]

    agg_dict = {}
    for c in rel_cols:
        agg_dict[c] = list
    agg_dict['time'] = 'mean'
    for c in best_cols:
        agg_dict[c] = majority_vote
    for c in rank_cols:
        agg_dict[c] = 'mean'
    for c in shap_cols:
        agg_dict[c] = 'mean'
    agg_dict['model'] = 'first'

    df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()

    rank_stats = df.groupby(group_cols)[rank_cols].agg(['min', 'max', 'std'])
    rank_stats.columns = [f'{col}_{stat}' for col, stat in rank_stats.columns]
    df_agg = df_agg.merge(rank_stats.reset_index(), on=group_cols)

    return df_agg


def compute_mrr(df_agg: pd.DataFrame, rank_col: str) -> float:
    """MRR_q = mean of 1/rank over all targets of a query; MRR = mean over queries."""
    mrr_per_query = df_agg.groupby(['type', 'query'])[rank_col].apply(
        lambda ranks: (1.0 / ranks).mean()
    )
    return mrr_per_query.mean()


def build_delta_mrr_report(df_agg: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (report, report_pct) where:
      - report     : absolute MRR and DeltaMRR values
      - report_pct : same values multiplied by 100
    Methods are inferred automatically from rank_necc_* columns.
    """
    rank_cols = [c for c in df_agg.columns if c.startswith('rank_') and
                 not any(c.endswith(s) for s in ('_min', '_max', '_std'))]

    mrr = {c: compute_mrr(df_agg, c) for c in rank_cols}

    methods = sorted({
        c.replace('rank_necc_', '')
        for c in rank_cols if c.startswith('rank_necc_')
    })

    records = []
    for m in methods:
        records.append({
            'Method':        m.capitalize(),
            'MRR_Necc':      mrr[f'rank_necc_{m}'],
            'MRR_Suff':      mrr[f'rank_suff_{m}'],
            'DeltaMRR_Necc': mrr[f'rank_necc_{m}'] - mrr['rank_full'],
            'DeltaMRR_Suff': mrr[f'rank_suff_{m}'] - mrr['rank_empty'],
        })

    records.append({
        'Method': 'Full  (Necc baseline)',
        'MRR_Necc': mrr['rank_full'], 'MRR_Suff': None,
        'DeltaMRR_Necc': 0.0, 'DeltaMRR_Suff': None,
    })
    records.append({
        'Method': 'Empty (Suff baseline)',
        'MRR_Necc': None, 'MRR_Suff': mrr['rank_empty'],
        'DeltaMRR_Necc': None, 'DeltaMRR_Suff': 0.0,
    })

    report = pd.DataFrame(records).set_index('Method')
    return report, report * 100


def run(csv_path: str, n_iterations: int = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load a CSV, aggregate, and build reports. Returns (df_agg, report, report_pct).

    n_iterations: if given, only use the first n iterations (0 .. n_iterations-1).
    """
    df = pd.read_csv(csv_path)
    df_agg = aggregate(df, n_iterations=n_iterations)
    report, report_pct = build_delta_mrr_report(df_agg)
    return df_agg, report, report_pct


# ── CLI ───────────────────────────────────────────────────────────────────────

def _section(title):
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")


def main(csv_path):
    print(f"Loading {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}")

    df_agg = aggregate(df)
    _section("Aggregated dataset")
    print(f"Shape: {df_agg.shape}  ({df_agg.shape[0]} unique type-query-target rows)")
    print(df_agg.head(3).to_string())

    rank_cols = [c for c in df_agg.columns if c.startswith('rank_') and
                 not any(c.endswith(s) for s in ('_min', '_max', '_std'))]
    mrr = {c: compute_mrr(df_agg, c) for c in rank_cols}

    _section("MRR values")
    for k, v in mrr.items():
        print(f"  {k:30s}: {v:.6f}")

    report, report_pct = build_delta_mrr_report(df_agg)

    _section("Delta MRR Report (absolute)")
    print(report.to_string(float_format=lambda x: f"{x:.6f}"))

    _section("Delta MRR Report (percentages, ×100)")
    print(report_pct.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python report.py <path_to_csv>")
        sys.exit(1)
    main(sys.argv[1])
