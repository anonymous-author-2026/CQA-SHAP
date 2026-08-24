"""
Relation Similarity Analysis
============================
For a given model and dataset, builds an R x R table where:
  - Row i corresponds to relation r_i (the "source" relation)
  - Column j corresponds to relation r_j (the "tested" relation)
  - Entry [i, j] = average score of the true tail entity t when querying (h, r_j)
    over N triples (h, r_i, t) sampled from graph_valid

Usage:
    python relation_similarity/relation_similarity.py \\
        --model cqd --dataset FB --N 50

    python relation_similarity/relation_similarity.py \\
        --model betae --dataset NELL   # N not specified → use all edges
"""

import sys
import os
import argparse
import warnings
import random
from collections import defaultdict

import numpy as np
import pandas as pd

# Run from the CQA-SHAP root directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Required so that 'import kbc' inside models/CQD/predict.py resolves correctly
sys.path.insert(0, 'models/CQD')

from utils import setup_dataset_and_graphs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_edge_index(graph):
    """Return a dict mapping relation_id -> list[Edge]."""
    index = defaultdict(list)
    for edge in graph.get_edges():
        index[edge.id].append(edge)
    return index



def _sample_edges(edges, n_samples, relation_id, rel_name):
    """
    Return up to n_samples edges.  If n_samples is None, return all.
    Warns when the graph has fewer edges than requested.
    """
    if n_samples is None:
        return edges
    if len(edges) > n_samples:
        return random.sample(edges, n_samples)
    if len(edges) < n_samples:
        warnings.warn(
            f"Relation {relation_id} ({rel_name}) has only {len(edges)} edge(s) "
            f"in graph_valid, fewer than the requested N={n_samples}. "
            f"Using all {len(edges)} edge(s)."
        )
    return edges  # use all available


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_relation_similarity_table(model, dataset, graph_valid, n_samples=None, seed=42, chunk_size=32):
    """
    Build the R x R relation-similarity table.

    Parameters
    ----------
    model        : loaded CQD or KGR instance (must expose predict_1p)
    dataset      : Dataset with id2rel mapping
    graph_valid  : Graph object (train + valid triples)
    n_samples    : int or None – edges sampled per source relation (None = all)
    seed         : random seed for reproducibility

    Returns
    -------
    table  : np.ndarray of shape (R, R)
    rel_ids: list of relation IDs in row/column order
    """
    random.seed(seed)

    rel_ids = sorted(dataset.id2rel.keys())
    R = len(rel_ids)

    print(f"Indexing {graph_valid.get_num_edges()} edges by relation …")
    edge_index = _build_edge_index(graph_valid)

    table = np.zeros((R, R), dtype=np.float32)

    for i, rel_i in enumerate(rel_ids):
        rel_name_i = dataset.id2rel[rel_i]
        edges = edge_index.get(rel_i, [])
        # Drop edges whose head or tail entity was not found in the dataset mapping
        # (skip_missing=False during graph loading can produce Node(id=None))
        edges = [e for e in edges if e.head.id is not None and e.tail.id is not None]

        if len(edges) == 0:
            print(
                f"  [{i+1:4d}/{R}] Relation {rel_i} ({rel_name_i}): "
                f"no edges with valid entity IDs in graph_valid — row left as zeros."
            )
            continue

        sampled = _sample_edges(edges, n_samples, rel_i, rel_name_i)
        n_used = len(sampled)

        print(f"  [{i+1:4d}/{R}] Relation {rel_i} ({rel_name_i}) — {n_used} edge(s)")

        # One batched forward pass per edge (all R column relations at once)
        col_sum = np.zeros(R, dtype=np.float64)
        for edge in sampled:
            # [R, n_entities] — scores for every column relation against every entity
            batch_scores = model.predict_1p_batch(edge.head.id, rel_ids, chunk_size=chunk_size)
            # Direct O(1) index: score of true tail for each column relation
            col_sum += batch_scores[:, edge.tail.id].cpu().numpy()
        table[i] = col_sum / n_used

    return table, rel_ids


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name, dataset_name):
    """Load and return a CQD or KGR model instance."""
    if model_name == 'cqd':
        from models.CQD.predict import CQD
        if dataset_name == 'FB':
            model = CQD(
                model_path='models/CQD/models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
                data_dir='models/CQD/data/FB15k-237',
                normalize = False
            )
        else:
            model = CQD(
                model_path='models/CQD/models/NELL-model-rank-1000-epoch-100-1602499096.pt',
                data_dir='models/CQD/data/NELL995',
                normalize = False
            )
    else:
        from models.KGReasoning.predict import KGR
        model = KGR(method=model_name, dataset=dataset_name)
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute the R×R relation-similarity table for CQD or KGReasoning models."
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['cqd', 'query2box', 'betae', 'gqe'],
        help='Model to evaluate',
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['FB', 'NELL'],
        help='Dataset shorthand: FB → FB15k-237, NELL → NELL995',
    )
    parser.add_argument(
        '--N', type=int, default=None,
        help='Edges sampled per source relation (omit to use ALL edges)',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for edge sampling (default: 42)',
    )
    parser.add_argument(
        '--output_dir', type=str, default='relation_similarity/results',
        help='Directory where output files are saved',
    )
    parser.add_argument(
        '--chunk_size', type=int, default=32,
        help='Relations per forward pass for KGReasoning models (lower = less GPU memory, default: 32)',
    )
    args = parser.parse_args()

    # Dataset paths
    if args.dataset == 'FB':
        data_dir = 'data/FB15k-237'
        kg_name = 'FB15k-237'
    else:
        data_dir = 'data/NELL995'
        kg_name = 'NELL995'

    # Setup dataset & graphs
    print(f"Loading dataset from {data_dir} …")
    dataset, _, graph_valid, _ = setup_dataset_and_graphs(data_dir, logging=True, add_reverse=True)

    R = len(dataset.id2rel)
    n_str = f'N{args.N}' if args.N is not None else 'all'
    print(f"Relations: {R}  |  Edges sampled per relation: {n_str}")

    # Load model
    print(f"Loading model '{args.model}' …")
    model = load_model(args.model, args.dataset)

    # Build table
    print(f"\nBuilding {R}×{R} relation-similarity table …")
    table, rel_ids = build_relation_similarity_table(
        model=model,
        dataset=dataset,
        graph_valid=graph_valid,
        n_samples=args.N,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = f'{kg_name}_{args.model}_{n_str}'

    npy_path = os.path.join(args.output_dir, base_name + '.npy')
    np.save(npy_path, table)
    print(f"\nSaved numpy array  → {npy_path}")

    rel_names = [dataset.id2rel[rid] for rid in rel_ids]
    df = pd.DataFrame(table, index=rel_names, columns=rel_names)
    csv_path = os.path.join(args.output_dir, base_name + '.csv')
    df.to_csv(csv_path)
    print(f"Saved CSV table    → {csv_path}")


if __name__ == '__main__':
    main()
