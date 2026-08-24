"""
Relation Cosine Similarity
==========================
For a given model and dataset, builds an R x R table where entry [i, j] is
the cosine similarity between the embedding of relation r_i and the embedding
of relation r_j, as stored in the trained model.

This is purely embedding-based (no inference / graph traversal needed), so it
runs in seconds regardless of dataset size.

Supported models
----------------
  cqd        – ComplEx KBC (kbc embedding matrix index 1)
  query2box  – KGReasoning box model  (relation_embedding)
  betae      – KGReasoning beta model (relation_embedding)
  gqe        – KGReasoning vec model  (relation_embedding)

Usage
-----
    # from the CQA-SHAP root directory
    python relation_similarity/cosine_similarity.py --model cqd    --dataset FB
    python relation_similarity/cosine_similarity.py --model betae  --dataset NELL
    python relation_similarity/cosine_similarity.py --model query2box --dataset FB --output_dir relation_similarity/cosine_results
"""

import sys
import os
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup – run from the CQA-SHAP root directory
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, 'models/CQD')   # so that 'import kbc' inside CQD/predict.py works

from utils import setup_dataset_and_graphs


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def get_relation_embeddings(model_name: str, model) -> torch.Tensor:
    """
    Return the relation embedding matrix as a CPU float tensor of shape (R, dim).

    For CQD the relation embeddings live at kbc.model.embeddings[1].
    For KGReasoning models (query2box / betae / gqe) they live at model.relation_embedding.
    """
    if model_name == 'cqd':
        emb = model.kbc.model.embeddings[1].weight.data  # (R, dim)
    else:
        emb = model.model.relation_embedding.data        # (R, dim)
    return emb.cpu().float()


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def build_cosine_similarity_table(embeddings: torch.Tensor) -> np.ndarray:
    """
    Compute the full R×R pairwise cosine-similarity matrix in one shot.

    Parameters
    ----------
    embeddings : FloatTensor of shape (R, dim)

    Returns
    -------
    np.ndarray of shape (R, R), values in [-1, 1]
    """
    normed = F.normalize(embeddings, p=2, dim=1)   # (R, dim)
    sim_matrix = (normed @ normed.T).numpy()        # (R, R)
    return sim_matrix.astype(np.float32)


# ---------------------------------------------------------------------------
# Model loading  (mirrors relation_similarity.py)
# ---------------------------------------------------------------------------

def load_model(model_name: str, dataset_name: str):
    """Load and return a CQD or KGR model instance."""
    if model_name == 'cqd':
        from models.CQD.predict import CQD
        if dataset_name == 'FB':
            model = CQD(
                model_path='models/CQD/models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt',
                data_dir='models/CQD/data/FB15k-237',
                normalize=False,
            )
        else:
            model = CQD(
                model_path='models/CQD/models/NELL-model-rank-1000-epoch-100-1602499096.pt',
                data_dir='models/CQD/data/NELL995',
                normalize=False,
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
        description="Compute the R×R cosine-similarity table from relation embeddings."
    )
    parser.add_argument(
        '--model', type=str, required=True,
        choices=['cqd', 'query2box', 'betae', 'gqe'],
        help='Model whose relation embeddings to use',
    )
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['FB', 'NELL'],
        help='Dataset shorthand: FB → FB15k-237, NELL → NELL995',
    )
    parser.add_argument(
        '--output_dir', type=str, default='relation_similarity/cosine_results',
        help='Directory where output files are saved (default: relation_similarity/cosine_results)',
    )
    args = parser.parse_args()

    # Dataset paths
    if args.dataset == 'FB':
        data_dir = 'data/FB15k-237'
        kg_name  = 'FB15k-237'
    else:
        data_dir = 'data/NELL995'
        kg_name  = 'NELL995'

    # Load dataset (only need id2rel mapping – graph not required here)
    print(f"Loading dataset from {data_dir} …")
    dataset, _, _, _ = setup_dataset_and_graphs(data_dir, logging=True, add_reverse=True)

    R = len(dataset.id2rel)
    rel_ids   = sorted(dataset.id2rel.keys())
    rel_names = [dataset.id2rel[rid] for rid in rel_ids]
    print(f"Relations: {R}")

    # Load model
    print(f"Loading model '{args.model}' …")
    model = load_model(args.model, args.dataset)

    # Extract embeddings
    print("Extracting relation embeddings …")
    embeddings = get_relation_embeddings(args.model, model)
    print(f"  Embedding matrix shape: {tuple(embeddings.shape)}")

    # Compute cosine similarity
    print(f"Computing {R}×{R} cosine-similarity matrix …")
    sim_matrix = build_cosine_similarity_table(embeddings)

    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = f'{kg_name}_{args.model}_cosine'

    npy_path = os.path.join(args.output_dir, base_name + '.npy')
    np.save(npy_path, sim_matrix)
    print(f"\nSaved numpy array → {npy_path}")

    df = pd.DataFrame(sim_matrix, index=rel_names, columns=rel_names)
    csv_path = os.path.join(args.output_dir, base_name + '.csv')
    df.to_csv(csv_path)
    print(f"Saved CSV table   → {csv_path}")


if __name__ == '__main__':
    main()
