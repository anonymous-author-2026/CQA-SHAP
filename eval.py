from utils import setup_dataset_and_graphs, load_all_queries, get_query_types
import argparse
import pandas as pd
from models.KGReasoning.predict import KGR
import sys
sys.path.insert(0, 'models/CQD')
from models.CQD.predict import CQD
import time
from utils import get_num_atoms
import numpy as np
from tqdm import tqdm
import os
from shapley import Shapley
import random
random.seed(42)
from utils import get_first, get_last

def choose_atom(query, method):
    if method == "random":
        num_atoms = get_num_atoms(query.query_type)
        return random.randint(0, num_atoms - 1), None
    elif method == "first":
        return random.choice(get_first(query.query_type)), None
    elif method == "last":
        return random.choice(get_last(query.query_type)), None
    else:
        raise ValueError(f"Unknown method: {method}")

def evaluation_per_type_and_model(query_type: str, queries_hard: list, queries_complete: list,
                                   iterations: int = 5, shapley=None):
    """
    Same evaluation loop as eval.py but compatible with shapley2.Shapley.

    When shapley.relation_selection == 'closest', shapley.set_iteration_index(itercount)
    is called before each iteration so that each iteration uses the next-closest
    relation neighbour as the substitute for masked atoms.
    """
    num_atoms = get_num_atoms(query_type)
    rows = []

    for query_idx, query_sample_hard in tqdm(enumerate(queries_hard), total=len(queries_hard),
                                              desc="Queries", position=0):

        answers = query_sample_hard.get_answer()
        query_sample_complete = queries_complete[query_idx]
        all_answers = set(query_sample_complete.get_answer())

        for itercount in tqdm(range(iterations), desc="Iterations", position=1, leave=False):

            # For 'closest' mode: tell Shapley which neighbour rank to use this iteration.
            # For 'random' mode: this call is a no-op (index is not consulted).
            shapley.set_iteration_index(itercount)

            shapley.reset_execution_cache()
            shapley.reset_predicate_cache()

            for answer_idx, target_entity in tqdm(enumerate(answers), total=len(answers),
                                                   desc="Answers", position=2, leave=False):

                filtered_answers = list(all_answers - {target_entity})

                start_time = time.time()
                shapley_vals = shapley.shapley_values(
                    query_sample_hard,
                    filtered_nodes=filtered_answers,
                    target_entity=target_entity,
                    qoi='rank',
                    logging=False)
                time_taken = time.time() - start_time

                best_atoms = {
                    'shapley': int(np.argmax(shapley_vals)),
                    'random':  choose_atom(query_sample_hard, "random")[0],
                    'first':   choose_atom(query_sample_hard, "first")[0],
                    'last':    choose_atom(query_sample_hard, "last")[0],
                }

                rank_empty = shapley.report_one_coalition(
                    query_sample_hard,
                    filtered_nodes=filtered_answers,
                    target_entity=target_entity,
                    coalition=[0] * num_atoms
                )

                rank_full = shapley.report_one_coalition(
                    query_sample_hard,
                    filtered_nodes=filtered_answers,
                    target_entity=target_entity,
                    coalition=[1] * num_atoms
                )

                new_relations = list(shapley.predicate_cache.values())

                row = {
                    'type': query_type,
                    'query': query_idx,
                    'target': target_entity,
                    'iteration': itercount,
                    'relation_selection': shapley.relation_selection,
                    'rel1': new_relations[0],
                    'rel2': new_relations[1],
                    'time': time_taken,
                    'rank_empty': rank_empty,
                    'rank_full': rank_full,
                    'shapley0': shapley_vals[0],
                    'shapley1': shapley_vals[1],
                }

                if num_atoms >= 3:
                    row['rel3'] = new_relations[2]
                    row['shapley2'] = shapley_vals[2]
                if num_atoms == 4:
                    row['rel4'] = new_relations[3]
                    row['shapley3'] = shapley_vals[3]

                # Necessity/sufficiency rank for EVERY atom, not just each
                # method's chosen "best" one. This is free: by this point
                # shapley.shapley_values() has already executed every one of
                # the 2**num_atoms coalitions this iteration, so
                # report_one_coalition below is always a cache hit (see
                # Shapley.query_execution). We need every atom's value so
                # that, once all K iterations are collected below, the
                # "shapley" method can be fixed to consistently use the atom
                # with the highest *average* Shapley value (Eq.
                # averageshapley) instead of each iteration re-picking its
                # own argmax independently, which can disagree between
                # iterations when Shapley values are close/tied.
                rank_necc_by_atom, rank_suff_by_atom = {}, {}
                for atom_idx in range(num_atoms):
                    coalition_necc = [1] * num_atoms
                    coalition_necc[atom_idx] = 0
                    coalition_suff = [0] * num_atoms
                    coalition_suff[atom_idx] = 1
                    rank_necc_by_atom[atom_idx] = shapley.report_one_coalition(
                        query_sample_hard,
                        filtered_nodes=filtered_answers,
                        target_entity=target_entity,
                        coalition=coalition_necc
                    )
                    rank_suff_by_atom[atom_idx] = shapley.report_one_coalition(
                        query_sample_hard,
                        filtered_nodes=filtered_answers,
                        target_entity=target_entity,
                        coalition=coalition_suff
                    )
                    row[f'rank_necc_atom{atom_idx}'] = rank_necc_by_atom[atom_idx]
                    row[f'rank_suff_atom{atom_idx}'] = rank_suff_by_atom[atom_idx]

                for method_name, best_atom in best_atoms.items():
                    row[f'best_atom_{method_name}'] = best_atom
                    row[f'rank_necc_{method_name}'] = rank_necc_by_atom[best_atom]
                    row[f'rank_suff_{method_name}'] = rank_suff_by_atom[best_atom]

                rows.append(row)

    df = pd.DataFrame(rows)

    # Fix up the "shapley" method so best_atom_shapley (and its necessity/
    # sufficiency ranks) are the SAME atom across all K iterations of a given
    # (type, query, target): pick the atom with the highest *average* Shapley
    # value over all iterations (Eq. averageshapley), instead of leaving each
    # iteration's own independent argmax in place.
    shapley_cols = sorted(c for c in df.columns if c.startswith('shapley') and c[len('shapley'):].isdigit())
    for _, group in df.groupby(['type', 'query', 'target']):
        a_star = int(group[shapley_cols].mean().to_numpy().argmax())
        idx = group.index
        df.loc[idx, 'best_atom_shapley'] = a_star
        df.loc[idx, 'rank_necc_shapley'] = df.loc[idx, f'rank_necc_atom{a_star}']
        df.loc[idx, 'rank_suff_shapley'] = df.loc[idx, f'rank_suff_atom{a_star}']

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate necessary/sufficient explanations (closest-relation variant)')
    parser.add_argument('--dataset', type=str, default="FB", help='Dataset name: FB or NELL')
    parser.add_argument('--model_name', type=str, default="all",
                        help='Model name: query2box, gqe, betae, cqd, or "all"')
    parser.add_argument('--query_type', type=str, default="2p", nargs='+',
                        help='Query type(s): 2p, 3p, 4p, 2i, 3i, 4i, ip, pi, 2u, up. '
                             'Accepts multiple values (e.g. 2p 3p) or "all".')
    parser.add_argument('--output_dir', type=str, default="evaluations",
                        help='Output directory to save results')
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of iterations. For closest mode each iteration uses a '
                             'different similarity-ranked neighbour relation.')
    parser.add_argument('--benchmark', type=int, default=2,
                        help='Benchmark type: 1 for CQD benchmark, 2 for Hard benchmark')
    parser.add_argument('--relation_selection', type=str, default='random',
                        choices=['random', 'score', 'cosine'],
                        help='How to choose substitute relations for masked atoms: '
                             '"random" picks a random substitute relation; '
                             '"score" picks neighbours ranked by prediction-score similarity '
                             '(files from relation_similarity/results/); '
                             '"cosine" picks neighbours ranked by embedding cosine similarity '
                             '(files from relation_similarity/cosine_results/).')
    parser.add_argument('--similarity_file', type=str, default=None,
                        help='Path to a precomputed relation-similarity CSV file. '
                             'Auto-detected from the appropriate results directory when omitted.')
    parser.add_argument('--top_r', type=int, default=3,
                        help='Number of top similar relations to use as substitutes '
                             'per source relation (excluding the relation itself). '
                             'The i-th iteration uses the i-th closest neighbour.')
    args = parser.parse_args()

    if args.relation_selection in ('score', 'cosine') and args.iterations != args.top_r:
        print(f"Setting --iterations to --top_r ({args.top_r}) for '{args.relation_selection}' mode.")
        args.iterations = args.top_r

    if args.dataset == "FB":
        data_dir = "data/FB15k-237+H" if args.benchmark == 2 else "data/FB15k-237"
    elif args.dataset == "NELL":
        data_dir = "data/NELL995+H" if args.benchmark == 2 else "data/NELL995"
    else:
        raise ValueError("Unsupported dataset. Choose either 'FB' or 'NELL'.")

    dataset, graph_train, graph_valid, graph_test = setup_dataset_and_graphs(
        data_dir, logging=True, add_reverse=(args.benchmark == 1))
    print("Dataset and Graphs are set up.")
    query_dataset, query_dataset_hard = load_all_queries(dataset, data_dir, "test", version=args.benchmark)
    print("Queries are loaded.")
    all_relations = list(dataset.id2rel.keys())
    print(f"There are {len(all_relations)} relations in the dataset.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    valid_query_types = get_query_types(version=args.benchmark)
    valid_model_names = ['query2box', 'gqe', 'betae', 'cqd']

    # Resolve model names: "all" expands to every valid model
    if args.model_name == 'all':
        model_names = valid_model_names
    else:
        assert args.model_name in valid_model_names, \
            f"Unknown model '{args.model_name}'. Valid: {valid_model_names}"
        model_names = [args.model_name]

    # Resolve query types: "all" expands to every valid type for this benchmark
    raw_types = args.query_type
    if len(raw_types) == 1 and raw_types[0] == 'all':
        query_types = valid_query_types
        # TODO: ignoring 4p and 4i for now
        query_types = [qt for qt in query_types if qt not in ['4p', '4i']]
    else:
        query_types = raw_types
        for qt in query_types:
            assert qt in valid_query_types, f"Unknown query type '{qt}'. Valid: {valid_query_types}"

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")

        if model_name == 'cqd':
            if args.dataset == "FB":
                model = CQD(
                    model_path="models/CQD/models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt",
                    data_dir="models/CQD/data/FB15k-237",
                    k=5, t_norm='prod', t_conorm='prod', normalize=False
                )
            elif args.dataset == "NELL":
                model = CQD(
                    model_path="models/CQD/models/NELL-model-rank-1000-epoch-100-1602499096.pt",
                    data_dir="models/CQD/data/NELL",
                    k=5, t_norm='prod', t_conorm='prod', normalize=False
                )
        else:
            model = KGR(method=model_name, dataset=args.dataset)

        similarity_df = None
        if args.relation_selection in ('score', 'cosine'):
            sim_file = args.similarity_file
            if sim_file is None:
                kg_name = "FB15k-237" if args.dataset == "FB" else "NELL995"
                if args.relation_selection == 'cosine':
                    sim_dir = "relation_similarity/cosine_results"
                    exact_name = f"{kg_name}_{model_name}_cosine.csv"
                    sim_file = os.path.join(sim_dir, exact_name)
                    if not os.path.exists(sim_file):
                        raise FileNotFoundError(
                            f"No cosine similarity file found at '{sim_file}'. "
                            f"Run relation_similarity/cosine_similarity.py first, or pass --similarity_file."
                        )
                else:  # 'score'
                    sim_dir = "relation_similarity/results"
                    pattern = f"{kg_name}_{model_name}_"
                    matches = [f for f in os.listdir(sim_dir)
                               if f.startswith(pattern) and f.endswith(".csv")]
                    if len(matches) == 0:
                        raise FileNotFoundError(
                            f"No similarity file found in '{sim_dir}' matching '{pattern}*.csv'. "
                            f"Run relation_similarity/relation_similarity.py first, or pass --similarity_file."
                        )
                    if len(matches) > 1:
                        raise ValueError(
                            f"Multiple similarity files found for {args.dataset}/{model_name}: "
                            f"{matches}. Use --similarity_file to specify one."
                        )
                    sim_file = os.path.join(sim_dir, matches[0])
            print(f"Loading relation similarity from: {sim_file}")
            similarity_df = pd.read_csv(sim_file, index_col=0)

        shapley = Shapley(
            model=model,
            all_relations=all_relations,
            predicate_cache={},
            execution_cache={},
            relation_selection=args.relation_selection,
            id2rel=dataset.id2rel,
            similarity_df=similarity_df,
            top_r=args.top_r,
        )

        for query_type in query_types:
            print(f"\n=== Processing query type: {query_type} ===")

            file_name = (f"evaluation_{args.dataset}_{args.benchmark}_{model_name}"
                         f"_{query_type}_{args.relation_selection}.csv")
            output_path = os.path.join(args.output_dir, file_name)

            queries_hard = query_dataset_hard.get_queries(query_type)
            queries_complete = query_dataset.get_queries(query_type)

            df = evaluation_per_type_and_model(
                query_type, queries_hard, queries_complete,
                iterations=args.iterations, shapley=shapley)
            df['model'] = model_name

            print(f"Saving results to {output_path}")
            df.to_csv(output_path, index=False)
            print(f"Done with query type: {query_type}")

        print(f"\nAll query types completed for model: {model_name}")

    print("\nAll models completed.")
