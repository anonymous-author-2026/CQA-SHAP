import random
from query import Query
from utils import get_num_atoms
import math
import torch
import numpy as np
np.random.seed(42)

def get_rank(tensor, target):
    return (tensor == target).nonzero(as_tuple=True)[0][0].item() + 1

class Shapley:
    def __init__(self, model, all_relations: list[int], predicate_cache: dict = {}, execution_cache: dict = {},
                 relation_selection: str = 'random', id2rel: dict = None, similarity_df=None, top_r: int = 10):
        """
        Args:
            model: an instance of CQD or KGR with a .predict() and .get_relation_similarities() method
            all_relations: list of all relation ids (integers)
            predicate_cache: dictionary to cache relation selections across calls
            execution_cache: dictionary to cache query execution results for each coalition
            relation_selection: 'random' (same behaviour as shapley.py) or 'closest'
                                (uses precomputed similarity-ranked neighbours instead of random
                                 substitutes; the i-th closest neighbour is used at iteration i)
            id2rel: dict mapping relation id (int) -> relation name (str). Required when
                    similarity_df is provided.
            similarity_df: pandas DataFrame with relation names as both index and columns,
                           values are similarity scores (as produced by
                           relation_similarity/relation_similarity.py). When provided and
                           relation_selection == 'closest', this is used instead of calling
                           model.get_relation_similarities().
            top_r: number of top similar relations to precompute per source relation
                   (excluding the relation itself). The iteration_index cycles through these.
        """
        self.model = model
        self.all_relations = all_relations
        self.predicate_cache = predicate_cache
        self.execution_cache = execution_cache
        self._filtered_nodes_cache: dict = {}
        self._empty_rank_cache: dict = {}

        self.relation_selection = relation_selection
        # index into the sorted-by-similarity list used by closest_relation_except;
        # should be updated via set_iteration_index() before each evaluation iteration
        self.iteration_index: int = 0

        # Precompute full similarity cache for all relations upfront when using
        # 'predicted' or 'cosine' mode, so that no similarity calls happen during
        # the Shapley computation itself.
        # For 'random' mode the cache stays empty (it is never consulted).
        if relation_selection in ('score', 'cosine'):
            if similarity_df is not None:
                if id2rel is None:
                    raise ValueError("id2rel must be provided when similarity_df is given.")
                rel2id = {v: k for k, v in id2rel.items()}
                print(f"Building relation similarity cache from precomputed DataFrame "
                      f"(top_r={top_r}) for {len(all_relations)} relations …")
                self._similarity_cache: dict = {}
                for rel_id in all_relations:
                    rel_name = id2rel[rel_id]
                    if rel_name not in similarity_df.index:
                        self._similarity_cache[rel_id] = []
                        print(f"Warning: relation '{rel_name}' (id {rel_id}) not found in similarity_df index; "
                              f"no candidates will be available for this relation.")
                        continue
                    row = similarity_df.loc[rel_name]
                    # Sort columns by score descending, exclude the relation itself
                    sorted_names = row.sort_values(ascending=False).index.tolist()
                    candidates = [
                        rel2id[name] for name in sorted_names
                        if name != rel_name and name in rel2id
                    ][:top_r]
                    self._similarity_cache[rel_id] = candidates
                print("Similarity cache ready.")
            else:
                print(f"Precomputing relation similarity cache for {len(all_relations)} relations …")
                self._similarity_cache: dict = {
                    rel: model.get_relation_similarities(rel) for rel in all_relations
                }
                print("Similarity cache ready.")
        else:
            self._similarity_cache: dict = {}

    # ------------------------------------------------------------------
    # iteration index (only relevant for relation_selection='closest')
    # ------------------------------------------------------------------

    def set_iteration_index(self, idx: int):
        """Set which closest neighbour to pick during the current iteration.

        Call this at the start of each outer iteration before resetting the
        predicate cache so that each iteration draws a different neighbour.
        """
        self.iteration_index = idx

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------

    def reset_predicate_cache(self):
        self.predicate_cache = {}
        self._empty_rank_cache = {}

    def set_predicate_cache(self, predicate_cache: dict):
        self.predicate_cache = predicate_cache
        self._empty_rank_cache = {}

    def reset_execution_cache(self):
        self.execution_cache = {}
        self._empty_rank_cache = {}

    def set_execution_cache(self, execution_cache: dict):
        self.execution_cache = execution_cache
        self._empty_rank_cache = {}

    # ------------------------------------------------------------------
    # relation selection
    # ------------------------------------------------------------------

    def random_relation_except(self, forbidden: int, selected_atom: int = None) -> int:
        '''
        Select a random relation from all_relations except the forbidden one.
        Result is cached so that the same atom always gets the same substitute
        within a single predicate-cache lifetime (i.e. one iteration).

        Args:
            forbidden: the relation id to avoid
            selected_atom: the atom index (used as part of the cache key)
        Returns:
            new_predicate: a random relation id different from forbidden
        '''
        if (selected_atom, forbidden) in self.predicate_cache:
            return self.predicate_cache[(selected_atom, forbidden)]
        flag = True
        while flag:
            new_predicate = random.choice(self.all_relations)
            if new_predicate != forbidden:
                flag = False
        self.predicate_cache[(selected_atom, forbidden)] = new_predicate
        return new_predicate

    def closest_relation_except(self, forbidden: int, selected_atom: int = None) -> int:
        '''
        Select the iteration_index-th most similar relation to *forbidden*,
        excluding forbidden itself.

        When a precomputed similarity_df was provided at construction time the
        cache already contains the top-R candidate relation IDs (as a plain
        list).  Otherwise (legacy path) the cache contains (rel_id, score)
        tuples from model.get_relation_similarities() and candidates are
        filtered on the fly.

        The predicate_cache is used to ensure the same choice is returned
        consistently within one iteration.

        Args:
            forbidden: the original relation id whose substitute we need
            selected_atom: atom index (cache key component)
        Returns:
            new_predicate: the selected relation id
        '''
        if (selected_atom, forbidden) in self.predicate_cache:
            return self.predicate_cache[(selected_atom, forbidden)]

        cached = self._similarity_cache[forbidden]

        # New path: cache already holds a plain list of candidate rel_ids
        if cached and not isinstance(cached[0], tuple):
            candidates = cached
        else:
            # Legacy path: cache holds (rel_id, score) tuples
            candidates = [r for r, _ in cached if r != forbidden]

        if not candidates:
            raise ValueError(f"No alternative relations found for relation {forbidden}")

        # Use modulo so we never exceed the list length
        idx = self.iteration_index % len(candidates)
        new_predicate = candidates[idx]

        self.predicate_cache[(selected_atom, forbidden)] = new_predicate
        return new_predicate

    def select_relation_except(self, forbidden: int, selected_atom: int = None) -> int:
        '''Dispatch to random_relation_except or closest_relation_except based on
        self.relation_selection.'''
        if self.relation_selection in ('score', 'cosine'):
            return self.closest_relation_except(forbidden, selected_atom)
        return self.random_relation_except(forbidden, selected_atom)

    # ------------------------------------------------------------------
    # query execution and value function
    # ------------------------------------------------------------------

    def query_execution(self, query: Query, coalition: list = [None]) -> dict:
        '''
        Execute the query on the model based on the coalition.
        Atoms not in the coalition have their predicates replaced using
        select_relation_except (random or closest depending on self.relation_selection).

        Args:
            query: Query object
            coalition: binary list – 1 means atom is in the coalition, 0 means it is masked
        Returns:
            result dict with keys 'ids' and 'scores'
        '''
        coalition_key = tuple(coalition)
        cached = self.execution_cache.get(coalition_key)
        if cached is not None:
            return cached

        if coalition is None:
            print("No coalition provided. Executing original query.")
            coalition_key = tuple([1] * get_num_atoms(query.query_type))
            prediction = self.model.predict(query)
        elif len(coalition) != get_num_atoms(query.query_type):
            raise ValueError(
                "Coalition length must match the number of atoms in the query "
                "({} for query type {})".format(get_num_atoms(query.query_type), query.query_type)
            )
        else:
            new_query = query.copy()
            for atom_index in range(len(coalition)):
                if coalition[atom_index] == 0:
                    predicate = new_query.atoms[atom_index]['relation']
                    new_predicate = self.select_relation_except(predicate, selected_atom=atom_index)
                    new_query.change_predicate(atom_index=atom_index, new_relation=new_predicate)
            prediction = self.model.predict(new_query)

        self.execution_cache[coalition_key] = prediction
        return prediction

    def value_function(self, query: Query, filtered_nodes: list, target_entity: int,
                       qoi: str = 'rank', coalition: list = None) -> float:
        '''
        Calculate the value function for a given coalition.

        Args:
            query: Query object
            filtered_nodes: list of other correct answers to filter out before ranking
            target_entity: the entity whose rank/hit we evaluate
            qoi: 'rank', 'hit1', 'hit3', or 'hit10'
            coalition: binary list for the coalition
        Returns:
            value relative to the empty-coalition baseline (positive = improvement)
        '''
        result = self.query_execution(query, coalition=coalition)['ids']

        fn_key = (tuple(filtered_nodes), str(result.device))
        if fn_key not in self._filtered_nodes_cache:
            self._filtered_nodes_cache[fn_key] = torch.tensor(filtered_nodes, device=result.device)
        filtered_tensor = self._filtered_nodes_cache[fn_key]

        empty_key = (tuple(filtered_nodes), target_entity)
        if empty_key not in self._empty_rank_cache:
            empty_ids = self.query_execution(query, coalition=[0] * len(coalition))['ids']
            empty_filtered = empty_ids[~torch.isin(empty_ids, filtered_tensor)]
            self._empty_rank_cache[empty_key] = get_rank(empty_filtered, target_entity)
        empty_rank = self._empty_rank_cache[empty_key]

        result = result[~torch.isin(result, filtered_tensor)]
        if qoi == 'rank':
            try:
                value = get_rank(result, target_entity)
            except IndexError:
                raise ValueError(f"Target entity {target_entity} not found in the result")
        elif qoi == 'hit1':
            value = 1 if target_entity in result[:1] else 0
        elif qoi == 'hit3':
            value = 1 if target_entity in result[:3] else 0
        elif qoi == 'hit10':
            value = 1 if target_entity in result[:10] else 0
        else:
            raise ValueError(f"Unsupported QoI: {qoi}. Supported: 'rank', 'hit1', 'hit3', 'hit10'.")

        value = -value + empty_rank
        return value

    # ------------------------------------------------------------------
    # Shapley value computation
    # ------------------------------------------------------------------

    def shapley_value(self, query: Query, atom_idx: int, filtered_nodes: list,
                      target_entity: int, qoi: str = 'rank', logging: bool = True):
        '''
        Calculate the Shapley value for a single atom.

        Args:
            query: Query object
            atom_idx: 0-based index of the atom to explain
            filtered_nodes: other correct answers to filter out
            target_entity: target entity id
            qoi: quantity of interest
            logging: whether to print per-coalition details
        '''
        num_atoms = get_num_atoms(query.query_type)
        shapley_value = 0.0

        num_remaining_atoms = num_atoms - 1
        for i in range(2 ** num_remaining_atoms):
            coalition = [int(x) for x in bin(i)[2:].zfill(num_remaining_atoms)]
            coalition_mask = [0] * num_atoms
            counter = 0
            for idx, _ in enumerate(coalition_mask):
                if idx == atom_idx:
                    coalition_mask[idx] = 0
                else:
                    coalition_mask[idx] = coalition[counter]
                    counter += 1
            if logging:
                print(f"Coalition: {coalition_mask}, Atom Index: {atom_idx}")

            weight = (math.factorial(sum(coalition)) * math.factorial(num_atoms - sum(coalition) - 1)) / math.factorial(num_atoms)

            value = self.value_function(query, filtered_nodes, target_entity, qoi=qoi, coalition=coalition_mask)

            added_coalition_mask = coalition_mask.copy()
            added_coalition_mask[atom_idx] = 1
            added_value = self.value_function(query, filtered_nodes, target_entity, qoi=qoi, coalition=added_coalition_mask)

            contribution = added_value - value
            if logging:
                print(f"Coalition: {coalition_mask}, Contribution: {contribution} "
                      f"(before: {value}, after: {added_value}), weight: {weight})")

            shapley_value += contribution * weight

        if logging:
            print(f"Shapley value for atom {atom_idx}: {shapley_value}")
        return shapley_value

    def shapley_values(self, query: Query, filtered_nodes: list, target_entity: int,
                       qoi: str = 'rank', logging: bool = True):
        num_atoms = get_num_atoms(query.query_type)
        shapley_values = []
        for atom_idx in range(num_atoms):
            sv = self.shapley_value(query, atom_idx, filtered_nodes, target_entity,
                                    qoi=qoi, logging=logging)
            shapley_values.append(sv)
        return shapley_values

    def shapley_values_batch(self, query: Query, filtered_nodes: list, target_entity: int,
                             qoi: str = 'rank', logging: bool = True, repeats: int = 1):
        shapeley_vals = []
        for _ in range(repeats):
            self.predicate_cache = {}
            svs = self.shapley_values(query, filtered_nodes, target_entity, qoi=qoi, logging=logging)
            shapeley_vals.append(svs)
        shapeley_vals = np.array(shapeley_vals)
        mean_svs = np.mean(shapeley_vals, axis=0)
        return mean_svs.tolist()

    # ------------------------------------------------------------------
    # reporting helpers
    # ------------------------------------------------------------------

    def report_all_coalitions(self, query: Query, filtered_nodes: list, target_entity: int):
        num_atoms = get_num_atoms(query.query_type)
        for i in range(2 ** num_atoms):
            coalition = [int(x) for x in bin(i)[2:].zfill(num_atoms)]
            predictions = self.query_execution(query, coalition=coalition)['ids']
            filtered_predictions = [pred for pred in predictions.tolist() if pred not in filtered_nodes]
            rank = (torch.tensor(filtered_predictions) == target_entity).nonzero(as_tuple=True)[0].item() + 1
            print(f"Coalition: {coalition}, Rank of target entity {target_entity}: {rank}")

    def report_one_coalition(self, query: Query, coalition: list, filtered_nodes: list, target_entity: int):
        predictions = self.query_execution(query, coalition=coalition)['ids']
        fn_key = (tuple(filtered_nodes), str(predictions.device))
        if fn_key not in self._filtered_nodes_cache:
            self._filtered_nodes_cache[fn_key] = torch.tensor(filtered_nodes, device=predictions.device)
        filtered_tensor = self._filtered_nodes_cache[fn_key]
        filtered_predictions = predictions[~torch.isin(predictions, filtered_tensor)]
        rank = (filtered_predictions == target_entity).nonzero(as_tuple=True)[0].item() + 1
        return rank
