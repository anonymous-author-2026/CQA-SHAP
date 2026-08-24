import os.path as osp
import pickle
from kbc.learn import kbc_model_load
from kbc.utils import QuerDAG
from kbc.utils import preload_env
import torch
from query import Query

def query_dict_generator(query_sample: Query):
    query = query_sample.query
    answers = query_sample.answer
    if query_sample.query_type == "1p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relation1 = atom1[1][0]
        # TYPE1_1: preload_env reads raw_chain as [anchor, relation, answers] directly
        # (flat, not a list of hops like the multi-hop cases)
        query_dict = {'raw_chain': [anchor1, relation1, answers],
                    'anchors': [anchor1],
                    'optimisable': answers,
                    'type': '1chain1',
                    'targets': answers}
    elif query_sample.query_type == "2p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_dict = {'raw_chain': [[anchor1, relations1[0], -1],
                                    [-1, relations1[1], answers]],
                    'anchors': [anchor1],
                    'optimisable': [-1] + answers,
                    'type': '1chain2',
                    'targets': answers}
    elif query_sample.query_type == "3p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_dict = {'raw_chain': [[anchor1, relations1[0], -1],
                                    [-1, relations1[1], -2],
                                    [-2, relations1[2], answers]],
                    'anchors': [anchor1],
                    'optimisable': [-1, -2] + answers,
                    'type': '1chain3',
                    'targets': answers}
    elif query_sample.query_type == "4p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_dict = {'raw_chain': [[anchor1, relations1[0], -1],
                                    [-1, relations1[1], -2],
                                    [-2, relations1[2], -3],
                                    [-3, relations1[3], answers]],
                    'anchors': [anchor1],
                    'optimisable': [-1, -2, -3] + answers,
                    'type': '1chain4',
                    'targets': answers}
    elif query_sample.query_type == "2i":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        query_dict = {'raw_chain': [[anchor1, relation1[0], answers],
                                    [anchor2, relation2[0], answers]],
                    'anchors': [anchor1, anchor2],
                    'optimisable': answers,
                    'type': '2chain2',
                    'targets': answers}
    elif query_sample.query_type == "3i":
        atom1 = query[0]
        atom2 = query[1]
        atom3 = query[2]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        anchor3 = atom3[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        relation3 = atom3[1]
        query_dict = {'raw_chain': [[anchor1, relation1[0], answers],
                                    [anchor2, relation2[0], answers],
                                    [anchor3, relation3[0], answers]],
                    'anchors': [anchor1, anchor2, anchor3],
                    'optimisable': answers,
                    'type': '2chain3',
                    'targets': answers}
    elif query_sample.query_type == "4i":
        atom1 = query[0]
        atom2 = query[1]
        atom3 = query[2]
        atom4 = query[3]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        anchor3 = atom3[0]
        anchor4 = atom4[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        relation3 = atom3[1]
        relation4 = atom4[1]
        query_dict = {'raw_chain': [[anchor1, relation1[0], answers],
                                    [anchor2, relation2[0], answers],
                                    [anchor3, relation3[0], answers],
                                    [anchor4, relation4[0], answers]],
                    'anchors': [anchor1, anchor2, anchor3, anchor4],
                    'optimisable': answers,
                    'type': '2chain4',
                    'targets': answers}
    elif query_sample.query_type == "2u":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        query_dict = {'raw_chain': [[anchor1, relation1[0], answers],
                                    [anchor2, relation2[0], answers]],
                    'anchors': [anchor1, anchor2],
                    'optimisable': answers,
                    'type': '2chain2_disj',
                    'targets': answers}
    elif query_sample.query_type == "up":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        query_dict = {'raw_chain': [[anchor1, relation1, -1],
                                    [anchor2, relation2, -1],
                                    [-1, relation3, answers]],
                    'anchors': [anchor1, anchor2],
                    'optimisable': [-1] + answers,
                    'type': '4chain3_disj',
                    'targets': answers}
        # print(query_dict)
    elif query_sample.query_type == "pi":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relations1 = atom1[1]
        relations2 = atom2[1]
        query_dict = {'raw_chain': [[anchor1, relations1[0], -1],
                                    [-1, relations1[1], answers],
                                    [anchor2, relations2[0], answers]],
                    'anchors': [anchor1, anchor2],
                    'optimisable': [-1] + answers,
                    'type': '3chain3',
                    'targets': answers}
    elif query_sample.query_type == "ip":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        query_dict = {'raw_chain': [[anchor1, relation1, -1],
                                    [anchor2, relation2, -1],
                                    [-1, relation3, answers]],
                    'anchors': [anchor1, anchor2],
                    'optimisable': [-1] + answers,
                    'type': '4chain3',
                    'targets': answers}
    return query_dict

class CQD:
    def __init__(self, model_path: str = "models/FB15k-237-model-rank-1000-epoch-100-1602508358.pt"
            , data_dir: str = "data/FB15k-237", k: int = 10, t_norm: str = 'prod', t_conorm: str = 'prod', normalize: bool = False):
        
        self.model_path = model_path
        self.data_dir = data_dir
        self.k = k
        self.t_norm = t_norm
        self.t_conorm = t_conorm
        self.normalize = normalize
        
        self.kbc, _, _ = kbc_model_load(model_path)
        
        self.chain_types = {'1p': QuerDAG.TYPE1_1.value,
                            '2p': QuerDAG.TYPE1_2.value,
                            '3p': QuerDAG.TYPE1_3.value,
                            '4p': QuerDAG.TYPE1_4.value,
                            '2i': QuerDAG.TYPE2_2.value,
                            '3i': QuerDAG.TYPE2_3.value,
                            '4i': QuerDAG.TYPE2_4.value,
                            'pi': QuerDAG.TYPE3_3.value,
                            'ip': QuerDAG.TYPE4_3.value,
                            '2u': QuerDAG.TYPE2_2_disj.value,
                            'up': QuerDAG.TYPE4_3_disj.value}

        self.chain_attr = {'1p':  'type1_1chain',
                            '2p':  'type1_2chain',
                            '3p':  'type1_3chain',
                            '4p':  'type1_4chain',
                            '2i':  'type2_2chain',
                            '3i':  'type2_3chain',
                            '4i':  'type2_4chain',
                            'pi':  'type3_3chain',
                            'ip':  'type4_3chain',
                            '2u':  'type2_2_disj_chain',
                            'up':  'type4_3_disj_chain'}
        
    def write_query(self, query: Query):
        file_name = "query_sample_temp.pkl"
        file_path = osp.join(self.data_dir, file_name)
        
        dataset = pickle.load(open(file_path, 'rb'))
        query_dict = query_dict_generator(query)
        getattr(dataset, self.chain_attr[query.query_type])[0].data = query_dict
        new_file_path = file_path.replace('_temp', '')
        with open(new_file_path, 'wb') as f:
            pickle.dump(dataset, f)
            
    def get_relation_similarities(self, relation_id: int) -> list:
        """
        Compute cosine similarity between the given relation's embedding and all other
        relation embeddings. Uses the kbc ComplEx model's relation embedding matrix
        (self.kbc.model.embeddings[1]).

        Args:
            relation_id: integer id of the query relation
        Returns:
            sorted list of (relation_id, similarity_score) tuples, descending by similarity
        """
        import torch.nn.functional as F
        rel_emb = self.kbc.model.embeddings[1].weight.data  # (nrelation, dim)
        query_emb = rel_emb[relation_id].unsqueeze(0)       # (1, dim)

        norms = F.normalize(rel_emb, p=2, dim=1)
        query_norm = F.normalize(query_emb, p=2, dim=1)
        similarities = (norms @ query_norm.T).squeeze(1)    # (nrelation,)

        sorted_indices = torch.argsort(similarities, descending=True)
        return [(idx.item(), similarities[idx].item()) for idx in sorted_indices]

    def predict_1p_batch(self, head_id: int, rel_ids: list, chunk_size: int = None) -> torch.Tensor:
        """
        Score (head_id, r, ?) against all entities for every r in rel_ids in one forward pass.

        Args:
            head_id:  integer entity ID of the head entity
            rel_ids:  list of R relation IDs to evaluate simultaneously
        Returns:
            FloatTensor of shape [R, n_entities], scores[j, e] = score(head, rel_ids[j], e)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        R = len(rel_ids)
        head_tensor = torch.full((R,), head_id, dtype=torch.long, device=device)
        rel_tensor  = torch.tensor(rel_ids, dtype=torch.long, device=device)
        with torch.no_grad():
            lhs_emb = self.kbc.model.embeddings[0](head_tensor)   # [R, emb_dim]
            rel_emb = self.kbc.model.embeddings[1](rel_tensor)    # [R, emb_dim]
            scores  = self.kbc.model.forward_emb(lhs_emb, rel_emb)  # [R, n_entities]
            if self.normalize:
                scores = torch.sigmoid(scores)
        return scores

    def predict_1p(self, head_id: int, relation_id: int) -> dict:
        """
        Execute a single-hop (1p) link prediction query and return scores for all entities.

        Bypasses query_answering_BF (which requires ≥2 hops to build chain instructions)
        and scores directly via ComplEx forward_emb: score(h, r, ?) against all entities.
        Applies sigmoid normalization when self.normalize is True, matching the behaviour
        of query_answering_BF for non-disjunctive queries.

        Args:
            head_id: integer entity ID of the head entity
            relation_id: integer relation ID
        Returns:
            dict with 'scores' (tensor) and 'ids' (tensor), sorted descending by score
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_tensor = torch.tensor([[head_id, relation_id]], dtype=torch.long, device=device)
        with torch.no_grad():
            lhs_emb = self.kbc.model.embeddings[0](query_tensor[:, 0])
            rel_emb = self.kbc.model.embeddings[1](query_tensor[:, 1])
            scores = self.kbc.model.forward_emb(lhs_emb, rel_emb).squeeze(0)
            if self.normalize:
                scores = torch.sigmoid(scores)
        sorted_results = torch.sort(scores, descending=True)
        return {'scores': sorted_results.values, 'ids': sorted_results.indices}

    def predict(self, query: Query):
        # preparing query file
        self.write_query(query)
        file_name = "query_sample.pkl"
        dataset = pickle.load(open(osp.join(self.data_dir, file_name), 'rb'))
        qdag = self.chain_types[query.query_type]
        # loading the query
        env = preload_env(self.model_path, dataset, qdag, mode='hard', kg_path="data/FB15k-237/", explain=False, kbc=self.kbc)
        env = preload_env(self.model_path, dataset, qdag, mode='complete', explain=False, kbc=self.kbc)
        # do the prediction
        scores = self.kbc.model.query_answering_BF(env, self.k, t_norm=self.t_norm , batch_size=1, scores_normalize = 1 if self.normalize else 0, explain=False)
        score = scores[0]
        sortedanswers = torch.sort(score, descending=True)
        sortedscores = sortedanswers.values
        sortedindices = sortedanswers.indices
        return {'scores': sortedscores, 'ids': sortedindices}