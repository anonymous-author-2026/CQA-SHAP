from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os.path as osp
from query import Query
import json
import logging
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from .models import KGReasoning
from .dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from tensorboardX import SummaryWriter
import time
import pickle
from collections import defaultdict
from tqdm import tqdm
from .util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple, flatten
import collections

query_name_dict = {('e',('r',)): '1p',
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    ('e', ('r', 'r', 'r', 'r')): '4p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '4i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r', 'n'))): '2in',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r', 'n'))): '3in',
                    ((('e', ('r',)), ('e', ('r', 'n'))), ('r',)): 'inp',
                    (('e', ('r', 'r')), ('e', ('r', 'n'))): 'pin',
                    (('e', ('r', 'r', 'n')), ('e', ('r',))): 'pni',
                    (('e', ('r',)), ('e', ('r',)), ('u',)): '2u-DNF',
                    ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)): 'up-DNF',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n',)): '2u-DM',
                    ((('e', ('r', 'n')), ('e', ('r', 'n'))), ('n', 'r')): 'up-DM'
                }
name_query_dict = {value: key for key, value in query_name_dict.items()}
all_tasks = list(name_query_dict.keys())

def query_mapping(query_sample: Query):
    query = query_sample.query
    answers = query_sample.answer
    if query_sample.query_type == "1p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relation1 = atom1[1][0]
        query_mapped = ((anchor1, (relation1,)), ('e', ('r',)))
    elif query_sample.query_type == "2p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_mapped = ((anchor1, (relations1[0], relations1[1])), ('e', ('r', 'r')))
    elif query_sample.query_type == "3p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_mapped = ((anchor1, (relations1[0], relations1[1], relations1[2])), ('e', ('r', 'r', 'r')))
    elif query_sample.query_type == "4p":
        atom1 = query[0]
        anchor1 = atom1[0]
        relations1 = atom1[1]
        query_mapped = ((anchor1, (relations1[0], relations1[1], relations1[2], relations1[3])), ('e', ('r', 'r', 'r', 'r')))
    elif query_sample.query_type == "2i":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        query_mapped = (((anchor1, (relation1,)), (anchor2, (relation2,))), (('e', ('r',)), ('e', ('r',))))
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
        query_mapped = (((anchor1, (relation1,)), (anchor2, (relation2,)), (anchor3, (relation3,))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',))))
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
        query_mapped = (((anchor1, (relation1,)), (anchor2, (relation2,)), (anchor3, (relation3,)), (anchor4, (relation4,))), (('e', ('r',)), ('e', ('r',)), ('e', ('r',)), ('e', ('r',))))
    elif query_sample.query_type == "2u":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1]
        relation2 = atom2[1]
        query_mapped = (((anchor1, (relation1,)), (anchor2, (relation2,)), (-1,)), (('e', ('r',)), ('e', ('r',)), ('u',)))
    elif query_sample.query_type == "up":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        query_mapped = ((((anchor1, (relation1,)), (anchor2, (relation2,)), (-1,)), (relation3,)), ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',)))
    elif query_sample.query_type == "pi":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relations1 = atom1[1]
        relations2 = atom2[1]
        query_mapped = (((anchor1, (relations1[0], relations1[1])), (anchor2, (relations2[0],))), (('e', ('r', 'r')), ('e', ('r',))))
    elif query_sample.query_type == "ip":
        atom1 = query[0]
        atom2 = query[1]
        anchor1 = atom1[0]
        anchor2 = atom2[0]
        relation1 = atom1[1][0]
        relation2 = atom2[1][0]
        relation3 = query[2]
        query_mapped = ((((anchor1, (relation1,)), (anchor2, (relation2,))), (relation3,)), ((('e', ('r',)), ('e', ('r',))), ('r',)))
    return query_mapped

def test_step(model, args, test_dataloader):
    model.eval()
    with torch.no_grad():
        # counter = 0
        for negative_sample, queries, queries_unflatten, query_structures in test_dataloader:
            batch_queries_dict = collections.defaultdict(list)
            batch_idxs_dict = collections.defaultdict(list)
            for i, query in enumerate(queries):
                batch_queries_dict[query_structures[i]].append(query)
                batch_idxs_dict[query_structures[i]].append(i)
            for query_structure in batch_queries_dict:
                if args.cuda:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
                else:
                    batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure])
            if args.cuda:
                negative_sample = negative_sample.cuda()

            _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
            queries_unflatten = [queries_unflatten[i] for i in idxs]
            query_structures = [query_structures[i] for i in idxs]
            """
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)
            if len(argsort) == args.test_batch_size: # if it is the same shape with test_batch_size, we can reuse batch_entity_range without creating a new one
                ranking = ranking.scatter_(1, argsort, model.batch_entity_range) # achieve the ranking of all entities
            else: # otherwise, create a new torch Tensor for batch_entity_range
                if args.cuda:
                    ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).cuda()) # achieve the ranking of all entitie
                else:
                    ranking = ranking.scatter_(1, argsort, torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1)) # achieve the ranking of all entities
            """
            return negative_logit                    
        

def get_args():
    args = argparse.Namespace()
    args.cuda = True
    args.do_train = False
    args.do_valid = False
    args.do_test = True
    args.data_path = "models/KGReasoning/data/FB15k-237-betae" # KG data path
    args.negative_sample_size = 128 # negative_sample_size -> negative entities sampled per query
    args.hidden_dim = 500 # hidden_dim -> embedding dimension
    args.gamma = 12.0 # gamma -> margin in the loss
    args.batch_size = 1024 # batch_size -> batch size of queries
    args.test_batch_size = 1 # valid/test batch size
    args.learning_rate = 0.0001 # learning_rate
    args.cpu_num = 10 # used to speed up torch.dataloader
    args.save_path = None # no need to set manually, will configure automatically
    args.max_steps = 100000 # maximum iterations to train
    args.warm_up_steps = None # no need to set manually, will configure automatically
    args.save_checkpoint_steps = 50000 # save checkpoints every xx steps
    args.valid_steps = 10000 # evaluate validation queries every xx steps
    args.log_steps = 100 # train log every xx steps
    args.test_log_steps = 1000 # valid/test log every xx steps
    args.nentity = 0 # DO NOT MANUALLY SET
    args.nrelation = 0 # DO NOT MANUALLY SET
    args.geo = 'vec' # the reasoning model, vec for GQE, box for Query2box, beta for BetaE
    args.print_on_screen = True
    args.tasks = '1p.2p.3p.4p.2i.3i.4i.ip.pi.2in.3in.inp.pin.pni.2u.up' # tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task
    args.seed = 0 # random seed
    args.beta_mode = "(1600,2)" # (hidden_dim,num_layer) for BetaE relational projection
    args.box_mode = "(none,0.02)" # (offset activation,center_reg)
    args.prefix = None # prefix of the log path
    args.checkpoint_path = None # path for loading the checkpoints
    args.evaluate_union = "DNF" # the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan's laws (DM)
    return args

class KGR:
    def __init__(self, method: str = "query2box", dataset: str = "FB", args: argparse.Namespace = None):
        # initialize args
        self.args = get_args()
        # set dataset-specific paths
        if dataset == "NELL":
            kg_suffix = "NELL995"
            data_suffix_betae = "NELL-betae"
            data_suffix_q2b = "NELL-q2b"
        else:
            kg_suffix = "FB15k-237"
            data_suffix_betae = "FB15k-237-betae"
            data_suffix_q2b = "FB15k-237-q2b"
        if method == "query2box":
            self.args.negative_sample_size = 128
            self.args.batch_size = 512
            self.args.hidden_dim = 400
            self.args.gamma = 24
            self.args.learning_rate = 0.0001
            self.args.max_steps = 450001
            self.args.cpu_num = 1
            self.args.geo = 'box'
            self.args.valid_steps = 15000
            self.args.box_mode = "(none, 0.02)"
            self.args.tasks = "1p.2p.3p.2i.3i.ip.pi.2u.up"
            self.args.data_path = f"models/KGReasoning/data/{data_suffix_q2b}"
            self.args.checkpoint_path = f"models/KGReasoning/logs/{kg_suffix}/query2box/checkpoint"
        elif method == "betae":
            self.args.negative_sample_size = 128
            self.args.batch_size = 512
            self.args.hidden_dim = 400
            self.args.gamma = 60
            self.args.learning_rate = 0.0001
            self.args.max_steps = 450001
            self.args.cpu_num = 1
            self.args.geo = 'beta'
            self.args.valid_steps = 15000
            self.args.beta_mode = "(1600, 2)"
            self.args.data_path = f"models/KGReasoning/data/{data_suffix_betae}"
            self.args.checkpoint_path = f"models/KGReasoning/logs/{kg_suffix}/betae/checkpoint"
        elif method == "gqe":
            self.args.negative_sample_size = 128
            self.args.batch_size = 512
            self.args.hidden_dim = 800
            self.args.gamma = 24
            self.args.learning_rate = 0.0001
            self.args.max_steps = 450001
            self.args.cpu_num = 1
            self.args.geo = 'vec'
            self.args.valid_steps = 15000
            self.args.tasks = "1p.2p.3p.2i.3i.ip.pi.2u.up"
            self.args.data_path = f"models/KGReasoning/data/{data_suffix_q2b}"
            self.args.checkpoint_path = f"models/KGReasoning/logs/{kg_suffix}/gqe/checkpoint"
        # override args with the input args
        if args is not None:
            for k, v in vars(args).items():
                setattr(self.args, k, v)
        # read data stats and set nentity and nrelation
        with open('%s/stats.txt'%self.args.data_path) as f:
            self.entrel = f.readlines()
            self.args.nentity = int(self.entrel[0].split(' ')[-1])
            self.args.nrelation = int(self.entrel[1].split(' ')[-1])
            
        set_global_seed(self.args.seed)
        tasks = self.args.tasks.split('.')
        for task in tasks:
            if 'n' in task and self.args.geo in ['box', 'vec']:
                assert False, "Q2B and GQE cannot handle queries with negation"
        if self.args.evaluate_union == 'DM':
            assert self.args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"
            
        self.model = KGReasoning(
            nentity = self.args.nentity,
            nrelation = self.args.nrelation,
            hidden_dim = self.args.hidden_dim,
            gamma = self.args.gamma,
            geo = self.args.geo,
            use_cuda = self.args.cuda,
            box_mode = eval_tuple(self.args.box_mode),
            beta_mode = eval_tuple(self.args.beta_mode),
            test_batch_size = self.args.test_batch_size,
            query_name_dict = query_name_dict
        )
        
        if self.args.cuda:
            self.model = self.model.cuda()
            
        if self.args.checkpoint_path is not None:
            checkpoint = torch.load(os.path.join(self.args.checkpoint_path))
            init_step = checkpoint['step']
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def get_relation_similarities(self, relation_id: int) -> list:
        """
        Compute cosine similarity between the given relation's embedding and all other
        relation embeddings. Uses the main relation_embedding weights.

        Args:
            relation_id: integer id of the query relation
        Returns:
            sorted list of (relation_id, similarity_score) tuples, descending by similarity
        """
        rel_emb = self.model.relation_embedding.data  # (nrelation, dim)
        query_emb = rel_emb[relation_id].unsqueeze(0)        # (1, dim)

        norms = torch.nn.functional.normalize(rel_emb, p=2, dim=1)
        query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=1)
        similarities = (norms @ query_norm.T).squeeze(1)     # (nrelation,)

        sorted_indices = torch.argsort(similarities, descending=True)
        return [(idx.item(), similarities[idx].item()) for idx in sorted_indices]

    def predict_1p_batch(self, head_id: int, rel_ids: list, chunk_size: int = 32) -> torch.Tensor:
        """
        Score (head_id, r, ?) against all entities for every r in rel_ids.

        Relations are processed in chunks of chunk_size to avoid GPU OOM when
        batch_size × n_entities × embedding_dim exceeds available memory.

        Args:
            head_id:    integer entity ID of the head entity
            rel_ids:    list of R relation IDs to evaluate simultaneously
            chunk_size: number of relations processed per forward pass (default 32)
        Returns:
            FloatTensor of shape [R, n_entities], scores[j, e] = score(head, rel_ids[j], e)
        """
        query_structure = ('e', ('r',))
        all_chunks = []

        for start in range(0, len(rel_ids), chunk_size):
            chunk = rel_ids[start:start + chunk_size]
            C = len(chunk)

            flat_queries    = [[head_id, rel_j] for rel_j in chunk]
            negative_sample = torch.arange(self.args.nentity).unsqueeze(0).repeat(C, 1)

            batch_queries_dict = {query_structure: torch.LongTensor(flat_queries)}
            batch_idxs_dict    = {query_structure: list(range(C))}

            if self.args.cuda:
                batch_queries_dict[query_structure] = batch_queries_dict[query_structure].cuda()
                negative_sample = negative_sample.cuda()

            self.model.eval()
            with torch.no_grad():
                _, scores, _, _ = self.model(
                    None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

            all_chunks.append(scores.cpu())

        return torch.cat(all_chunks, dim=0)  # [R, n_entities]

    def predict_1p(self, head_id: int, relation_id: int) -> dict:
        """
        Execute a single-hop (1p) link prediction query and return scores for all entities.

        Args:
            head_id: integer entity ID of the head entity
            relation_id: integer relation ID
        Returns:
            dict with 'scores' (tensor) and 'ids' (tensor), sorted descending by score
        """
        query_tuple = ((head_id, (relation_id,)),)
        query = Query('1p', (query_tuple, [head_id]))  # dummy answer satisfies constructor
        return self.predict(query)

    def predict(self, sample_query: Query):
        # mapping the query to the desired format of KGReasoning model
        mapped = query_mapping(sample_query)
        query_tuple, query_structure = mapped[0], mapped[1]

        # Build tensors directly, bypassing DataLoader overhead
        flat_query = flatten(query_tuple)
        negative_sample = torch.LongTensor(range(self.args.nentity)).unsqueeze(0)

        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        batch_queries_dict[query_structure].append(flat_query)
        batch_idxs_dict[query_structure].append(0)

        if self.args.cuda:
            batch_queries_dict[query_structure] = torch.LongTensor(
                batch_queries_dict[query_structure]).cuda()
            negative_sample = negative_sample.cuda()
        else:
            batch_queries_dict[query_structure] = torch.LongTensor(
                batch_queries_dict[query_structure])

        self.model.eval()
        with torch.no_grad():
            _, scores, _, _ = self.model(
                None, negative_sample, None, batch_queries_dict, batch_idxs_dict)

        score = scores[0]
        sortedanswers = torch.sort(score, descending=True)
        return {'scores': sortedanswers.values, 'ids': sortedanswers.indices}