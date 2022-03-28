import numpy as np
import dgl
import torch
from torch.utils.data import IterableDataset, DataLoader
import random

class NeighborSamplerForMGIR(object):
    def __init__(self, g, num_layers,args,neg_sample_num=1,is_eval=False):
        self.g = g
        self.num_layers=num_layers
        self.is_eval=is_eval
        self.neg_sample_num=neg_sample_num
        self.args=args
        self.rng=random.Random(self.args.random_seed)
        self.error_count=0
        self.total=0

    def sample_from_item_pairs(self, seq_tensors):
        neg_src=[]
        pos_src=[]
        pos_dst=[]
        neg_dst=[]
        batch_tensors=[[] for _ in range(len(seq_tensors[0]))]
        for seq in seq_tensors:
            user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,b3,con_len,sampled_click,pos_buy_item_seq=seq
            
            
            for i,data in enumerate(seq):
                batch_tensors[i].append(data)
            if self.is_eval:

                neg_dst.append(neg)
                pos_dst.append(pos)
                # pos_dst.append(pos.repeat(self.neg_sample_num))
            else:
                masked=pos-neg
                
                pos_dst.append(pos[masked!=0])
                neg_dst.append(neg[masked!=0])
                
                
        
            pos_src.append(user_id.repeat(pos_dst[-1].shape[0]))
            neg_src.append(user_id.repeat(neg_dst[-1].shape[0]))
       
        batch_tensors=[ torch.stack(tensors,dim=0) for tensors in batch_tensors]
        batch_tensors[0]=batch_tensors[0].reshape(-1)


        return None,None, None, None,None,batch_tensors,None

class NeighborSampler(object):
    def __init__(self, g, num_layers,args,neg_sample_num=1,is_eval=False):
        self.g = g
        self.num_layers=num_layers
        self.is_eval=is_eval
        self.neg_sample_num=neg_sample_num
        self.args=args
        self.rng=random.Random(self.args.random_seed)
        self.error_count=0
        self.total=0

    

    def sample_blocks(self, seeds, heads=None, tails=None, neg_tails=None):
        blocks = []
        block_src_nodes=[]
        for layer in range(self.num_layers):
            frontier = dgl.in_subgraph(self.g, seeds)
            frontier=dgl.compact_graphs(frontier,always_preserve=seeds)
            seeds = frontier.srcdata[dgl.NID]
            blocks.insert(0, frontier)
            src_nodes={}
            for ntype in frontier.ntypes:
                src_nodes[ntype]=frontier.nodes(ntype=ntype)
            block_src_nodes.insert(0,src_nodes)
        input_nodes=seeds
       

        return input_nodes,blocks,block_src_nodes

    def sample_neg_user(self,batch_users):
        batch_users=batch_users.tolist()
        neg_batch_users=[]
        user_set=list(set(batch_users))
        
        for user in batch_users:
            neg_user=user
            while neg_user==user:
                neg_user_idx=self.rng.randint(0,len(user_set)-1)
                neg_user=user_set[neg_user_idx]
                
            neg_batch_users.append(neg_user_idx)
        return torch.tensor(neg_batch_users)

    def sample_from_item_pairs(self, seq_tensors):
        neg_src=[]
        pos_src=[]
        pos_dst=[]
        neg_dst=[]
        batch_tensors=[[] for _ in range(len(seq_tensors[0]))]
        for seq in seq_tensors:
            user_id,masked_item_seq,pv_item_seq,cart_item_seq,fav_item_seq,pos,neg,b1,b2,b3,con_len,sampled_click,pos_buy_item_seq=seq
            
            for i,data in enumerate(seq):
                batch_tensors[i].append(data)
            if self.is_eval:

                neg_dst.append(neg)
                pos_dst.append(pos)
            else:
                masked=pos-neg
                pos_dst.append(pos[masked!=0])
                neg_dst.append(neg[masked!=0])
            pos_src.append(user_id.repeat(pos_dst[-1].shape[0]))
            neg_src.append(user_id.repeat(neg_dst[-1].shape[0]))
       
        batch_tensors=[ torch.stack(tensors,dim=0) for tensors in batch_tensors]
        batch_tensors[0]=batch_tensors[0].reshape(-1)
        neg_user_ids=torch.tensor([0])
        pos_dst=torch.cat(pos_dst,axis=0)
        neg_dst=torch.cat(neg_dst,axis=0)
        neg_src=torch.cat(neg_src,axis=0)
        pos_src=torch.cat(pos_src,axis=0)
        
        pos_graph = dgl.heterograph({('user','buy','item'):
            (pos_src, pos_dst),
           
            
            }
            )
        neg_graph = dgl.heterograph(
            {('user','buy','item'):
            (neg_src, neg_dst),
            }
            )
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])
        buy_items=torch.cat((pos_dst,neg_dst),dim=0)
        seeds = {'user':batch_tensors[0],'item':torch.cat((pos_dst,neg_dst),dim=0)}
        input_nodes,blocks,block_src_nodes = None,None,None
        return input_nodes,pos_graph, neg_graph, blocks,block_src_nodes,batch_tensors,neg_user_ids