from .base import BaseModel
from .bert_modules.bert import BERT
import torch
import torch.nn as nn
from.bert_modules.embedding.token import TokenEmbedding
import json
class BERTModel(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        self.token = TokenEmbedding(vocab_size=args.item_size, embed_size=args.embedding_size)
        self.buy_bert = BERT(args)
        self.cart_bert=BERT(args)
        self.fav_bert=BERT(args)
        self.pv_bert=BERT(args)
        self.args=args
        self.fc=nn.Linear(args.hidden_emb_size*4, args.embedding_size)

    @classmethod
    def code(cls):
        return 'bert'

    def forward(self, batch,is_eval):
        _,masked_item_seq,masked_cate_seq,pv_item_seq,pv_cate_seq,cart_item_seq,cart_cate_seq,fav_item_seq,fav_cate_seq,pos,neg=batch
       

        masked_item_embedding=self.token(masked_item_seq)
        seq_len=masked_item_seq.size(1)
        pv_seq_embedding=self.token(pv_item_seq)
        fav_seq_embedding=self.token(fav_item_seq)
        cart_seq_embedding=self.token(cart_item_seq)
        
        masked_item_embedding = self.buy_bert(batch[1],masked_item_embedding)#B*L*H
        
            


        pv_embedding= self.buy_bert(pv_item_seq,pv_seq_embedding)[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)#B*H
        fav_embedding= self.buy_bert(fav_item_seq,fav_seq_embedding)[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)#B*H
        cart_embedding= self.buy_bert(cart_item_seq,cart_seq_embedding)[:,-1,:].unsqueeze(1).repeat(1,seq_len,1)#B*H
        # print(cart_embedding.shape)
        x=torch.cat([masked_item_embedding,pv_embedding,fav_embedding,cart_embedding],axis=2)
        # x=masked_item_embedding
        pos,neg=batch[-2],batch[-1]

        if not is_eval:
            x=x.view(-1,x.shape[-1])
            
            x=self.fc(x)
            pos=self.token(pos).view(-1,self.args.embedding_size)
            neg=self.token(neg).view(-1,self.args.embedding_size)
            point_i=torch.mul(pos, x)
            point_j=torch.mul(neg, x)
            
            point_i=point_i.sum(-1)
            point_j=point_j.sum(-1)
            
        if is_eval:
            x=x[:,-1,:]
            

            # print(batch[0][0:10,:],self.args.item_size)
            x=self.fc(x)
            x=x.unsqueeze(-1)
            pos=self.token(pos)
            neg=self.token(neg)
            

            point_i=torch.bmm(pos,x).squeeze(-1)
            point_j=torch.bmm(neg,x).squeeze(-1)
            
            
            
        return point_i,point_j
