import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import datetime
from DataUtil import tools
import random
import json
from dgl.data import DGLDataset
import pickle

'''
    this file for TIMADataset2 as having big change in TIMADataset.py, i reserve the TIMADaaset.py


'''

class MultiViewDataset(Dataset):
    def __init__(self,args,neg_sample_num=1,root_dir='MMCLR/dataset/TIMA/UserBehavior.10%.seq.splited.pickle',eval=None):
        
        super(MultiViewDataset, self).__init__()
        self.root_dir=root_dir
        self.eval=eval
        self.args=args
        self.item_set=set(self.args.item_ids)
        self.count=0
        print(root_dir)
        self.eavl=eval
        if eval is None :
            self.data=self.read_data(self.root_dir)
        else:
            self.data=self.read_data_eval(self.root_dir)
        
        self.rng=random.Random(args.random_seed)
        self.neg_sample_num=neg_sample_num
        self.raw_data=self.make_raw_data(root_dir)
        self.hardSet=self.make_hard_sample_item_set(root_dir)


    def make_hard_sample_item_set(self,file):
        f=open(file,'rb')
        all_seq={}
        b_id_set=[]
        all_info=pickle.load(f)
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            bs=['pv','cart']
            
            time=buy_times[-1]
            for b in bs:
                if len(user_info[b]['item_id'])==0:
                    continue
                b_ids=np.array(user_info[b]['item_id'])
                b_times=np.array(user_info[b]['times'])
                index=b_times>time
                pos_b_ids=b_ids[index]
                b_ids=b_ids[~index]
                pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]
                b_id_set.extend(pos_b_ids)
                
        print(len(set(b_id_set)))
        return set(b_id_set)
    def make_raw_data(self,file): 
        f=open(file,'rb')
        all_seq={}
        all_info=pickle.load(f)
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            bs=['fav','pv','cart']
            
            if len(buy_ids)==1 and self.eavl is None:## 如果少于2（不在训练集中）
                # one_seq['fav']=[]
                # one_seq['pv']=[]
                # one_seq['cart']=[]
                # one_seq['buy']=buy_ids
                # all_seq[user]=one_seq
                continue
           
            if self.eavl is None:
                time=buy_times[-2]## 训练集中最后购买的时间戳
                buy_sub_item_ids=buy_ids[:-1]
            else:
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids  
            
            for b in bs:
                if b=='buy':
                    continue
                if b not in user_info or len(user_info[b]['item_id'])==0:
                    b_ids=[]
                    pos_b_ids=[]
                else:
                    b_ids=np.array(user_info[b]['item_id'])
                    b_times=np.array(user_info[b]['times'])
                    index=b_times>time
                    pos_b_ids=b_ids[index].tolist() ## 后面的行为有多少
                    index=b_times<=time
                    b_ids=b_ids[index].tolist() #前面的行为
                    
                one_seq[b]=b_ids
                one_seq['pos'+b]=pos_b_ids
                one_seq['buy']=buy_sub_item_ids
            
            all_seq[user]=one_seq
        return all_seq
            
    def read_data_eval(self,file):
        f=open(file,'rb')
        
        all_info=pickle.load(f)
        
        f.close()
        all_seq=[]
        for user,user_info in tqdm(all_info.items()):
            buy_ids=user_info['buy']['item_id']
            buy_times=user_info['buy']['times']
            one_seq={'user_id':user}
            one_seq['posbuy']=[buy_ids[-1]]
            # if len(buy_ids)<4:
            #     continue
            
            if self.eval=='vaild':
                if buy_ids[-2] not in self.item_set:
                    
                    continue
                if buy_ids[-2] in buy_ids[:-2]:
                    
                    continue
                time=buy_times[-2]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='test':
                
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='cold_start':
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                if len(buy_ids)>3:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            elif self.eval=='uncold_start':
                if buy_ids[-1] not in self.item_set:
                    self.count+=1
                    continue
                if buy_ids[-1] in buy_ids[:-1]:
                    continue
                if len(buy_ids)<=3:
                    continue
                time=buy_times[-1]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times

            else:
                time=buy_times[-2]
                buy_sub_item_ids=buy_ids
                buy_sub_times=buy_times
            
            bs=['fav','pv','cart']
            for b in bs:
                if b=='buy':
                    continue
                if b not in user_info or len(user_info[b]['item_id'])==0:
                    b_ids=[]
                    pos_b_ids=[]
                else:
                    b_ids=np.array(user_info[b]['item_id'])
                    b_times=np.array(user_info[b]['times'])
                    index=b_times>time
                    pos_b_ids=b_ids[index].tolist()
                    
                    index=b_times<=time
                    b_ids=b_ids[index].tolist()
                    pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]

                one_seq[b]=b_ids
                one_seq['pos'+b]=pos_b_ids
            one_seq['pospv']=[later_item for later_item in one_seq['pospv'] if later_item not in one_seq['cart']]
            one_seq['poscart']=[later_item for later_item in one_seq['poscart'] if later_item not in one_seq['pv']]
            one_seq['buy']=buy_sub_item_ids
            
            if len(buy_sub_item_ids)==1 and len(one_seq['cart'])==0 and len(one_seq['pv'])==0: ## if only have one buy behavior
                    continue
            all_seq.append(one_seq)
        return all_seq
                

    def read_data(self,file):
        f=open(file,'rb')
        
        all_info=pickle.load(f)
        f.close()
        all_seq=[]
        
        for user,user_info in tqdm(all_info.items()):
           
            buy_ids=user_info['buy']['item_id']
            
            buy_times=user_info['buy']['times']
            # if len(buy_ids)<4:
            #     continue
            for i,item_id in enumerate(buy_ids[:-1]):
                
                if item_id not in self.item_set:
                    print(user,item_id)
                    continue
                
                one_seq={'user_id':user}
                time=buy_times[i]
                next_time=buy_times[i+1]
                buy_sub_item_ids=buy_ids[:i+1]
                one_seq['posbuy']=buy_ids[i+1:-1]
                buy_sub_times=buy_times[:i+1]
                bs=['fav','pv','cart']
                for b in bs:
                    if b=='buy':
                        continue
                    if b not in user_info:
                        b_ids=[]
                        pos_b_ids=[]
                    else:

                        b_ids=np.array(user_info[b]['item_id'])
                        pos_b_ids=np.array([])
                        if len(b_ids)!=0: ##当有该行为的时候
                            
                            b_times=np.array(user_info[b]['times'])
                            index=(b_times>time) & (b_times<=next_time)
                            

                            pos_b_ids=b_ids[index]
                            
                            index=b_times<=time
                            b_ids=b_ids[index]
                            
                        
                        b_ids=b_ids.tolist()
                        
                       
                        pos_b_ids=[later_item for later_item in pos_b_ids if later_item not in b_ids]

                    one_seq[b]=b_ids
                    one_seq['pos'+b]=pos_b_ids
                one_seq['buy']=buy_sub_item_ids
                if len(buy_sub_item_ids)==1 and len(one_seq['cart'])==0 and len(one_seq['pv'])==0: ## if only have one buy behavior
                    continue
                all_seq.append(one_seq)
        return all_seq
                
    def __len__(self):
        return len(self.data)

    def encode_behavior(self,behvaior):
        be2code={'pv':1,'cart':2,'fav':3,'buy':4}
        return be2code[behvaior]
    
    def mask_seq(self,mask_item):
        masked_item_seq=[]
        negtive_seq=[]
        mask_num=1
        for i in mask_item[:-1]:
                prob=self.rng.random()
                if prob<self.args.mask_prob:
                    prob=prob/self.args.mask_prob
                    
                    if prob < 0.8:
                        mask_num+=1
                        masked_item_seq.append(self.args.mask_id)
                        # masked_cate_seq.append(self.args.mask_cate)
                        neg=tools.neg_sample(set(mask_item), self.args.item_ids, self.neg_sample_num)
                        negtive_seq.append(neg[0]) 
                    elif prob < 0.9:
                        mask_num+=1
                        masked_item_seq.append(self.rng.randint(1,self.args.item_size-4))
                        # masked_cate_seq.append(self.rng.randint(1,self.args.cate_size-4))
                        # negtive_seq.append(tools.neg_sample(set(buy_item_seq), self.args.item_ids, self.neg_sample_num))
                        neg=tools.neg_sample(set(mask_item), self.args.item_ids, self.neg_sample_num)
                        negtive_seq.append(neg[0])
                    else:
                        masked_item_seq.append(i)
                        # masked_cate_seq.append(c)
                        negtive_seq.append(i)
                else:
                        masked_item_seq.append(i)
                        # masked_cate_seq.append(c)
                        negtive_seq.append(i)

       
        pos_seq=mask_item
        negtive_seq.append(tools.neg_sample(set(mask_item), self.args.item_ids, self.neg_sample_num)[0])
        masked_item_seq.append(self.args.mask_id)
        # masked_cate_seq.append(self.args.mask_cate)
        
        return masked_item_seq,pos_seq,negtive_seq,mask_num
    def __getitem__(self,index):
        user_id=self.data[index]['user_id']
        
        
    
        pv_item_seq=[self.args.start_id]+self.data[index]['pv']+[self.args.end_id]
      
        buy_item_seq=self.data[index]['buy']
        fav_item_seq=[self.args.start_id]+self.data[index]['buy']+[self.args.end_id]
        cart_item_seq=[self.args.start_id]+self.data[index]['cart']+[self.args.end_id]
        pos_buy_item_seq=[self.args.start_id]+self.data[index]['posbuy']+[self.args.end_id]
        ### get constractive sample
        multi_items=[self.data[index]['buy'],self.data[index]['pv'],self.data[index]['cart']]
        have_constra=1
        have_click=1 ## have cart for inner cons
        if len(multi_items[0])==0 or len(multi_items[1])==0 or len(multi_items[2])==0:
                      
            b=[ i for i,j in enumerate(multi_items) if  len(j)>0]
            if len(b)>1:
                
                b3=b
                c=self.rng.randint(0, 1)
                b1=b3[c]
                b2=b3[1-c]
            else:
                have_constra=0
                have_click=0
                b3=[0,1]
                b1,b2=0,0
            if 1 not in b:
                have_click=0
        else:
            have_cart=1         
            b1,b2,b3=0,1,[0]
            b1=self.rng.randint(0, 2)
            b2=b1
            while b2==b1:
                b2=self.rng.randint(0, 2)
            b3=[b1,b2]
        b1,b2=multi_items[b1][-self.args.max_seq_len:],multi_items[b2][-self.args.max_seq_len:]
        if b1==0:
            b1=([self.args.start_id]+b1[-self.args.max_seq_len+2:]+[self.args.end_id])
        if b2==0:
            b2=([self.args.start_id]+b2[-self.args.max_seq_len+2:]+[self.args.end_id])
        con_len=[len(b1),len(b2),len(b3)]
        b1=[0]*(self.args.max_seq_len-len(b1))+b1
        b2=[0]*(self.args.max_seq_len-len(b2))+b2
        if len(b1)!=len(b2):
            print(len(b1),len(b2),b3)
        behavior_ctra_sample=(b1,b2)
        if self.eval is None:
            ## here we only mask last item 
            masked_item_seq,pos_seq,negtive_seq,mask_num=self.mask_seq(buy_item_seq)
        else:
            mask_num=1
            if self.eval =='test' or self.eval=='cold_start' or self.eval=='uncold_start':
                pos_seq=[buy_item_seq[-1]]
                masked_item_seq=buy_item_seq[:-1]
            elif self.eval=='vaild':
                pos_seq=[buy_item_seq[-2]]
                masked_item_seq=buy_item_seq[:-2]
            
            negtive_seq=tools.neg_sample(set(buy_item_seq), self.args.item_ids, self.neg_sample_num)
            masked_item_seq.append(self.args.mask_id)
        ## here we smaple the click for innerCL
        sampled_clicks=[-1]*50
        sample_item=self.data[index]['pospv']
        aragen=len(sample_item) # pospv是当前预测的 buy item time 之后到下一个 buy item之前的所有click item
        for i in range(mask_num):
            if aragen==0 :
                sampled_clicks[i]=0 #当sampleclick 没有时候
                continue
            aragen=min(len(sample_item),10) ##如果太多只取10个
            sampled_click=self.rng.randint(0, aragen-1)
            sampled_click=sample_item[sampled_click]
            sampled_clicks[i]=sampled_click
        sample_items=self.data[index]['poscart']+self.data[index]['pospv']
        if self.eavl:
            if aragen!=0:
                aragen=min(len(sample_items),10)
                for i in range(aragen):
                    sampled_clicks[i]=sample_items[i]
        pad_len=self.args.max_seq_len-len(masked_item_seq)
        masked_item_seq=masked_item_seq[-self.args.max_seq_len:]
        masked_item_seq=[0]*pad_len+masked_item_seq
        pad_len=self.args.max_seq_len-len(pv_item_seq)
        pv_item_seq=[0]*pad_len+pv_item_seq[-self.args.max_seq_len:]
        pad_len=self.args.max_seq_len-len(cart_item_seq)
       
        cart_item_seq=[0]*pad_len+cart_item_seq[-self.args.max_seq_len:]
        pad_len=self.args.max_seq_len-len(fav_item_seq)
        fav_item_seq=[0]*pad_len+fav_item_seq[-self.args.max_seq_len:]
        pad_len=self.args.max_seq_len-len(pos_buy_item_seq)
        pos_buy_item_seq=[0]*pad_len+pos_buy_item_seq[-self.args.max_seq_len:]               
        if self.eval is None:
            pad_len=self.args.max_seq_len-len(pos_seq)
            pos_seq=[0]*pad_len+pos_seq[-self.args.max_seq_len:]        
            negtive_seq=[0]*pad_len+negtive_seq[-self.args.max_seq_len:]
        cur_tensor=(
            torch.LongTensor([user_id]),
            torch.LongTensor(masked_item_seq),
            torch.LongTensor(pv_item_seq),
            torch.LongTensor(cart_item_seq),
            torch.LongTensor(fav_item_seq),
            torch.tensor(pos_seq,dtype=torch.long),
            torch.tensor(negtive_seq,dtype=torch.long),
            torch.tensor(b1,dtype=torch.long),
            torch.tensor(b2,dtype=torch.long),
            torch.tensor(b3,dtype=torch.long),
            torch.tensor([have_click],dtype=torch.long),
            torch.tensor(sampled_clicks,dtype=torch.long),
            torch.tensor([have_constra],dtype=torch.long)
        )
        return cur_tensor
