import argparse
from DataUtil import TIMADataset2 as TIMADataset
import torch
from torch.utils.data import DataLoader,RandomSampler
from model import MultiViewModel
from torch.optim import Adam,SGD
from DataUtil import tools
from DataUtil.sampler import  NeighborSampler,NeighborSamplerForMGIR
from tqdm import tqdm
import numpy as np
from models import BERTModel
import json
from sklearn.metrics import accuracy_score
import dgl
from sklearn import metrics

def main(graph_cons_weight=0.2,seq_cons_weight=0.2,cross_cons_weight=0.2,temp=1.0,counter=-1):
    parser=argparse.ArgumentParser()
    # parser.add_argument('--seed',type=int,default=0,help='Random Seed')
    arser=argparse.ArgumentParser()
    # parser.add_argument('--seed',type=int,default=0,help='Random Seed')
    parser.add_argument('--max_seq_len',type=int,default=100,help='max length of seq')
    parser.add_argument('--batch_size',type=int,default= 256,help='the batch size of model')
    parser.add_argument('--kernel_gcn',default='lightgcn',type=str)
    parser.add_argument('--epochs',type=int,default=1000)
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--weight_decay',type=float,default=1e-4)
    parser.add_argument('--device',type=str,default='cuda:2')
    parser.add_argument('--use_cuda',type=bool,default=True)
    parser.add_argument('--embedding_size',type=int,default=64)
    parser.add_argument('--neg_sample_num',type=int,default=99)
    parser.add_argument('--mask_prob',type=float,default=0.2)
    parser.add_argument('--random_seed',type=int,default=0)
    parser.add_argument('--patience',type=int,default=25,help='the patience of early stopping')
    parser.add_argument('--bert_dropout',type=float,default=0.1)
    parser.add_argument('--bert_num_heads',type=int,default=2)
    parser.add_argument('--saved_model_name',type=str,default='checkpoint.pt')
    parser.add_argument('--bert_layer',type=int,default=2)
    parser.add_argument('--no_constra',type=bool,default=False)
    parser.add_argument('--uniform',type=bool,default=True)
    parser.add_argument('--n_gcn_layers',type=int,default=2)
    parser.add_argument('--link_weight',type=float,default=1.0)
    parser.add_argument('--graph_cons_weight',type=float,default=0.2)
    parser.add_argument('--seq_cons_weight',type=float,default=0.2)
    parser.add_argument('--cross_cons_weight',type=float,default=0.2)
    parser.add_argument('--saving_model',type=bool,default=True)
    parser.add_argument('--mode',type=str,default='multi')
    parser.add_argument('--hidden_act',type=str,default='gelu')
    parser.add_argument('--hidden_size',type=int,default=64)
    parser.add_argument('--hidden_emb_size',type=int,default=128)
    parser.add_argument('--initializer_range',type=float,default=0.02)
    parser.add_argument('--describe',type=str,default='this is desrcible for model')
    parser.add_argument('--inner_loss_weight',type=float,default=0.00)
    parser.add_argument('--buy_click_weight',type=float,default=0.00)
    parser.add_argument('--curriculum',type=bool,default=False)
    parser.add_argument('--remove_click_edges',type=int,default=1)
    parser.add_argument('--test',type=int,default=1)
    parser.add_argument('--clamp',type=int,default=0)
    parser.add_argument('--save_each_step',type=int,default=0)
    parser.add_argument('--temp',type=float,default=1.0)
    parser.add_argument('--sim',type=str,default='dot')
    parser.add_argument('--lamda',type=float,help='the weight of click and random dis')
    parser.add_argument('--main_weight',type=float,default=1.0)
    
    args=parser.parse_args()
    # args.user_size,args.item_size,args.cate_size,args.behavior_size=987994,4162024,9439,5
    args.user_size,args.item_size,args.cate_size,args.behavior_size=22014,27155,9439,5
    tools.set_seed(counter)   
    args.mask_id=args.item_size+1
    args.start_id=args.item_size+2
    args.end_id=args.item_size+3
    args.item_size=args.item_size+4
    args.mask_cate=args.cate_size+1
    args.user_size+=1
    args.cate_size+=2
    args.seq_cons_weight=seq_cons_weight
    args.graph_cons_weight=graph_cons_weight
    args.cross_cons_weight=cross_cons_weight
    args.temp=temp
    args.root='/data/wuyq/MMCLR/checkPoints/'
    args.describe=args.describe+'{},{},{}'.format(seq_cons_weight,graph_cons_weight,cross_cons_weight)
    
    if args.describe=='this is desrcible for model':
     
      args.saved_model_name=args.mode+'{},{},{},{}_{}_checkpoint.pt'.format(args.seq_cons_weight,args.graph_cons_weight,args.cross_cons_weight,args.inner_loss_weight,counter)
    else:
      args.saved_model_name=args.mode+'{},{},{},{}_{}_checkpoint.pt'.format(args.seq_cons_weight,args.graph_cons_weight,args.cross_cons_weight,args.buy_click_weight,counter)+args.describe
  
    print(args)
  
    print('loading train dataset ....')
    train_file='/data/wuyq/MMCLR/dataset/TIMA2/train_seq'
    train_graph,test_graph,args.item_ids,args.item_set=tools.get_TIMA_split_traintest()
    co_graph=tools.get_co_graph()
    args.co_g=co_graph.to(args.device)
    print(co_graph)
    # train_graph,args.item_ids,item_set=tools.get_TIMA_MRIG()
    args.g=train_graph.to(args.device)
    args.test_g=test_graph.to(args.device)
    print(args.test_g)
    print(train_graph)
    
    train_dataset=TIMADataset.MultiViewDataset(args,root_dir=train_file)
    
    train_sampler=NeighborSampler(train_graph, num_layers=args.n_gcn_layers,args=args)
    # train_sampler=NeighborSamplerForMGIR(train_graph, num_layers=args.n_gcn_layers,args=args)
    print(len(train_dataset))
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,collate_fn=train_sampler.sample_from_item_pairs
    ,shuffle=True,num_workers=8)
    eval_sampler=NeighborSampler(train_graph, num_layers=args.n_gcn_layers, args=args,neg_sample_num=args.neg_sample_num,is_eval=True)
    # eval_sampler=NeighborSamplerForMGIR(train_graph, num_layers=args.n_gcn_layers, args=args,neg_sample_num=args.neg_sample_num,is_eval=True)
    vaild_dataset=TIMADataset.MultiViewDataset(args,root_dir=train_file,eval='test',neg_sample_num=args.neg_sample_num)
    vaild_dataloader=DataLoader(vaild_dataset,batch_size=256,collate_fn=eval_sampler.sample_from_item_pairs,shuffle=True,num_workers=8)
    test_dataset=TIMADataset.MultiViewDataset(args,root_dir=train_file,eval='test',neg_sample_num=args.neg_sample_num)
    test_dataloader=DataLoader(test_dataset,batch_size=256,collate_fn=eval_sampler.sample_from_item_pairs,shuffle=True,num_workers=8)
    cold_start_dataset=TIMADataset.MultiViewDataset(args,root_dir=train_file,eval='cold_start',neg_sample_num=args.neg_sample_num)
    cold_start_dataloader=DataLoader(cold_start_dataset,batch_size=256,collate_fn=eval_sampler.sample_from_item_pairs,shuffle=True,num_workers=8)
    uncold_start_dataset=TIMADataset.MultiViewDataset(args,root_dir=train_file,eval='uncold_start',neg_sample_num=args.neg_sample_num)
    uncold_start_dataloader=DataLoader(uncold_start_dataset,batch_size=256,collate_fn=eval_sampler.sample_from_item_pairs,shuffle=True,num_workers=8)
    print('graph con w is {} seq is {} cross is {} innner is {} buy click is {} seed is{}'.format(args.graph_cons_weight,args.seq_cons_weight,args.cross_cons_weight,args.inner_loss_weight,args.buy_click_weight,counter))
    print(len(cold_start_dataset))
    print(len(vaild_dataset))
    print(len(test_dataset))
    print(vaild_dataset.count,'dataset size')
    model=MultiViewModel(args)
    early_stop=tools.EarlyStopping(patience=args.patience,verbose=True,root=args.root,path=args.saved_model_name,saving_model=args.saving_model)
    
    if args.curriculum:
      args.buy_click_weight=0.05
      model_path='/data/wuyq/MBBaseline/paramTest/'+'sequence0.2,0.2,0.2,0.0_244checkpoint.ptSEQNODIS0.2,0.2,0.2'
      model=model.to(device=args.device)
      early_stop.path=model_path+'curriculum'+str(args.buy_click_weight)
      print(early_stop.path)
      optimizer=Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
      optimizer.load_state_dict(torch.load(model_path)['optimizer'])
      
      
    else:
      model=model.to(device=args.device)
      optimizer=Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    
    print('model loaded')
    losses=[]
    link_losses=[]
    graph_constra_losses=[]
    seq_constra_losses=[]
    corss_cons_losses=[]
    val_link_losses=[]
    val_seq_constra_losses=[]
    val_graph_constra_losses=[]
    val_corss_cons_losses=[]
    look_losses=[]
    val_losses=[]
    graph_inner_losses=[]
    seq_inner_losses=[]
    val_graph_inner_losses=[]
    val_seq_inner_losses=[]
    scores=[]
    step=0
    for epoch in range(10000):
      if args.test==1:
        break
      if args.save_each_step==1 and not args.curriculum:
          args.saved_model_name=early_stop.root+'saveALLStep/'+args.mode+'{},{},{},{}_{}checkpoint.pt'.format(args.seq_cons_weight,args.graph_cons_weight,args.cross_cons_weight,args.buy_click_weight,epoch)+args.describe
          early_stop.path=args.saved_model_name
          print(early_stop.path)
      model.train()
      for input_nodes,pos_graph,neg_graph,blocks,block_src_nodes,seq_tensors,neg_user_ids in tqdm(train_dataloader):
        if block_src_nodes is not None:
          block_src_nodes=[{k:v.to(args.device)   for k,v in nodes.items()} for nodes in block_src_nodes ]
          input_nodes={k:v.to(args.device) for k,v in input_nodes.items()}
          pos_graph=pos_graph.to(args.device)
          neg_graph=neg_graph.to(args.device)
          blocks=[block.to(args.device) for block in blocks]
        seq_tensors=[seq.to(args.device) for seq in seq_tensors]
        graph_data=(input_nodes,pos_graph,neg_graph,blocks,block_src_nodes)
        loss,link_loss,seq_cl_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,seq_inner_loss=model(graph_data,seq_tensors,is_eval=False)
        step+=1
        if True:
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
        graph_inner_losses.append(graph_inner_loss.item())
        seq_inner_losses.append(seq_inner_loss.item())
        look_losses.append(loss.item())
        link_losses.append(link_loss.item())
        seq_constra_losses.append(seq_cl_loss.item())
        graph_constra_losses.append(graph_cl_loss.item())
        corss_cons_losses.append(cross_constra_loss.item())
      print(optimizer.state_dict()['param_groups'][0]['lr'])
      loss_inf='Epoch:{}----> train loss is {}  link loss is {} seq CL loss is {} graph CL loss is {} corss CL loss is {} graph_inner_loss is {} seq_inner_loss is {}'.format(
        epoch,np.array(look_losses).mean(),
        np.array(link_losses).mean(),
        np.array(seq_constra_losses).mean(),
        np.array(graph_constra_losses).mean(),
        np.array(corss_cons_losses).mean(),
        np.array(graph_inner_losses).mean(),
        np.array(seq_inner_losses).mean(),

        )
      
      print(loss_inf)  
      look_losses=[]
      link_losses=[]
      graph_constra_losses=[]
      corss_cons_losses=[]
      seq_constra_losses=[]
      graph_inner_losses=[]
      seq_inner_losses=[]
      with torch.no_grad():
        model.eval()
        auc_scores=[[],[]]
        for input_nodes,pos_graph,neg_graph,blocks,block_src_nodes,seq_tensors,neg_user_ids in tqdm(vaild_dataloader):
          if block_src_nodes is not None:
            block_src_nodes=[{k:v.to(args.device)   for k,v in nodes.items()} for nodes in block_src_nodes ]
            input_nodes={k:v.to(args.device) for k,v in input_nodes.items()}
            pos_graph=pos_graph.to(args.device)
            neg_graph=neg_graph.to(args.device)
            blocks=[block.to(args.device) for block in blocks]
          seq_tensors=[seq.to(args.device) for seq in seq_tensors]
          graph_data=(input_nodes,pos_graph,neg_graph,blocks,block_src_nodes)
          loss,link_loss,seq_cl_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,seq_inner_loss,point_j=model(graph_data,seq_tensors,is_eval=True)
          point_j=point_j.cpu()
          val_losses.append(loss.item())
          val_link_losses.append(link_loss.item())
          val_seq_constra_losses.append(seq_cl_loss.item())
          val_graph_constra_losses.append(graph_cl_loss.item())
          val_corss_cons_losses.append(cross_constra_loss.item())
          val_graph_inner_losses.append(graph_inner_loss.item())
          val_seq_inner_losses.append(seq_inner_loss.item())
          score=tools.get_score(point_j)
          scores.append(score)
      is_earlying=False
      HIT_1,HIT_5,HIT_10,NDCG_1,NDCG_5,NDCG_10,MRR,AUC=np.array(scores).mean(axis=0)
      if args.test!=1:
        is_earlying=early_stop(NDCG_10,model,optimizer=optimizer,epoch=epoch)
      elif epoch==10:
        is_earlying.early_stop=True
      scores=[]
      
      if not is_earlying:
        loss_inf='Epoch:{}----> vaild loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
         np.array(val_seq_inner_losses).mean(),
        )
      else:
        loss_inf='counter  {}  Epoch:{}----> vaild loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(early_stop.counter,epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
         np.array(val_seq_inner_losses).mean(),
        )
      print(loss_inf)
      val_losses,val_link_losses,val_seq_constra_losses,val_graph_constra_losses,val_corss_cons_losses,val_inner_losses=[],[],[],[],[],[]
      post_fix = {
          "Epoch": epoch,
          "MRR": '{:.4f}'.format(MRR),
          "AUC": '{:.4f}'.format(AUC),
          "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
          "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
          "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
      }
      print(post_fix)
      if early_stop.early_stop:          
            break
      if epoch<-1:
        continue
      with torch.no_grad():
        model.eval()
        auc_scores=[[],[]]
        for input_nodes,pos_graph,neg_graph,blocks,block_src_nodes,seq_tensors,neg_user_ids in tqdm(cold_start_dataloader):
          
          if block_src_nodes is not None:
            block_src_nodes=[{k:v.to(args.device)   for k,v in nodes.items()} for nodes in block_src_nodes ]
            input_nodes={k:v.to(args.device) for k,v in input_nodes.items()}
            pos_graph=pos_graph.to(args.device)
            neg_graph=neg_graph.to(args.device)
            blocks=[block.to(args.device) for block in blocks]
          seq_tensors=[seq.to(args.device) for seq in seq_tensors]
          graph_data=(input_nodes,pos_graph,neg_graph,blocks,block_src_nodes)
          loss,link_loss,seq_cl_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,seq_inner_loss,point_j=model(graph_data,seq_tensors,is_eval=True)
          point_j=point_j.cpu()
          val_losses.append(loss.item())
          val_link_losses.append(link_loss.item())
          val_seq_constra_losses.append(seq_cl_loss.item())
          val_graph_constra_losses.append(graph_cl_loss.item())
          val_corss_cons_losses.append(cross_constra_loss.item())
          val_graph_inner_losses.append(graph_inner_loss.item())
          val_seq_inner_losses.append(seq_inner_loss.item())
          score=tools.get_score(point_j)
          scores.append(score)  
      HIT_1,HIT_5,HIT_10,NDCG_1,NDCG_5,NDCG_10,MRR,AUC=np.array(scores).mean(axis=0)
      scores=[]
      if not is_earlying:
        loss_inf='Epoch:{}----> cold_start loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
         np.array(val_seq_inner_losses).mean(),
        )
      else:
        loss_inf='counter  {}  Epoch:{}----> cold_start loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(early_stop.counter,epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
         np.array(val_seq_inner_losses).mean(),
        )
      
      val_losses,val_link_losses,val_seq_constra_losses,val_graph_constra_losses,val_corss_cons_losses,val_inner_losses=[],[],[],[],[],[]
      post_fix = {
          "Epoch": epoch,
          "MRR": '{:.4f}'.format(MRR),
          "AUC": '{:.4f}'.format(AUC),
          "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
          "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
          "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
      }
      print(loss_inf)
      print(post_fix)
      with torch.no_grad():
          model.eval()
          auc_scores=[[],[]]
          for input_nodes,pos_graph,neg_graph,blocks,block_src_nodes,seq_tensors,neg_user_ids in tqdm(uncold_start_dataloader):
            
            if block_src_nodes is not None:
              block_src_nodes=[{k:v.to(args.device)   for k,v in nodes.items()} for nodes in block_src_nodes ]
              input_nodes={k:v.to(args.device) for k,v in input_nodes.items()}
              pos_graph=pos_graph.to(args.device)
              neg_graph=neg_graph.to(args.device)
              
              blocks=[block.to(args.device) for block in blocks]
            seq_tensors=[seq.to(args.device) for seq in seq_tensors]
            graph_data=(input_nodes,pos_graph,neg_graph,blocks,block_src_nodes)
            # loss,pr_loss,ce_loss,_,point_j,_,_,_=model(seq_tensors,is_eval=True)
            loss,link_loss,seq_cl_loss,graph_cl_loss,cross_constra_loss,graph_inner_loss,seq_inner_loss,point_j=model(graph_data,seq_tensors,is_eval=True)
            point_j=point_j.cpu()
            val_losses.append(loss.item())
            val_link_losses.append(link_loss.item())
            val_seq_constra_losses.append(seq_cl_loss.item())
            val_graph_constra_losses.append(graph_cl_loss.item())
            val_corss_cons_losses.append(cross_constra_loss.item())
            val_graph_inner_losses.append(graph_inner_loss.item())
            val_seq_inner_losses.append(seq_inner_loss.item())
            score=tools.get_score(point_j)
            scores.append(score)
      HIT_1,HIT_5,HIT_10,NDCG_1,NDCG_5,NDCG_10,MRR,AUC=np.array(scores).mean(axis=0)
    
      scores=[]

      if not is_earlying:
        loss_inf='Epoch:{}----> cold_start loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
        np.array(val_seq_inner_losses).mean(),
        )
      else:
        loss_inf='counter  {}  Epoch:{}----> cold_start loss is {} link loss is {} seq constra loss is {} graph constra loss {} corss CL loss {} graph_inner_loss is {} seq_inner_loss is {}'.format(early_stop.counter,epoch,np.array(val_losses).mean(),
        np.array(val_link_losses).mean(),np.array(val_seq_constra_losses).mean(),np.array(val_graph_constra_losses).mean(),
        np.array(val_corss_cons_losses).mean(),
        np.array(val_graph_inner_losses).mean(),
        np.array(val_seq_inner_losses).mean(),
        )
      print(loss_inf)
      val_losses,val_link_losses,val_seq_constra_losses,val_graph_constra_losses,val_corss_cons_losses,val_inner_losses=[],[],[],[],[],[]
      post_fix = {
          "Epoch": epoch,
          "MRR": '{:.4f}'.format(MRR),
          "AUC": '{:.4f}'.format(AUC),
          "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
          "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
          "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
      }

      

      
if __name__=='__main__':
  graph_cons_weights=[0.2]
  seq_cons_weights=[0.2]
  cross_constra_weights=[0.2]
  # temps=[1.0,2.0,3.0,4.0,4.5,5.0,6.0]
  temps=[2.0]
  c=10
  for graph_cons_weight in graph_cons_weights:
    for  seq_cons_weight in seq_cons_weights:
      for cross_constra_weight in cross_constra_weights:
        for temp in temps:
          c+=10
          main(graph_cons_weight=graph_cons_weight,seq_cons_weight=seq_cons_weight,cross_cons_weight=cross_constra_weight,temp=temp,counter=c)
         
