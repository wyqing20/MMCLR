#   Multi-view Multi-behavior Contrastive Learning in Recommendation  (MMCLR)

 This is our implementation of the paper: 

  *Yiqing Wu, Ruobing Xie, Yongchun Zhu, Xiang Ao, Xin Chen, Xu Zhang, Fuzhen Zhuang, Leyu Lin, and Qing He  Multi-view Multi-behavior Contrastive Learning in Recommendation In DASFAA22*

**Please cite our DASFAA22* paper if you use our codes. Thanks!**

```
@inproceedings{wu2022Multi,
  title={Multi-view Multi-behavior Contrastive Learning in Recommendation},
  author={Yiqing Wu, Ruobing Xie, Yongchun Zhu, Xiang Ao, Xin Chen, Xu Zhang, Fuzhen Zhuang, Leyu Lin,Qing He},
  booktitle={Proceedings of DASFAA},
  year={2022},
}
```



## Example to run the codes	

```
Train and evaluate our model:
# Training without BCL
	python MultiView.py  --mode 'multi'  
# Training with BCL
	python MultiView.py --mode 'multi'  --buy_click_weight 0.05
# Set the mode to sequence and feature would be processed only using BERT model
	python MultiView.py --mode 'sequence'
# Set the mode to sequence and feature would be processed only using LightGCN model
	python MultiView.py --mode 'graph'
```
