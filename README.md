#### GNN_table_QA(基于图神经网络的表格知识库问答)
----------  
#### 环境搭建  
>1. python==3.7
>2. torch==1.0.1.post2
>3. torch-scatter==1.1.2
>4. torch-sparse==0.2.4
>5. torch-cluster==1.2.4
>6. torch-geometric==1.1.2
>7. tensorflow==1.14.0  
>8. Keras==2.2.4  
>9. Ubuntu环境
----------  
#### 运行步骤  
>1.首先运行nl2sql_bert_wwm_V1.py,其中同时运行exact_feature.py（获取文本问题与表格对应信息的关联）。  
>2.由于词向量维度为768，本身代码在获取词向量过程为串行运算（样本依次计算），计算机算力运行过慢，训练过程会慢，希望大家理解。
