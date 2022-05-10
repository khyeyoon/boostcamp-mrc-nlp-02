import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import pandas as pd

import json
import random
from tqdm.auto import tqdm
from pprint import pprint
import wandb
import argparse
import os 
import time
from datasets import load_from_disk, load_dataset
from typing import List, NoReturn, Optional, Tuple, Union
from datasets import (
    Dataset,
    DatasetDict,
)
from arguments import DataTrainingArguments, ModelArguments
from contextlib import contextmanager
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
    HfArgumentParser,
)

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
set_seed(42) # magic number :)

class DenseRetrieval:
    def __init__(self, args, dataset, tokenizer, p_encoder, q_encoder, num_neg=2, mode='train', bm25=False):

        self.args = args[0]
        self.additional_args = args[1]

        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder
        self.bm25 = bm25
        self.q_embs = None
        self.p_embs = None
        self.q_embs_eval = None
        self.p_embs_eval = None
        self.mode = mode

        if mode=='train':

            self.train_dataset = dataset['train']

            dict_data=dict()

            KorQuAD_dataset = load_dataset("squad_kor_v1")['train']
            dict_data['context']=self.train_dataset['context']+KorQuAD_dataset['context'][:2000]
            dict_data['question']=self.train_dataset['question']+KorQuAD_dataset['question'][:2000]
            print("데이터셋 개수",len(dict_data['context']))
            self.train_dataset=dict_data

            self.validation_dataset = dataset['validation']

            self.prepare_in_batch_negative()

        else:
            # infernce를 위한 passages 임베딩 시 수행 (default=train)
            if mode=='inference':
                self.get_dense_embedding(dataset)
            else:
                self.get_dense_embedding(self.validation_dataset)
                self.compute_topk()

        if bm25=='True':
            corpus = np.array(list(set([example for example in self.train_dataset['context']])))
            tokenized_corpus = [self.split_space(doc) for doc in corpus]
            self.train_bm25 = BM25Okapi(tokenized_corpus)

            corpus = np.array(list(set([example for example in self.validation_dataset['context']])))
            tokenized_corpus = [self.split_space(doc) for doc in corpus]
            self.validation_bm25 = BM25Okapi(tokenized_corpus)


    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.p_embs is not None, "get_dense_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": ('\n'+"="*150+'\n').join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas
    
    # query, passage embedding 수행하여 변수에 저장
    def get_dense_embedding(self, dataset):

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder
        tokenizer = self.tokenizer
        BATCHSIZE = 32

        if self.mode=='inference':
            # 중복 passage 제거 필요함

            passages, questions = dataset[0],dataset[1]

            passages_list=[]
            for idx, p in enumerate(passages):
                # passages[idx]['text']
                passages_list.append(passages[str(idx)]['text'])

            passages_list=list(set(passages_list))
            print('[passages num]:',len(passages_list))

            p_seqs = tokenizer(passages_list, padding="max_length", truncation=True, return_tensors='pt')

            passage_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            )

            self.passage_dataloader = DataLoader(passage_dataset, batch_size=BATCHSIZE)
            
            with torch.no_grad():
                p_encoder.eval()
                p_embs=[]

                with tqdm(self.passage_dataloader , unit="batch") as tepoch:
                    for idx, batch in enumerate(tepoch):

                        p_inputs = {
                            'input_ids': batch[0].to(args.device),
                            'attention_mask': batch[1].to(args.device),
                            'token_type_ids': batch[2].to(args.device)
                        }
                
                        p_outputs = self.p_encoder(**p_inputs) # (batch_size*(num_neg+1), emb_dim)
                        p_embs.append(p_outputs)

            self.p_embs= torch.concat(p_embs, dim=0)
            self.contexts = passages_list

        else:
            # 2. (Question, Passage) 데이터셋 만들어주기
            q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
            p_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

            print(q_seqs['input_ids'].shape,p_seqs['input_ids'].shape)

            train_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
            )

            passage_dataset = TensorDataset(
                p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            )

            self.passage_dataloader = DataLoader(passage_dataset, batch_size=BATCHSIZE)

            dataloader = DataLoader(train_dataset, batch_size=BATCHSIZE)

            with torch.no_grad():
                p_encoder.eval()
                q_encoder.eval()
                p_embs,q_embs=[],[]
                p_embs_eval,q_embs_eval=[],[]

                with tqdm(dataloader , unit="batch") as tepoch:
                    for idx, batch in enumerate(tepoch):

                        p_inputs = {
                            'input_ids': batch[0].to(args.device),
                            'attention_mask': batch[1].to(args.device),
                            'token_type_ids': batch[2].to(args.device)
                        }
                
                        q_inputs = {
                            'input_ids': batch[3].to(args.device),
                            'attention_mask': batch[4].to(args.device),
                            'token_type_ids': batch[5].to(args.device)
                        }
                
                        p_outputs = self.p_encoder(**p_inputs) # (batch_size*(num_neg+1), emb_dim)
                        p_embs.append(p_outputs)
                        q_outputs = self.q_encoder(**q_inputs) # (batch_size*, emb_dim)
                        q_embs.append(q_outputs)

            self.q_embs= torch.concat(q_embs, dim=0)
            self.p_embs= torch.concat(p_embs, dim=0)

    def compute_topk(self):
        dataset_len = self.q_embs.size(0)
        top1,top20,top100=0,0,0

        # 쿼리 하나씩 받아오면서 계산하기 
        for idx in range(dataset_len):

            q_emb = self.q_embs[idx,:]
            
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embs, 0, 1))
            rank = torch.argsort(dot_prod_scores, dim=0, descending=True).squeeze()

            if idx in rank[:100]:
                top100+=1
            if idx in rank[:20]:
                top20+=1
            if idx == rank[0]:
                top1+=1

        top1_acc=top1/dataset_len
        top20_acc=top20/dataset_len
        top100_acc=top100/dataset_len

        print('[Top-1 acc]',top1_acc,' | ','[Top-20 acc]',top20_acc,' | ','[Top-100 acc]', top100_acc)

    def split_space(self, sent):
        return sent.split(" ")

    def BM25(self, bm25, query, corpus):

        tokenized_query = self.split_space(query)
        doc_scores = bm25.get_scores(tokenized_query)
        top_n_passages = bm25.get_top_n(tokenized_query, corpus, n=20)
        
        return top_n_passages

    def prepare_in_batch_negative(self, dataset=None, tokenizer=None):

        train_dataset = self.train_dataset
        validation_dataset = self.validation_dataset
        tokenizer = self.tokenizer

        if self.bm25:
            num=self.num_neg+1+self.additional_args.bm_num
        else:
            num=self.num_neg+1

        print("negative sample",num)

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in train_dataset['context']])))

        p_with_neg_train = []
        print("prepare_in_batch_negative for train")
        with tqdm(train_dataset['context'], unit="batch") as tepoch:
            for idx, c in enumerate(tepoch):
            
                while True:
                    neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]

                        p_with_neg_train.append(c)
                        p_with_neg_train.extend(p_neg)
                        

                        # BM25로 질문과 유사도 높은 지문 negative sample로 추가 (--bm_num으로 몇개 추가할건지 설정가능)
                        if self.bm25:
                            cnt=0
                            top_n_passages=self.BM25(self.train_bm25,train_dataset['question'][idx], corpus)
                            
                            for p in top_n_passages:
                                if p!=c: # and dataset['answers'][idx]['text'][0] not in p:
                                    p_with_neg_train.append(p)
                                    cnt+=1
                                if cnt==self.additional_args.bm_num:
                                    break

                        break

        corpus = np.array(list(set([example for example in validation_dataset['context']])))

        p_with_neg_val = []
        print("prepare_in_batch_negative for validation")
                    
        with tqdm(validation_dataset['context'], unit="batch") as tepoch:
            for idx, c in enumerate(tepoch):
            
                while True:
                    neg_idxs = np.random.randint(len(corpus), size=self.num_neg)

                    if not c in corpus[neg_idxs]:
                        p_neg = corpus[neg_idxs]

                        p_with_neg_val.append(c)
                        p_with_neg_val.extend(p_neg)
                        

                        # BM25로 질문과 유사도 높은 지문 negative sample로 추가 (--bm_num으로 몇개 추가할건지 설정가능)
                        if self.bm25:
                            cnt=0
                            top_n_passages=self.BM25(self.validation_bm25, validation_dataset['question'][idx], corpus)
                            
                            for p in top_n_passages:
                                if p!=c: # and dataset['answers'][idx]['text'][0] not in p:
                                    p_with_neg_val.append(p)
                                    cnt+=1
                                if cnt==self.additional_args.bm_num:
                                    break

                        break

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs_train = tokenizer(train_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs_train = tokenizer(p_with_neg_train, padding="max_length", truncation=True, return_tensors='pt')

        q_seqs_val = tokenizer(validation_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs_val = tokenizer(p_with_neg_val, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs_train['input_ids'].size(-1)
        p_seqs_train['input_ids'] = p_seqs_train['input_ids'].view(-1, num, max_len)
        p_seqs_train['attention_mask'] = p_seqs_train['attention_mask'].view(-1, num, max_len)
        p_seqs_train['token_type_ids'] = p_seqs_train['token_type_ids'].view(-1, num, max_len)

        max_len = p_seqs_val['input_ids'].size(-1)
        p_seqs_val['input_ids'] = p_seqs_val['input_ids'].view(-1, num, max_len)
        p_seqs_val['attention_mask'] = p_seqs_val['attention_mask'].view(-1, num, max_len)
        p_seqs_val['token_type_ids'] = p_seqs_val['token_type_ids'].view(-1, num, max_len)

        train_dataset = TensorDataset(
            p_seqs_train['input_ids'], p_seqs_train['attention_mask'], p_seqs_train['token_type_ids'], 
            q_seqs_train['input_ids'], q_seqs_train['attention_mask'], q_seqs_train['token_type_ids']
        )

        validation_dataset = TensorDataset(
            p_seqs_val['input_ids'], p_seqs_val['attention_mask'], p_seqs_val['token_type_ids'], 
            q_seqs_val['input_ids'], q_seqs_val['attention_mask'], q_seqs_val['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)
        self.validation_dataloader = DataLoader(validation_dataset, shuffle=False, batch_size=self.args.per_device_train_batch_size)


    def train(self, args=None):

        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        t_total = len(self.train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor =0.3, verbose=True)
        # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0
        global_loss = 100
        loss = 0.0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        if self.bm25:
            num=self.num_neg+1+self.additional_args.bm_num
        else:
            num=self.num_neg+1

        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for idx, batch in enumerate(tepoch):

                    self.p_encoder.train()
                    self.q_encoder.train()
                    
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (num), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (num), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (num), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, num, -1)
     
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets) / args.gradient_accumulation_steps

                    loss.backward()
                    if (global_step+1) % args.gradient_accumulation_steps == 0:             # Wait for several backward steps
                        optimizer.step()                            # Now we can do an optimizer step
                        self.p_encoder.zero_grad()
                        self.q_encoder.zero_grad()

                    if self.additional_args.wandb=='True':
                        wandb.log({"loss": loss})

                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    # evaluation 
                    if global_step%100==0 and global_step>0:
                        loss_sum=0
                        with tqdm(self.validation_dataloader, unit="batch") as tepoch:
                            for idx, batch in enumerate(tepoch):
                                with torch.no_grad():
                                    self.p_encoder.eval()
                                    self.q_encoder.eval()
                                    
                                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                                    targets = targets.to(args.device)

                                    p_inputs = {
                                        'input_ids': batch[0].view(batch_size * (num), -1).to(args.device),
                                        'attention_mask': batch[1].view(batch_size * (num), -1).to(args.device),
                                        'token_type_ids': batch[2].view(batch_size * (num), -1).to(args.device)
                                    }
                            
                                    q_inputs = {
                                        'input_ids': batch[3].to(args.device),
                                        'attention_mask': batch[4].to(args.device),
                                        'token_type_ids': batch[5].to(args.device)
                                    }
                            
                                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                                    # Calculate similarity score & loss
                                    p_outputs = p_outputs.view(batch_size, num, -1)
                    
                                    q_outputs = q_outputs.view(batch_size, 1, -1)

                                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                                    sim_scores = sim_scores.view(batch_size, -1)
                                    sim_scores = F.log_softmax(sim_scores, dim=1)

                                    # loss = F.nll_loss(sim_scores, targets)
                                    loss_sum+=F.nll_loss(sim_scores, targets)

                        print("[evaluation loss]",(loss_sum/idx).item())
                        # loss 값 제일 낮을 때 encoder 저장
                        if loss_sum/idx<global_loss:
                            global_loss=loss_sum/idx
                            # model save
                            save_path= self.additional_args.save_dir
                            os.makedirs(save_path, exist_ok=True)
                            self.p_encoder.save_pretrained(os.path.join(save_path, f"p_encoder-{self.additional_args.report_name}"))
                            self.q_encoder.save_pretrained(os.path.join(save_path, f"q_encoder-{self.additional_args.report_name}"))

                        scheduler.step((loss_sum/idx).item())

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  

        p_embs = torch.tensor(self.p_embs, device='cpu').squeeze()
        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1)).squeeze()
        rank = torch.argsort(dot_prod_scores, descending=True) 

        doc_score = dot_prod_scores[rank].tolist()[:k]
        doc_indices = rank.tolist()[:k]

        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:

        args = self.args
        p_encoder = self.p_encoder
        q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer(queries, padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_embs = q_encoder(**q_seqs_val).to('cpu')  

        p_embs = torch.tensor(self.p_embs, device='cpu').squeeze()
        dot_prod_scores = torch.matmul(q_embs, torch.transpose(p_embs, 0, 1)).squeeze()

        rank = torch.argsort(dot_prod_scores, descending=True) 

        tensor_stack=[]
        for i in range(dot_prod_scores.size(0)):
            tensor_stack.append(dot_prod_scores[i,rank[i,:k]])

        doc_scores=torch.stack(tensor_stack, dim=0).tolist()
        doc_indices = rank[:,:k].tolist()
        
        return doc_scores, doc_indices
  
class BertEncoder(BertPreTrainedModel):

    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()
      
    def forward(
            self,
            input_ids, 
            attention_mask=None,
            token_type_ids=None
        ): 
  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        return pooled_output

def main(args):

    # arguments.py 참고 (--epochs --batch_size --num_neg --save_dir --report_name --project_name --wandb --test_query --bm25 --bm_num --dataset --topk)
    # [example] python dense_retrieval.py --report_name HY-BERT_baseline_wiki_BM25_ex3 --bm25 True --num_neg 4 --bm_num 2 --dataset wiki --wandb True
    
    if args.wandb=='True':
        wandb.init(project=args.project_name, entity="salt-bread", name=args.report_name)
    
    print(args)

    # 대회 데이터셋 불러오기
    if args.dataset=='wiki':
        dataset_train = load_from_disk("../data/train_dataset/")
        train_dataset = dataset_train

    # korQuad 불러오기
    if args.dataset=='squad_kor_v1':
        train_dataset = load_dataset("squad_kor_v1")['train']

    train_args = TrainingArguments(
        output_dir= args.save_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size, # 아슬아슬합니다. 작게 쓰세요 !
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
    )

    # Train 
    model_checkpoint = 'klue/bert-base'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)

    retriever = DenseRetrieval(args=[train_args, args], dataset=train_dataset, bm25=args.bm25 ,num_neg=args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.train()

    if args.test_query=='True':
    
        model_checkpoint = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        p_encoder = BertEncoder.from_pretrained(os.path.join(args.save_dir, f"p_encoder-{args.report_name}")).to(train_args.device)
        q_encoder = BertEncoder.from_pretrained(os.path.join(args.save_dir, f"q_encoder-{args.report_name}")).to(train_args.device)

        # squad_kor_v1 데이터셋
        # dataset = load_dataset("squad_kor_v1")['train']

        # 대회 validation set
        dataset = load_from_disk("../data/train_dataset/")
        # dataset = dataset['validation']

        retriever = DenseRetrieval(args=[train_args, args], dataset=dataset, bm25=args.bm25, num_neg=args.num_neg, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder,mode='test')

        # # 단일 쿼리 테스트 (str)
        # index = 0
        # doc_scores, doc_indices = retriever.get_relevant_doc(query=dataset[index]['question'] ,k=args.topk)

        # print(f"[Search Query] {dataset[index]['question']}\n")
        # print(f"[Passage] {dataset[index]['context']}\n")

        # for i, idx in enumerate(doc_indices):
        #     print(f"Top-{i + 1}th Passage (Score {doc_scores[i]})")
        #     pprint(retriever.dataset['context'][idx])

        # # 다중 쿼리 테스트 (List)
        # doc_scores, doc_indices = retriever.get_relevant_doc_bulk(queries=[dataset[index]['question'],dataset[index+1]['question'],dataset[index+2]['question']] ,k=args.topk)

        # for i in range(len(doc_indices)):
        #     for j, idx in enumerate(doc_indices[i]):
        #         pprint(retriever.dataset['question'][idx])
        #         print(f"Top-{j + 1}th Passage (Score {doc_scores[i][j]})")
        #         pprint(retriever.dataset['context'][idx])
        #     print("---------------------------------------------------------------------")

if __name__ == '__main__':
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments)
    )
    args, _ = parser.parse_args_into_dataclasses()
    main(args)
    