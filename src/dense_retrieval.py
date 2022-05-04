import json
import random
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pprint import pprint
import wandb
import argparse
import os 
from datasets import load_from_disk

from sklearn.feature_extraction.text import TfidfVectorizer

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BertModel, BertPreTrainedModel,
    AdamW, get_linear_schedule_with_warmup,
    TrainingArguments,
)
from rank_bm25 import BM25Okapi

# 난수 고정
def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    random.seed(random_seed)
    np.random.seed(random_seed)
    
set_seed(42) # magic number :)

print ("PyTorch version:[%s]."%(torch.__version__))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print ("device:[%s]."%(device))


class DenseRetrieval:
    def __init__(self, args, dataset, num_neg, tokenizer, p_encoder, q_encoder):

        '''
        학습과 추론에 사용될 여러 셋업을 마쳐봅시다.
        '''
        self.args = args[0]
        self.additional_args = args[1]
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        self.prepare_in_batch_negative(num_neg=num_neg)

        self.bm25 = None # BM25Okapi(tokenized_corpus)

    def split_space(self, sent):
        return sent.split(" ")

    def BM25(self, query):

        corpus = np.array(list(set([example for example in self.dataset['context']])))

        tokenized_corpus = [self.split_space(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = self.split_space(query)

        doc_scores = self.bm25.get_scores(tokenized_query)

        top_n_passages = self.bm25.get_top_n(tokenized_query, corpus, n=10)
        
        return top_n_passages

    def prepare_in_batch_negative(self, dataset=None, num_neg=2, tokenizer=None):

        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        # CORPUS를 np.array로 변환해줍니다.        
        corpus = np.array(list(set([example for example in dataset['context']])))

        tokenized_corpus = [tokenizer(doc) for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)

        p_with_neg = []

        for c in dataset['context']:
            
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_neg = corpus[neg_idxs]

                    p_with_neg.append(c)
                    p_with_neg.extend(p_neg)
                    break
                    

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
        p_seqs = tokenizer(p_with_neg, padding="max_length", truncation=True, return_tensors='pt')

        max_len = p_seqs['input_ids'].size(-1)
        p_seqs['input_ids'] = p_seqs['input_ids'].view(-1, num_neg+1, max_len)
        p_seqs['attention_mask'] = p_seqs['attention_mask'].view(-1, num_neg+1, max_len)
        p_seqs['token_type_ids'] = p_seqs['token_type_ids'].view(-1, num_neg+1, max_len)

        train_dataset = TensorDataset(
            p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )

        self.train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=self.args.per_device_train_batch_size)

        valid_seqs = tokenizer(dataset['context'], padding="max_length", truncation=True, return_tensors='pt')
        passage_dataset = TensorDataset(
            valid_seqs['input_ids'], valid_seqs['attention_mask'], valid_seqs['token_type_ids']
        )

        self.passage_dataloader = DataLoader(passage_dataset, batch_size=self.args.per_device_train_batch_size)


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
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

        # Start training!
        global_step = 0
        global_loss = 100

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")

        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for idx, batch in enumerate(tepoch):

                    self.p_encoder.train()
                    self.q_encoder.train()
                    
                    targets = torch.zeros(batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        'input_ids': batch[0].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'attention_mask': batch[1].view(batch_size * (self.num_neg + 1), -1).to(args.device),
                        'token_type_ids': batch[2].view(batch_size * (self.num_neg + 1), -1).to(args.device)
                    }
            
                    q_inputs = {
                        'input_ids': batch[3].to(args.device),
                        'attention_mask': batch[4].to(args.device),
                        'token_type_ids': batch[5].to(args.device)
                    }
            
                    p_outputs = self.p_encoder(**p_inputs)  # (batch_size*(num_neg+1), emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)  # (batch_size*, emb_dim)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
     
                    q_outputs = q_outputs.view(batch_size, 1, -1)
            

                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()  #(batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)

                    if self.additional_args.wandb=='True':
                        wandb.log({"loss": loss})

                    # loss 값 제일 낮을 때 encoder 저장
                    if loss<global_loss:
                        global_loss=loss
                        # model save
                        save_path= './' + self.additional_args.save_dir
                        os.makedirs(save_path, exist_ok=True)
                        self.p_encoder.save_pretrained(os.path.join(save_path, f"p_encoder-{self.additional_args.report_name}"))
                        self.q_encoder.save_pretrained(os.path.join(save_path, f"q_encoder-{self.additional_args.report_name}"))

                    tepoch.set_postfix(loss=f'{str(loss.item())}')

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs


    def get_relevant_doc(self, query, k=1, args=None, p_encoder=None, q_encoder=None):

        if args is None:
            args = self.args

        if p_encoder is None:
            p_encoder = self.p_encoder

        if q_encoder is None:
            q_encoder = self.q_encoder

        with torch.no_grad():
            p_encoder.eval()
            q_encoder.eval()

            q_seqs_val = self.tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to(args.device)
            q_emb = q_encoder(**q_seqs_val).to('cpu')  # (num_query=1, emb_dim)

            p_embs = []
            for batch in self.passage_dataloader:

                batch = tuple(t.to(args.device) for t in batch)
                p_inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]
                }
                p_emb = p_encoder(**p_inputs).to('cpu')
                p_embs.append(p_emb)

        p_embs = torch.stack(p_embs, dim=0).view(len(self.passage_dataloader.dataset), -1)  # (num_passage, emb_dim)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
        rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
        return rank[:k]

  
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
    
    if args.wandb=='True':
        wandb.init(project="MRC_retrieval", entity="salt-bread", name="BERT_squad_kor_v1_batch2")

    # 대회 데이터셋 불러오기
    dataset_train = load_from_disk("../data/train_dataset/")
    train_dataset = dataset_train['train']

    # korQuad 불러오기
    # train_dataset = load_dataset("squad_kor_v1")['train']

    train_args = TrainingArguments(
        output_dir="dense_retireval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size, # 아슬아슬합니다. 작게 쓰세요 !
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=2,
        weight_decay=0.01,
    )

    # Train 
    model_checkpoint = 'klue/bert-base'

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(train_args.device)

    retriever = DenseRetrieval(args=[train_args,args], dataset=train_dataset, num_neg=1, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)
    retriever.train()

    if args.test_query=='True':
    
        model_checkpoint = 'klue/bert-base'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        # p_encoder-squad_kor_v1_batch2
        p_encoder = BertEncoder.from_pretrained("/opt/ml/input/code/dense_retrieval/p_encoder-squad_kor_v1_batch2").to(train_args.device)
        q_encoder = BertEncoder.from_pretrained("/opt/ml/input/code/dense_retrieval/q_encoder-squad_kor_v1_batch2").to(train_args.device)

        # Retriever는 아래와 같이 사용할 수 있도록 코드를 짜봅시다.
        dataset_val = load_from_disk("../data/train_dataset/")
        dataset_val = dataset_val['train']
        retriever = DenseRetrieval(args=[train_args,args], dataset=dataset_val, num_neg=1, tokenizer=tokenizer, p_encoder=p_encoder, q_encoder=q_encoder)

        index = 200
        # query = '모든 악티늄족의 공통적인 자기적 성질은?' 
        query = dataset_val[index]['question']
        passage = dataset_val[index]['context']
        results = retriever.get_relevant_doc(query=query, k=args.topk)

        print(f"[Search Query] {query}\n")
        print(f"[Passage] {passage}\n")

        indices = results.tolist()
        for i, idx in enumerate(indices):
            print(f"Top-{i + 1}th Passage (Index {idx})")
            pprint(retriever.dataset['context'][idx])


if __name__ == '__main__':
    # wandb.login()

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default="./dense_retrieval")
    parser.add_argument('--report_name', type=str)
    parser.add_argument('--project_name', type=str, default="MRC_retrieval")
    parser.add_argument('--wandb', type=str, default="True")
    parser.add_argument('--test_query', type=str, default="True")
    
    args = parser.parse_args()
    main(args)