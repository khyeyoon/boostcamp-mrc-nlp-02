import pandas as pd
import math
import random
import re
import copy

from datasets import DatasetDict, load_from_disk, load_metric, load_dataset, Dataset, DatasetDict
from kiwipiepy import Kiwi
from tqdm import tqdm

# kiwi 호출 (문장 분리)
kiwi = Kiwi()

# context에 answer_token 추가
def add_answer_token(examples):
    ans_context_list = []
    context_column_name = 'context'
    contexts = examples[context_column_name]
    
    for i in range(len(contexts)):
        answer_start = datasets['train'][i]['answers']['answer_start'][0]
        ans_context = datasets['train'][i]['context'][:answer_start] + '[ANSWER]' + datasets['train'][i]['context'][answer_start:]
        ans_context_list.append(ans_context)
        
    return ans_context_list

# 단일 문서 -> sentence_list
# 특수문자 제거 전처리
def split_to_sentences(context):#, data_args):    
    sentence_list = []
    sentences = kiwi.split_into_sents(context)
    
    for sentence in sentences:
        sentence = re.sub(r'n\\n', '', sentence.text)
        sentence_list.append(re.sub(r'[@%\\*=()~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '', sentence)) # 전처리
        sentence = re.sub(r'[@%\\*=~#&\+á?\xc3\xa1\-\|\:\;\!\-\,\_\~\$\'\"]', '', sentence) # 소괄호, 마침표 살리기
        if sentence.find('[ANSWER]') != -1: # 정답이 있는 문장 앞뒤에 특수문자 붙이기
            sentence = '@'+sentence+'#' 
        sentence_list.append(sentence)
    
    return sentence_list

## 특정 비율 만큼 문장의 순서가 섞인 Dataset 반환
def mix_sentences(examples, ratio):
    answers = []
    context = []
    document_id = []
    ID = []
    question = []
    title = []

    contexts = add_answer_token(examples)

    for i in tqdm(range(len(contexts))):
        sentence_list = split_to_sentences(contexts[i])
        
        idx_list = list(range(len(sentence_list)))
        mix_num = math.ceil(len(sentence_list)*ratio)
        mix_idx = random.sample(idx_list, mix_num)
        sorted_mix_idx = sorted(mix_idx)

        for j in range(mix_num):
            idx_list[sorted_mix_idx[j]] = mix_idx[j]

        org_context = ' '.join([sentence_list[i] for i in range(len(sentence_list))])
        mixed_context = ' '.join([sentence_list[idx_list[i]] for i in range(len(sentence_list))])
        
        org_answer_start = org_context.find('[ANSWER]')
        mx_answer_start = mixed_context.find('[ANSWER]')
        org_context = re.sub('\[ANSWER\]', '', org_context)
        mixed_context = re.sub('\[ANSWER\]', '', mixed_context)

        org_answer = copy.deepcopy(examples[i]['answers'])
        org_answer['answer_start'] = [org_answer_start]
        mx_answer = copy.deepcopy(examples[i]['answers'])
        mx_answer['answer_start'] = [mx_answer_start]

        answers.append(org_answer)
        context.append(org_context)
        document_id.append(examples[i]['document_id'])
        ID.append(examples[i]['id'])
        question.append(examples[i]['question'])
        title.append(examples[i]['title'])

        if org_context != mixed_context:
            answers.append(mx_answer)
            context.append(mixed_context)
            document_id.append(examples[i]['document_id'])
            ID.append(examples[i]['id'])
            question.append(examples[i]['question'])
            title.append(examples[i]['title'])

    if len(answers) == len(document_id):
        result = {'answers': answers,
                'context': context,
                'document_id': document_id,
                'id': ID,
                'question': question,
                'title': title}

        return Dataset.from_dict(result)
    else:
        print("생성된 데이터의 크기가 맞지 않습니다.")

datasets = load_from_disk("/opt/ml/input/data/train_dataset")
mix_dataset = mix_sentences(datasets['train'], 0.2)
valid_dataset = datasets['validation']
new_datasets = DatasetDict({"train":mix_dataset, "validation":valid_dataset})
new_datasets.save_to_disk("/opt/ml/input/level2-mrc-level2-nlp-02/src/org_mix_datasets")
# new_datasets.push_to_hub("salt-bread/mixed_wiki", private=True)