## 소개

MRC 대회를 위한 베이스라인 코드 

## 설치 방법

### 요구 사항

```
# data (51.2 MB)
tar -xzf data.tar.gz

# 필요한 파이썬 패키지 설치. 
bash ./install/install_requirements.sh
```

## 파일 구성


### 저장소 구조

```
.
├── code[github repository]
|   ├── assets                          # readme 에 필요한 이미지 저장
|   │   ├── dataset.png
|   │   ├── mrc.png
|   │   └── odqa.png
|   |
|   ├── install                         # 요구사항 설치 파일
|   │   └── install_requirements.sh
|   |
|   ├── src
|   │   ├── utils
|   │   |   ├── __init__.py
|   │   |   ├── arguments.py            # 실행되는 모든 argument가 dataclass 의 형태로 저장되어있음
|   │   |   ├── model.py
|   │   |   ├── trainer_qa.py           # MRC 모델 학습에 필요한 trainer 제공.
|   │   |   └── utils_qa.py             # 기타 유틸 함수 제공 
|   |   |
|   │   ├── __init__.py
|   │   ├── retrieval.py                # sparse retreiver 모듈 제공 
|   │   ├── dense_retrieval.py          # dense retreiver 모듈 제공 
|   │   ├── train.py
|   │   └── inference.py                # ODQA 모델 평가 또는 제출 파일 (predictions.json) 생성
|   |
|   ├── sample_dense_retrieval.sh       # dense_retrieval 실행 sample
|   ├── sample_train.sh                 # train 실행 sample
|   ├── sample_inference.sh             # inference 실행 sample
|   ├── .gitignore
|   └── README.md
|
├── data                                # 전체 데이터, 데이터 소개에서 설명
|   ├── train_dataset                   # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
|   |   ├── train                       
|   |   └── validation
|   |   
|   ├── test_dataset                    # 제출에 사용될 데이터셋. validation 으로 구성 
|   |   └── validation
|   |
|   └── wikipedia_documents.json        # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
|
├── dpr_encoders                        # dpr encoder가 저장되는 dir
|   ├── p_encoder     
│   └── p_encoder
|
├── models                              # train 이후 model이 저장되는 dir
│   └── output
|
└── predictions                         # inference 이후 예측값이 저장되는 dir
    └── prediction

```

## 데이터 소개

아래는 제공하는 데이터셋의 분포를 보여줍니다.

![데이터 분포](./assets/dataset.png)

데이터셋은 편의성을 위해 Huggingface 에서 제공하는 datasets를 이용하여 pyarrow 형식의 데이터로 저장되어있습니다. 다음은 데이터셋의 구성입니다.

```
data                                # 전체 데이터
├── train_dataset                   # 학습에 사용할 데이터셋. train 과 validation 으로 구성 
|   ├── train                       
|   └── validation
|   
├── test_dataset                    # 제출에 사용될 데이터셋. validation 으로 구성 
|   └── validation
|
└── wikipedia_documents.json        # 위키피디아 문서 집합. retrieval을 위해 쓰이는 corpus.
```

data에 대한 argument 는 `arguments.py` 의 `DataTrainingArguments` 에서 확인 가능합니다. 

# 훈련, 평가, 추론

### train

만약 arguments 에 대한 세팅을 직접하고 싶다면 `arguments.py` 를 참고해주세요. 
```bash
python ./src/train.py \
--output_dir ../models/output \
--do_train
```


### eval

MRC 모델의 평가는(`--do_eval`) 따로 설정해야 합니다.  위 학습 예시에 단순히 `--do_eval` 을 추가로 입력해서 훈련 및 평가를 동시에 진행할 수도 있습니다.

```bash
# 학습, mrc 모델 평가 예시 (train_dataset 사용)
python ./src/train.py \
--output_dir "../models/output" \
--model_name_or_path "../models/output" \
--do_eval 
```

### inference

retrieval 과 mrc 모델의 학습이 완료되면 `inference.py` 를 이용해 odqa 를 진행할 수 있습니다.

* 학습한 모델의  test_dataset에 대한 결과를 제출하기 위해선 추론(`--do_predict`)만 진행하면 됩니다. 

* 학습한 모델이 train_dataset 대해서 ODQA 성능이 어떻게 나오는지 알고 싶다면 평가(`--do_eval`)를 진행하면 됩니다.

```bash
# ODQA 실행 (test_dataset 사용)
# wandb 가 로그인 되어있다면 자동으로 결과가 wandb 에 저장됩니다. 아니면 단순히 출력됩니다
python ./src/inference.py \
--output_dir "../models/output" \
--dataset_name "../data/test_dataset" \
--model_name_or_path "../models/output/" \
--do_predict
```
    
### How to submit

`inference.py` 파일을 위 예시처럼 `--do_predict` 으로 실행하면 `--output_dir` 위치에 `predictions.json` 이라는 파일이 생성됩니다. 해당 파일을 제출해주시면 됩니다.

### Things to know

1. `train.py` 에서 sparse embedding 을 훈련하고 저장하는 과정은 시간이 오래 걸리지 않아 따로 argument 의 default 가 True로 설정되어 있습니다. 실행 후 sparse_embedding.bin 과 tfidfv.bin 이 저장이 됩니다. **만약 sparse retrieval 관련 코드를 수정한다면, 꼭 두 파일을 지우고 다시 실행해주세요!** 안그러면 기존 파일이 load 됩니다.

2. 모델의 경우 `--overwrite_cache` 를 추가하지 않으면 같은 폴더에 저장되지 않습니다. 

3. `--output_dir` 폴더 또한 `--overwrite_output_dir` 을 추가하지 않으면 같은 폴더에 저장되지 않습니다.

# sh sample
**<!주의>** sh 명령을 통해 실행하지 않을 경우, path가 정상적으로 작동되지 않을 수 있습니다.

## dense_retrieval sample
```bash
sh sample_dense_retrieval.sh
```
<details>

**<summary> sample_dense_retrieval.sh </summary>**

```bash
python ./src/dense_retrieval.py \
--batch_size 4 \
--bm25 True \
--epochs 3 \
--num_neg 3 --bm_num 2 \
--dataset "wiki" \
--test_query True \
--dpr_gradient_accumulation_steps 16
```
    
</details>

## train sample
```bash
sh sample_train.sh
```
<details>
    
**<summary> sample_train.sh </summary>**
```bash
python ./src/train.py \
--output_dir "../models/output" \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 16 \
--eval_steps 100 --save_steps 100 --save_strategy steps \
--evaluation_strategy steps \
--model_name_or_path "klue/roberta-large" \
--num_train_epochs 2 \
--save_total_limit 3 \
--greater_is_better True \
--metric_for_best_model "exact_match" \
--fp16 True \
--load_best_model_at_end True \
--overwrite_output_dir True \
--do_train --do_eval
```

</details>


    
## inference sample
```bash
sh sample_inference.sh
```
<details>
   
**<summary> sample_inference.sh </summary>**
```bash
python ./src/inference.py \
--model_name_or_path "../models/output" \
--output_dir "../predictions/prediction" \
--dataset_name "../data/test_dataset" \
--per_device_eval_batch_size 64 \
--retrieval "both" \
--fp16 \
--top_k_retrieval 20 \
--do_predict
```
    
</details>

위의 순서대로 실행하면 학습(dpr, reader) 및 추론이 완료됩니다.

sh file 내의 arguments를 변경시키며 여러 실험들을 진행할 수 있습니다.
