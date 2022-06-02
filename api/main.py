import os
import sys

_CUR_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0,_CUR_PATH)

import numpy as np
import torch
from model import CustomSTS
from transformers import BertTokenizer

from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


##### PRE-LODAD #####
# check device
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# load checkpoint file    
_CKPT_PATH = os.path.join(_CUR_PATH, "./") # 나의 checkpoint 파일 위치 (변경 필수)
ckpt = torch.load(os.path.join(_CKPT_PATH, "sample_model.ckpt"), map_location=device)


# 모델 사전 로드 (필수)
model = CustomSTS(hidden_size=768, model_name = 'klue/bert-base')
model.load_state_dict(ckpt["model_state_dict"])

# Tokenzier 사전 로드 (필수)
tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
##### PRE-LODAD #####

class Data(BaseModel):
    sentence1: str
    sentence2: str


@app.post("/")
def classifier(request: Data):

    # 기본적인 전처리
    data = [
        request.sentence1.strip(),
        request.sentence2.strip()
    ]
    data1 = request.sentence1.strip()
    data2 = request.sentence2.strip()

    # 토크나이징
    tensorized_data = tokenizer(
        data1, data2,
        add_special_tokens=True,
        padding="longest",  # 배치내 가장 긴 문장을 기준으로 부족한 문장은 [PAD] 토큰을 추가
        truncation=True, # max_length를 넘는 문장은 이 후 토큰을 제거함
        max_length=512,
        return_tensors='pt' # 토크나이즈된 결과 값을 텐서 형태로 반환
    )
    
    # Inference
    with torch.no_grad():
        logits = model(**tensorized_data)


    score = logits.item()

    return {"Sentence1" : data1, "Sentence2": data2, "Score" : score}
