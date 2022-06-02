
import torch.nn as nn

from transformers import BertForNextSentencePrediction, BertConfig


# 모델 클래스
class CustomSTS(nn.Module):
    def __init__(self, hidden_size: int, model_name):
        super(CustomSTS, self).__init__()
        self.bert_config = BertConfig.from_pretrained(model_name)   
        self.model = BertForNextSentencePrediction.from_pretrained(model_name, config=self.bert_config)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        """
        outputs(NextSentencePredictorOutput) : logtis, loss(next_sentence_label이 주어질 때 return)
                                              hidden_states(optional), attentions(optional) 을 가지고 있다.
        loss는 주어진 label이 0~5 사이의 값으로 scale 되어있기 때문에 직접 구해야한다!
        """
        # logits's shape : (batch_size, 2)
        logits = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        probs = self.softmax(logits)
        probs = probs[:, 0] * 5    # 0~5 사이의 값으로 정답(T)일 확률 뽑아내기
        return probs    # 정답(T)일 확률, 정답일때 1