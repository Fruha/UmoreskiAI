<p align="center">
    <img src="https://github.com/Fruha/UmoreskiAI/blob/master/git_images/umoreski.jpg" width="50%">
</p>


## Description
- **Parsing** vk group for dataset
- Fine tuning **BertForSequenceClassification**
- Uploading on **Huggingface** 🤗

## Pipeline

https://user-images.githubusercontent.com/57077738/230724600-aca4664d-4723-4358-a19d-cea9d1ca8fea.mp4

## Usage

### Online
https://huggingface.co/Fruha/UmoreskiAI

### Downloading model
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("Fruha/UmoreskiAI")

model = AutoModelForSequenceClassification.from_pretrained("Fruha/UmoreskiAI")
```
## Training results

| Training Loss | Epoch | Step | Validation Loss | Roc auc | Maxf1  | Best Threshold |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:------:|:--------------:|
| 0.1013        | 1.0   | 993  | 0.0932          | 0.7972  | 0.4358 | 0.1875         |
| 0.0876        | 2.0   | 1986 | 0.0956          | 0.8014  | 0.4348 | 0.1332         |


## Plot of metrics

<p align="center">
    <img src="https://github.com/Fruha/UmoreskiAI/blob/master/git_images/curves.png" width="80%">
</p>