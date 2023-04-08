<p align="center">
    <img src="https://github.com/Fruha/UmoreskiAI/blob/master/git_images/umoreski.jpg" width="50%">
</p>


## Description
- **Parsing** vk group for dataset
- Fine tuning **BertForSequenceClassification**
- Uploading on **Huggingface** 🤗

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

## Pipeline



## Plot of metrics

<p align="center">
    <img src="https://github.com/Fruha/UmoreskiAI/blob/master/git_images/curves.png" width="80%">
</p>