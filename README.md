# Long Context Biencoder

****
Models on Huggingface Hub  
* [bert-base-1024-biencoder-6M-pairs](https://huggingface.co/shreyansh26/bert-base-1024-biencoder-6M-pairs)
* [bert-base-1024-biencoder-64M-pairs](https://huggingface.co/shreyansh26/bert-base-1024-biencoder-64M-pairs)
****

This repository has the code for training, inference and evaluation of long context biencoders based on [MosaicML's BERT pretrained on 1024 sequence length](https://huggingface.co/mosaicml/mosaic-bert-base-seqlen-1024). This model maps sentences & paragraphs to a 768 dimensional dense vector space 
and can be used for tasks like clustering or semantic search.

## Usage

### Download the model and related scripts
```git clone https://huggingface.co/shreyansh26/bert-base-1024-biencoder-6M-pairs``` / ```git clone https://huggingface.co/shreyansh26/bert-base-1024-biencoder-64M-pairs```

### Inference
```python
import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, AutoModel
from mosaic_bert import BertModel

# pip install triton==2.0.0.dev20221202 --no-deps if using Pytorch 2.0

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model, tokenizer, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = model.to("cuda")
        self.normalize = normalize
        self.tokenizer = tokenizer

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

model = AutoModel.from_pretrained("<path-to-model>", trust_remote_code=True).to("cuda")
model = AutoModelForSentenceEmbedding(model, tokenizer)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

sentences = ["This is an example sentence", "Each sentence is converted"]

encoded_input = tokenizer(sentences, padding=True, truncation=True, max_length=1024, return_tensors='pt').to("cuda")
embeddings = model(**encoded_input)

print(embeddings)
print(embeddings.shape)
```

## Other details

### Training

This model has been trained on 6.4M randomly sampled pairs of sentences/paragraphs from the same training set that Sentence Transformers models use. Details of the
training set [here](https://huggingface.co/sentence-transformers/all-mpnet-base-v2#training-data). 

* Training script - [train_biencoder.py](train_biencoder.py)
* Data Utils (important) - [data_utils.py](data_utils.py)
* Inference/Testing - [test_biencoder.py](test_biencoder.py)
* Benchmarking scripts - [models/evaluate_models_mteb.py](models/evaluate_models_mteb.py)

### Evaluations

We ran the model on a few retrieval based benchmarks (CQADupstackEnglishRetrieval, DBPedia, MSMARCO, QuoraRetrieval) and the results are [here](https://github.com/shreyansh26/Long-Context-Biencoder/tree/master/models/results/).
