import torch
from torch import nn
from transformers import AutoModelForMaskedLM, AutoTokenizer, pipeline, AutoModel
from mosaic_bert import BertModel

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

model = AutoModel.from_pretrained("/home/shreyansh/long_context_biencoder_v2/models/bert-base-1024-biencoder-6M-pairs", trust_remote_code=True).to("cuda")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = AutoModelForSentenceEmbedding(model, tokenizer)

def get_embeddings(text):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=1024, return_tensors='pt').to("cuda")
    print(encoded_input['input_ids'].shape)
    # Compute token embeddings
    with torch.no_grad():
        sentence_embeddings = model(**encoded_input)

    return sentence_embeddings

query_text = "Is there a place which I can call heaven on earth"

long_text = open('text.txt').read()
# long_text = "My wifi is not working"
# long_text_v1 = query_text + " " + long_text
# long_text_v2 = long_text + " " + query_text

doc_embedding_v0 = get_embeddings(long_text)
# doc_embedding_v1 = get_embeddings(long_text_v1)
# doc_embedding_v2 = get_embeddings(long_text_v2)

query_embedding = get_embeddings(query_text)

simm0 = doc_embedding_v0 @ query_embedding.T
# simm1 = doc_embedding_v1 @ query_embedding.T
# simm2 = doc_embedding_v2 @ query_embedding.T

print(simm0)
# print(simm1)
# print(simm2)

from sentence_transformers import SentenceTransformer
sentences = [query_text, long_text]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings[0] @ embeddings[1])