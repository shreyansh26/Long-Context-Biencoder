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

class MyModel():
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def encode(self, sentences, batch_size=16384, **kwargs):
        """ Returns a list of embeddings for the given sentences.
        Args:
            sentences (`List[str]`): List of sentences to encode
            batch_size (`int`): Batch size for the encoding

        Returns:
            `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
        """
        sentence_embeddings = []
        for idx in range(0, len(sentences), batch_size):
            sent_list = sentences[idx: idx + batch_size]
            encoded_input = tokenizer(sent_list, padding=True, truncation=True, max_length=1024, return_tensors='pt').to("cuda")
            with torch.no_grad():
                embeddings = model(**encoded_input)

            sentence_embeddings.append(embeddings)
        
        return torch.vstack(sentence_embeddings)

model = BertModel.from_pretrained('mosaicml/mosaic-bert-base-seqlen-1024', trust_remote_code=True).to("cuda")
model.load_state_dict(torch.load('/home/shreyansh/long_context_biencoder_v2/models/bert-base-1024-biencoder-64M-pairs/pytorch_model.bin'))
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

model = AutoModelForSentenceEmbedding(model, tokenizer)
encoded_input = tokenizer("I am a boy", padding=True, truncation=True, max_length=1024, return_tensors='pt').to("cuda")
embb1 = model(**encoded_input)
print(embb1.shape)

model2 = AutoModel.from_pretrained("/home/shreyansh/long_context_biencoder_v2/models/bert-base-1024-biencoder-64M-pairs", trust_remote_code=True)

model2 = AutoModelForSentenceEmbedding(model2, tokenizer)
encoded_input = tokenizer("I am a boy", padding=True, truncation=True, max_length=1024, return_tensors='pt').to("cuda")
embb2 = model2(**encoded_input)
print(embb2.shape)

print(torch.allclose(embb1, embb2, atol=1e-4))