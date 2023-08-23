# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import random
import evaluate
import torch
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, set_seed

from accelerate import Accelerator, DistributedType
from torch.utils.data.distributed import DistributedSampler
from torch import nn
from data_utils import get_dataset_for_dataloaders
from mosaic_bert import BertModel
from tqdm import tqdm

########################################################################
# This is a fully working simple example to use Accelerate
#
# This example trains a Bert base model on GLUE MRPC
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - (multi) TPUs
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, follow the instructions
# in the readme for examples:
# https://github.com/huggingface/accelerate/tree/main/examples
#
########################################################################


MAX_GPU_BATCH_SIZE = 32

class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, tokenizer, accelerator, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()

        self.model = BertModel.from_pretrained(model_name, trust_remote_code=True, return_dict=True).to(accelerator.device)
        self.normalize = normalize
        self.tokenizer = tokenizer
        self.accelerator = accelerator

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

    def save_pretrained(self, output_path):
        if self.accelerator.is_main_process:
            self.tokenizer.save_pretrained(output_path)
            self.model.config.save_pretrained(output_path)

            torch.save(self.model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

def get_dataloaders(config, accelerator: Accelerator, batch_size: int = 16):
    """
    Creates a set of `DataLoader`s for the `glue` dataset,
    using "bert-base-cased" as the tokenizer.

    Args:
        accelerator (`Accelerator`):
            An `Accelerator` object
        batch_size (`int`, *optional*):
            The batch size for the train and validation DataLoaders.
    """
    # Instantiate dataloaders.
    datasets_for_dataloader_cols2, datasets_for_dataloader_cols3, weights_cols2, weights_cols3 = get_dataset_for_dataloaders(config, batch_size)

    if accelerator.num_processes > 1:
        train_sampler_cols2 = DistributedSampler(
                datasets_for_dataloader_cols2,
                rank=accelerator.local_process_index,
                num_replicas=accelerator.num_processes,
                shuffle=True,
            )
        train_sampler_cols3 = DistributedSampler(
                datasets_for_dataloader_cols3,
                rank=accelerator.local_process_index,
                num_replicas=accelerator.num_processes,
                shuffle=True,
            )

    if accelerator.num_processes > 1:
        train_dataloader_cols2 = DataLoader(datasets_for_dataloader_cols2, batch_size=batch_size, drop_last=True, sampler=train_sampler_cols2)
        train_dataloader_cols3 = DataLoader(datasets_for_dataloader_cols3, batch_size=batch_size, drop_last=True, sampler=train_sampler_cols3)
        # train_dataloader_cols2 = DataLoader(datasets_for_dataloader_cols2, batch_size=batch_size, drop_last=True, shuffle=True)
        # train_dataloader_cols3 = DataLoader(datasets_for_dataloader_cols3, batch_size=batch_size, drop_last=True, shuffle=True)
    else:
        train_dataloader_cols2 = DataLoader(datasets_for_dataloader_cols2, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)
        train_dataloader_cols3 = DataLoader(datasets_for_dataloader_cols3, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=True)

    return train_dataloader_cols2, train_dataloader_cols3, weights_cols2, weights_cols3


def training_function(config, args):
    # Initialize accelerator
    accelerator = Accelerator(cpu=args.cpu, mixed_precision=args.mixed_precision)
    # Sample hyper-parameters for learning rate, batch size, seed and a few other HPs
    lr = config["lr"]
    seed = int(config["seed"])
    batch_size = int(config["batch_size"])

    # If the batch size is too big we use gradient accumulation
    gradient_accumulation_steps = 1
    if batch_size > MAX_GPU_BATCH_SIZE and accelerator.distributed_type != DistributedType.TPU:
        gradient_accumulation_steps = batch_size // MAX_GPU_BATCH_SIZE
        batch_size = MAX_GPU_BATCH_SIZE

    set_seed(seed)
    accelerator.print(f"Batch size: {batch_size}")
    train_dataloader_cols2, train_dataloader_cols3, weights_cols2, weights_cols3 = get_dataloaders(config, accelerator, batch_size)
    # Instantiate the model (we build the model here so that the seed also control new weights initialization)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSentenceEmbedding("mosaicml/mosaic-bert-base-seqlen-1024", tokenizer, accelerator)
    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)

    # Instantiate scheduler
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=config["num_steps"]
    )

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    cross_entropy_loss = nn.CrossEntropyLoss()

    model, optimizer, train_dataloader_cols2, train_dataloader_cols3, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader_cols2, train_dataloader_cols3, lr_scheduler
    )

    # Now we train the model
    model.train()
    losses = []
    for step in range(config["num_steps"]):
        dataloader_choice = random.choices([0, 1], weights=(weights_cols2, weights_cols3))[0]

        if dataloader_choice == 0:
            batch = next(iter(train_dataloader_cols2))

            text1 = tokenizer(batch[0], return_tensors="pt", max_length=config["max_length"], truncation=True, padding="max_length")
            text2 = tokenizer(batch[1], return_tensors="pt", max_length=config["max_length"], truncation=True, padding="max_length")

            ### Compute embeddings
            embeddings_a = model(**text1.to(accelerator.device))
            embeddings_b = model(**text2.to(accelerator.device))

            ### Compute similarity scores 512 x 512
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * config["scale"]
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## Symmetric loss as in CLIP
            loss = (cross_entropy_loss(scores, labels) + cross_entropy_loss(scores.transpose(0, 1), labels)) / 2

        else:
            batch = next(iter(train_dataloader_cols3))
            
            text1 = tokenizer([b[0] for b in batch], return_tensors="pt", max_length=config["max_length"], truncation=True, padding="max_length")
            text2 = tokenizer([b[1] for b in batch], return_tensors="pt", max_length=config["max_length"], truncation=True, padding="max_length")
            text3 = tokenizer([b[2] for b in batch], return_tensors="pt", max_length=config["max_length"], truncation=True, padding="max_length")

            embeddings_a  = model(**text1.to(accelerator.device))
            embeddings_b1 = model(**text2.to(accelerator.device))
            embeddings_b2 = model(**text3.to(accelerator.device))

            embeddings_b = torch.cat([embeddings_b1, embeddings_b2])

            ### Compute similarity scores 512 x 1024
            scores = torch.mm(embeddings_a, embeddings_b.transpose(0, 1)) * config["scale"]
        
            ### Compute cross-entropy loss
            labels = torch.tensor(range(len(scores)), dtype=torch.long, device=embeddings_a.device)  # Example a[i] should match with b[i]
            
            ## One-way loss
            loss = cross_entropy_loss(scores, labels)

        loss = loss / gradient_accumulation_steps
        losses.append(loss)

        accelerator.backward(loss)

        # accelerator.print(f"step {step}, loss {loss}")

        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        #Save model
        if (step+1) % config["log_steps"] == 0:
            output_path = os.path.join(config["output_dir"], str(step+1))
            # Use accelerator.print to print only on the main process.
            accelerator.print(f"step {step}, avg loss {sum(losses) / len(losses)}")
            losses = []
            if (step+1) % config["save_steps"] == 0:
                accelerator.print(f"saving model at {output_path}")
                model.save_pretrained(output_path)


def main():
    parser = argparse.ArgumentParser(description="Simple example of training script.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    args = parser.parse_args()
    config = {
        "lr": 2e-5, 
        "num_steps": 1000000, 
        "seed": 42, 
        "batch_size": 64, 
        "log_steps": 500,
        "save_steps": 50000,
        "scale": 20,
        "output_dir": "long_biencoder_1m",
        "data_config": "data_config.json", 
        "num_cols_config": "num_cols.json", 
        "data_folder": "/home/shreyansh/embedding-training-data",
        "max_length": 1024
    }
    training_function(config, args)


if __name__ == "__main__":
    main()