# LLM-for-Sentiment-Analysis-on-Product-Reviews 
# By: Jaden Antoine (jja435) and Ritvik Nair (rn2520)
# High Performance Machine Learning Fall 2024
# 
# Run this code to apply LORA, Pruning, and Quantization on one of three
# LLMs; BERT, GPT2, and LLAMA3.2-1B
#
# main.py
# Driver code to run fine tuning of these models

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import evaluate
import accelerate

# For Pruning
import torch.nn.utils.prune as prune
import torch_pruning as tp

# For Quantization and LoRA
from transformers import BertForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import pandas as pd
import numpy as np
import kagglehub

import matplotlib.pyplot as plt
import seaborn as sns
import os
import random

import argparse
import time

import wandb

# import functions from utils
import utils
from utils import *

def main():
    print("########## Running Experiment LLM ##########")
    # Setup for Argument Parsing from CLI
    parser = argparse.ArgumentParser(description="Apply LORA, Pruning, and Quantization to LLMs")
    parser.add_argument('--num_workers', type=int, default= 2, help='Number of data loading workers')
    parser.add_argument('--batchsize', type=int, default=16, help='Number of batch size')
    parser.add_argument('--model', type=str, default="bert", choices=["bert", "gpt2","llama"], help='Model to use')
    parser.add_argument('--wandb', type=str, default="None", help='WandB API Key')
    args = parser.parse_args()

    _BATCH_SIZE = args.batchsize # batch size for training and eval
    _NUM_WORKERS = args.num_workers # num workers for dataloaders
    _MODEL = args.model  # model type so we know which model to load
    _WANDB_TOKEN = args.wandb # Wandb api key to log runs

    wandb.login(key=_WANDB_TOKEN)
    os.environ['HF_TOKEN'] = "" #NEEDED FOR LLAMA3B VERSION

    # Retreive Model and Tokenizer according to model input argument
    model, tokenizer = get_model_tokenizer(_MODEL, False)

    # Get the dataset, train, evaluation, and test datasets and dataloaders, passing the tokenizer and number of workers for dataloaders
    base_dataset, train_dataset, eval_dataset, test_dataset, train_loader, eval_loader, test_loader = create_dataset_dataloaders(_NUM_WORKERS, tokenizer)

    # Apply LORA to the model
    model = apply_LORA(_MODEL, model)

    # Train Model
    train_model(_MODEL, model, train_dataset, eval_dataset, tokenizer, _BATCH_SIZE)

    # Evaluate Model
    evaluate_model(_MODEL, model, test_dataset, _BATCH_SIZE)

    # Prune the model (30% prune percentage)
    model = apply_Pruning(_MODEL, model)

    # Quantize the model (4-bit quantization)
    quantized_model = apply_Quantization(_MODEL, model)

    # Finetune the model on the dataset performing both training and evaluation once again
    post_quantization_finetuning(_MODEL, model, quantized_model, tokenizer, base_dataset, 160, _BATCH_SIZE)


if __name__ == '__main__':
    main()
