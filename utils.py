# LLM-for-Sentiment-Analysis-on-Product-Reviews 
# By: Jaden Antoine (jja435) and Ritvik Nair (rn2520)
# High Performance Machine Learning Fall 2024
# 
# Run this code to apply LORA, Pruning, and Quantization on one of three
# LLMs; BERT, GPT2, and LLAMA3.2-1B
#
# utils.py
# Holds all functions needed to perform experiment

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
from transformers import BertForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoConfig, AutoModel
from transformers import Trainer, TrainingArguments
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaMLP
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftConfig, PeftModel

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

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get metric accuracy for model training and evaluation
metric = evaluate.load("accuracy")

# Custom Dataset class to use for converting AmazonReviewsDataset into data format to feed to model
class AmazonReviewDataset(Dataset):
    # Constructor
        def __init__(self, reviews, targets, tokenizer, max_len):
            self.reviews = reviews
            self.targets = targets
            self.tokenizer = tokenizer
            self.max_len = max_len

        # Length method
        def __len__(self):
            return len(self.reviews)

        # get item method
        def __getitem__(self, item):
            review = str(self.reviews[item])
            target = int(self.targets[item])

            # Encoded format to return
            #encoding = self.tokenizer.encode_plus(
            #    review,
            #    add_special_tokens=True,
            #    max_length=self.max_len,
            #    return_token_type_ids=False,
            #    truncation=True,
            #    #pad_to_max_length=True,
            #    #padding = True,
            #    padding='max_length',
            #    return_attention_mask=True,
            #    return_tensors='pt',
            #)

            encoding = self.tokenizer(
                review,
                max_length = self.max_len,
                truncation= True,
                padding='max_length',
                return_tensors='pt'
            )

            return {
                'text': review,
                #'input_ids': encoding['input_ids'].flatten(),
                #'attention_mask': encoding['attention_mask'].flatten(),
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': torch.tensor(target, dtype=torch.long)
                #'labels': target
            }

# Convert rating "1-5 stars" from dataset into numerical value for sentiment analysis
def rating_to_sentiment(rating):
        rating = int(rating)

        if(rating <= 2):
            return 0
        elif(rating == 3):
            return 1
        else:
            return 2

# Return the correct model and tokenizer according to argument input from CLI
def get_model_tokenizer(model_name, quantize=False):
    print("########## GET MODEL AND TOKENIZER ##########")

    if(model_name == "bert"):
        # Load BERT model for our analysis
        MODEL_NAME = "bert-base-uncased"
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels = 3
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if(model_name == "gpt2"):
        # Load GPT2 model for our analysis
        MODEL_NAME = "gpt2"
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels = 3)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
         
    if(model_name == "llama"):
        MODEL_NAME = "meta-llama/Llama-3.2-1B"

        # If limited GPU and/or memory, set quantize=True to load smaller version of LLAMA model
        if(quantize):
            # 4-bit Quantization config for LLAMA in case we dont have access to multiple GPU's or Memory, to load smaller version
            # of LLAMA model instead
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,  # Enable 4-bit quantization
                bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype to FP16 for efficiency
                bnb_4bit_use_double_quant=True,  # Use double quantization for better precision
                bnb_4bit_quant_type="nf4"  # Use NormalFloat4 (NF4), which often improves quantization accuracy
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels = 3,
                quantization_config=quantization_config
                )
        else: #otherwise just load the entire model
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels = 3)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    print(model.num_parameters())
    return model, tokenizer

# Return dataset and dataloaders
def create_dataset_dataloaders(num_workers, tokenizer):
    print("########## GET DATASET ##########")

    # Download latest version from Kaggle
    path = kagglehub.dataset_download("dongrelaxman/amazon-reviews-dataset")

    print("Path to dataset files:", path)

    # Preprocess Data

    # Set in a dataframe
    # Combine review title + review description
    # Convert “Rated X out of 5 stars” to 0, 1, 2 (positive, neutral, negative

    amazon_reviews = pd.read_csv(f'{path}/Amazon_Reviews.csv', engine='python')

    amazon_reviews.head()

    len(amazon_reviews)

    amazon_reviews_cleaned = pd.DataFrame({
        "Review Combined": amazon_reviews['Review Title'] + " | " + amazon_reviews['Review Text'],
        "Review Title" : amazon_reviews['Review Title'],
        "Review Text" : amazon_reviews['Review Text'],
        "Rating": amazon_reviews['Rating']
        }
    )

    amazon_reviews_cleaned.head()

    len(amazon_reviews_cleaned)

    # Rated 1 out of 5 stars, Some ratings are "None", mapped to 0 in cleaned dataset
    
    ratings = {
        None: 0,
        "Rated 1 out of 5 stars": 1,
        "Rated 2 out of 5 stars": 2,
        "Rated 3 out of 5 stars": 3,
        "Rated 4 out of 5 stars": 4,
        "Rated 5 out of 5 stars": 5
    }

    rating_cleaned = [ratings[i] for i in amazon_reviews_cleaned["Rating"]]

    amazon_reviews_cleaned["Rating_cleaned"] = rating_cleaned

    amazon_reviews_cleaned.head()

    amazon_reviews_cleaned.isnull().sum()

    amazon_reviews_cleaned.shape

    sns.countplot(amazon_reviews_cleaned, x = "Rating_cleaned")
    plt.xlabel("Review Score")

    
    amazon_reviews_cleaned["sentiment"] = amazon_reviews_cleaned.Rating_cleaned.apply(rating_to_sentiment)

    amazon_reviews_cleaned.head()

    amazon_reviews_cleaned.sentiment.dtypes

    amazon_reviews_cleaned["sentiment"]

    class_names = ["negative", "neutral", "positive"]
    ax = sns.countplot(amazon_reviews_cleaned, x = "sentiment")
    plt.xlabel('Review sentiment')
    ax.set_xticklabels(class_names)


    # Split Dataset into Train, Validation, and Test

    SEED = 42
    amRev_train, amRev_test = train_test_split(amazon_reviews_cleaned, test_size=0.2, random_state=SEED)
    amRev_eval, amRev_test = train_test_split(amRev_test, test_size=0.5, random_state=SEED)

    print(amRev_train.shape, amRev_test.shape, amRev_eval.shape)

    # Create Dataset and Dataloaders

    BATCH_SIZE = 16
    NUM_WORKERS = num_workers
    MAX_LEN = 160 # token count contained within reviews

    # Create Train, eval, and test datasets
    train_dataset = AmazonReviewDataset(
            reviews=amRev_train["Review Combined"].to_numpy(),
            targets=amRev_train.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    eval_dataset = AmazonReviewDataset(
            reviews=amRev_eval["Review Combined"].to_numpy(),
            targets=amRev_eval.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    test_dataset = AmazonReviewDataset(
            reviews=amRev_test["Review Combined"].to_numpy(),
            targets=amRev_test.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers = NUM_WORKERS)

    unique_labels = set([example["labels"] for example in train_dataset])
    print(f"Unique labels in the dataset: {unique_labels}")
    print(len(unique_labels))

    unique_labels = set([example for example in amRev_train.sentiment])
    print(f"Unique labels in the dataset: {unique_labels}")
    print(len(unique_labels))

    return amazon_reviews_cleaned, train_dataset, eval_dataset, test_dataset, train_loader, eval_loader, test_loader

######## LORA ########
# Applies LORA to models with rank 8, alpha of 16, dropout 0.1, no bias, and SEQ_CLS task type

def apply_LORA(model_name, model):
    print("########## APPLY LORA ##########")
    if(model_name == "bert"):
        lora_config = LoraConfig(
            r=8,  # Rank of the low-rank updates
            lora_alpha=16,  # Scaling factor for LoRA
            target_modules=["query", "key", "value"],  # Apply LoRA to specific layers for bert
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
    if(model_name == "gpt2"):
        lora_config = LoraConfig(
            r=8,  # Rank of the low-rank updates
            lora_alpha=16,  # Scaling factor for LoRA
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
    if(model_name == "llama"):
        lora_config = LoraConfig(
            r=8,  # Rank of the low-rank updates
            lora_alpha=16,  # Scaling factor for LoRA
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )

    # Apply LoRA configuration to model
    lora_model = get_peft_model(model, lora_config)
    return lora_model


# Used to print model size in megabytes
def print_size_of_model(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size (MB): {:.3f}MB'.format(size_all_mb))

# Use to compute accuracy metric during per-epoch training
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) 
    return metric.compute(predictions=predictions, references=labels)

# Train model on train and eval sets
def train_model(model_name, model, train_dataset, eval_dataset, tokenizer, batch_size):
    
    # initialize wandb run to record results
    run = wandb.init(
        project = "LLM For Sentiment Analysis on Product Reviews",
        name= f"{model_name} LORA Training"
    )

    # Sample training configuration
    training_args = TrainingArguments(
        output_dir=f"./lora_4bit_{model_name}",
        per_device_train_batch_size= batch_size,
        per_device_eval_batch_size=batch_size,
        label_names = ["labels"],
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay = 0.01,
        logging_dir=f"./logs_{model_name}",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy = "epoch",
        push_to_hub=False,
        report_to= ["tensorboard", "wandb"], # save results for wandb and tensorboard
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False  # Distributed Data Parallel
    )


    # Define Trainer
    # Call train and test datasets from AmazonReviewDataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.can_return_loss = True

    # Train model with LoRA
    trainer.train()

    wandb.finish() # Finish wandb run

    # Save the LoRA-adapted model
    model.save_pretrained(f"./lora_{model_name}")
    tokenizer.save_pretrained(f"./lora_{model_name}")
    print(f"Model saved to ./lora_{model_name}")

    print_size_of_model(model)

# Evaluate the model using test set
def evaluate_model(model_name, model, test_dataset, batch_size):
    training_args = TrainingArguments(
        output_dir=f"./lora_4bit_{model_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        label_names = ["labels"],
        num_train_epochs=3,
        learning_rate=2e-5,
        weight_decay = 0.01,
        logging_dir="./logs",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy = "epoch",
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
        ddp_find_unused_parameters=False  # Distributed Data Parallel implementation
    )

    # Setup trainer for evaluation
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate on Test Set
    result = trainer.evaluate()

    print(f"Accuracy after LORA: {result['eval_accuracy']}")

######## Pruning #############
# This is an unstructured pruning approach. Where we prune 30% of neurons from each intermediate dense layer for each model

# Function to prune neurons from intermediate dense layers for BERT
def prune_intermediate_neurons(model, amount=0.2):
    # Prunes a percentage of neurons from each intermediate dense layer

    for i, layer in enumerate(model.bert.encoder.layer):

        # Apply unstructured pruning on the intermediate dense layer
        prune.l1_unstructured(layer.intermediate.dense, name="weight", amount=amount)

        # Remove pruning mask, leave pruned weights
        prune.remove(layer.intermediate.dense, "weight")

# Function to prune neurons from intermediate dense layers for GPT2
def prune_intermediate_neurons_gpt2(model, amount=0.2):
    # Prunes a percentage of neurons from each intermediate dense layer (c_fc in GPT-2's MLP submodule)

    for i, block in enumerate(model.transformer.h):
        # Access the intermediate dense layer for GPT2 c_fc
        dense_layer = block.mlp.c_fc
        
        # Apply unstructured pruning
        prune.l1_unstructured(dense_layer, name="weight", amount=amount)
        
        # Remove pruning mask, leave pruned weights
        prune.remove(dense_layer, "weight")
    
    print(f"Pruned {amount * 100}% of neurons from each intermediate dense layer.")

# Function to prune neurons from intermediate dense layers for LLAMA
def prune_intermediate_neurons_llama(prune_model, amount=0.2):
    # Prunes a percentage of neurons from the intermediate dense layers (gate_proj) in the feed-forward network of a LLaMA model

    pruned_layers = 0

    # Go through all the modules to find LlamaMLP layers
    for name, module in prune_model.named_modules():
        if isinstance(module, LlamaMLP):
            try:
                # Access the intermediate dense layer for llama gate_proj
                dense_layer = module.gate_proj

                # Apply unstructured L1 pruning
                prune.l1_unstructured(dense_layer, name="weight", amount=amount)

                # Remove the pruning mask, leave pruned weights
                prune.remove(dense_layer, "weight")

                pruned_layers += 1
                print(f"Pruned {amount * 100}% of neurons in layer: {name}.gate_proj")
            except AttributeError:
                print(f"Layer {name} does not have 'gate_proj'. Skipping.")

    if pruned_layers == 0:
        print("No layers with 'gate_proj' were found for pruning.")
    else:
        print(f"Pruned {amount * 100}% of neurons in {pruned_layers} layers.")

# Function to call to apply correct pruning function to model
def apply_Pruning(model_name, model):
    print("########## APPLY PRUNING ##########")
    print(model.num_parameters())

    prune_percentage = 0.3  # Prune 30% of neurons in intermediate dense layers

    # Perform Pruning by model type
    if(model_name=="bert"):
        prune_intermediate_neurons(model, prune_percentage)
    if(model_name == "gpt2"):
        prune_intermediate_neurons_gpt2(model, prune_percentage)
    if(model_name == "llama"):
        prune_intermediate_neurons_llama(model, prune_percentage)

    # Save the pruned model
    output_dir = f"./lora_pruned_{model_name}"
    model.save_pretrained(output_dir)
    print(f"Pruned model saved to {output_dir}")

    print(model.num_parameters())

    print_size_of_model(model)
    return model

######## Quantization #############
# We perform 4-bit quantization on each model

# Applies quantization to pretrained version of the model
def apply_Quantization(model_name, model):
    
    print("########## APPLY QUANTIZATION ##########")

    # Set up quantization configuration for 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_compute_dtype=torch.float16,  # Set compute dtype to FP16 for efficiency
        bnb_4bit_use_double_quant=True,  # Use double quantization for better precision
        bnb_4bit_quant_type="nf4"  # Use NormalFloat4 (NF4), which often improves quantization accuracy
    )

    # Setup model for 4-bit quantization by model type

    if(model_name == "bert"):
        # Call trained bert model
        quantized_model = BertForSequenceClassification.from_pretrained(f"./lora_pruned_{model_name}/",
                                                    quantization_config= quantization_config,
                                                    num_labels=3,
                                                    local_files_only=True)
    if(model_name == "gpt2"):
        # Use config from trained GPT2
        config = AutoConfig.from_pretrained(f"./lora_pruned_{model_name}")
        config.num_labels = 3

        # Call trained gpt2 model
        quantized_model = AutoModelForSequenceClassification.from_pretrained(f"./lora_pruned_{model_name}/",
                                                    config=config,
                                                    quantization_config=quantization_config,
                                                    local_files_only=True)
        # Set padding token to eos token for model
        quantized_model.config.pad_token_id = quantized_model.config.eos_token_id
    if(model_name == "llama"):
        #Set path to folder that contains adapter_config.json and the associated .bin files for the Peft model
        peft_model_id = f'./lora_pruned_{model_name}'

        #Get PeftConfig from the finetuned Peft Model. This config file contains the path to the base model
        config = PeftConfig.from_pretrained(peft_model_id)

        #Load the base model
        quantized_model = AutoModelForSequenceClassification.from_pretrained(
            peft_model_id,
            quantization_config=quantization_config,
            use_auth_token=True,
            num_labels=3,
            ignore_mismatched_sizes=True
        )

        # Load quantized model as PeftModel due to LORA modifications
        quantized_model = PeftModel.from_pretrained(quantized_model, peft_model_id)
        # Set padding token to eos token for model
        quantized_model.config.pad_token_id = quantized_model.config.eos_token_id

    print(quantized_model.num_parameters())

    print_size_of_model(quantized_model)
    return quantized_model

# Finetune the quantized, pruned, and LORA'd model
def post_quantization_finetuning(model_name, model, quantized_model, tokenizer, amazon_reviews_cleaned, MAX_LEN, batch_size):

    # Prepare quantized model for training again
    quantized_model = prepare_model_for_kbit_training(quantized_model)
    
    # Generate New Dataset to finetune model on
    import random
    SEED = random.randint(1, 100)
    print("Seed: " + str(SEED))
    amRev_train, amRev_test = train_test_split(amazon_reviews_cleaned, test_size=0.2, random_state=SEED)
    amRev_eval, amRev_test = train_test_split(amRev_test, test_size=0.5, random_state=SEED)

    print(amRev_train.shape, amRev_test.shape, amRev_eval.shape)

    # Create Train, evaluation, test dataset
    train_dataset = AmazonReviewDataset(
            reviews=amRev_train["Review Combined"].to_numpy(),
            targets=amRev_train.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    eval_dataset = AmazonReviewDataset(
            reviews=amRev_eval["Review Combined"].to_numpy(),
            targets=amRev_eval.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    test_dataset = AmazonReviewDataset(
            reviews=amRev_test["Review Combined"].to_numpy(),
            targets=amRev_test.sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=MAX_LEN
    )

    # Initialize new wandb run
    run = wandb.init(
        project = "LLM For Sentiment Analysis on Product Reviews",
        name= f"{model_name} Post Quantization Fine Tuning"
    )

    # Defining new fine-tuning training arguments
    training_args = TrainingArguments(
        output_dir=f"./fine_tuned_optimized_{model_name}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        label_names = ["labels"],
        num_train_epochs=3,
        learning_rate=5e-6,
        weight_decay=0.01,
        logging_dir=f"./logs_{model_name}",
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        max_grad_norm=1.0,  # Gradient clipping for stability
        push_to_hub=False,
        report_to= ["tensorboard", "wandb"],
        load_best_model_at_end=False, #set false to stop loading after training
        ddp_find_unused_parameters=False  # Distributed Data Parallel implementation
    )

    # Setup trainer function with train and eval datasets
    trainer = Trainer(
        model=quantized_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    # Set model to prepare to be trained
    quantized_model.train()

    # Train model
    trainer.train()

    # Finish WandB run
    wandb.finish()

    # Evaluate Model

    trainer = Trainer(
        model=quantized_model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate on Test Set
    result = trainer.evaluate()

    print(f"Accuracy after Quantization: {result['eval_accuracy']}")

    # Save the pruned model
    output_dir = f"./fine_tuned_optimized_{model_name}"
    quantized_model.save_pretrained(output_dir)
    print(f"Pruned model saved to {output_dir}")

    print_size_of_model(model)

    print_size_of_model(quantized_model)