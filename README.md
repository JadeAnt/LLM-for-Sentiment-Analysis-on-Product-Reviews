# LLM for Sentiment Analysis on Product Reviews

Team members: Jaden Antoine (jja435), Ritvik Nair (rn2520)

## Table of Contents

1. Objective
2. Milestones
3. How to Run Code
4. Results
5. Repository Structure

## Objective

Our main goal with this project is to create an optimized Large Language Model (LLM) that is fine-tuned to perform sentiment analysis on e-commerce product reviews. The final models will be capable of providing fast and accurate sentiment classification making it suitable for use in a variety of applications.

We finetuned multiple LLMs on e-commerce reviews and improved the inference speed while maintaining accurate results. Due to size of the models we had to be wary of the time it will take to train and evaluate them, as LLM’s are known for having large parameter counts. One of the challenges we faced was model evaluation. We had to ensure that the models were consistent in its results and does not give out too many false positive results. Another aspect we had to be careful of is over fitting the models on a specific set of data. The model sshould not give accurate results for just one type of product or review. Furthermore, our models should be able to give not only positive or negative sentimental outputs, but neutral as well. We performed this experiment on the following LLMs; BERT, GPT2, and LLAMA3.2-1B.

## Milestones

For our project we outlined a series of steps we needed to take in order to complete our experiment. We were able to successfully complete each step and have our results displayed below.

1. **Preprocess Dataset**: We preprocessed a 21,000 AmazonReviewsDataset. The dataset contained a variety of details such as account location, profile, review title, description, and score. Out of these, we considered only the review title, description, and score to be relevant to our project. We combined the title and description together and divided the score into three sections based on positive, neutral, and negative sentiments.
2. **Test using BERT**: We loaded BERT into a Google Colab environment and tested its sentimental analysis capabilities. We fine tuned it with LORA and discovered that the model still performed as a 94.2% accuracy.
3. **Apply LORA, Quantization, and Pruning**: We applied LORA, Quantization, and Pruning to the BERT model. We then retrained the model to ensure consistency in our model accuracy. We maintained a similar accuracy as before of 92.1%. Our model size was reduced by 67.7%.
4. **Perform experiment with GPT-2**: We converted the code for BERT to perform the same for GPT-2. and successfully applied LORA, Quantization and Pruning to the GPT-2 model, following the same steps we performed with the BERT model. We fine tuned it with LORA and got an accuracy of 91.23%. We maintained a similar accuracy as before of 89.82%. Our model size was reduced by 58.1%.
5. **Perform experiment with LLAMA**: Using a LLAMA3.2-1B model we were able to perform the same optimizations we did with BERT and GPT2. We fine tuned it with LORA and got an accuracy of 93.64%. We maintained a similar accuracy as before of 94.02%. Our model size was reduced by 68.8%.
6. **Perform experiment with any additional LLMS if we have time**: We tested the experiment using LLAMA-7B, however, while we were able to apply LORA to the model, we were unable to prune or quantize the model due to time and compute resource constraints.
7. **Perform experiment on HPC**: We performed our experiments on the HPC with a distirbuted training method using 4 L4 GPUs on a g2-standard-48 partition, 128 GB of memory, 12 CPUs, and 3 hours of runtime.
8. **Save results**: We saved the results of our experiments using log files from Wandb, Tensorboard, and the Pytorch Profiler.

## How to Run Code

### Batch Script

To run our code there is a supplied batch script called `hpml_project_dis.sh`. Within this batch script you can supply the necessary amount of GPU's, memory, CPUs, and time to run to replicate our results. The generated output from this batch file can be found in `hpml_project_dis.out`.

As we are working with LLMs there is a large compute requirement to complete the training for even one of these models. As such here are the recommended parameters for running our code:

- Number of GPUs: 4
- Number of CPUs: 1 - 16
- Memory: 128 GB
- Time to run: 4 Hours

### Arguments

Our code contains a `main.py` file that handles any input arguments you may want to change for running the models using ArgumentParser.

An example call to run our code is as follows:
`torchrun --nproc_per_node=[NUM_PROCESSES] main.py --num_workers [NUM_WORKERS] --batchsize [BATCH_SIZE] --model [MODEL_NAME] --wandb [WANDB_API_KEY];`

Due to the large size of the LLAMA model we chose to switch to a distributed training method with 4 GPUs, requiring us to use `torchrun` to run our code.

The input parameters for the code are:

- num_workers: Number of workers for dataloading
- model: Model type to run optimizations on (bert, gpt2, llama)
- batchsize: Batch size for training
- wandb: API key for recording results of runs

### Commands

The command parameters (except the model) are free to change for each model type, below however, is the exact setup we used to get our results for our code and is recommended you use the same setup for your own experiments.

To run BERT:
`torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 16 --model bert --wandb [WANDB_API_KEY];`

To run GPT2:
`torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 8 --model gpt2 --wandb  [WANDB_API_KEY];`

To run LLAMA:
NOTE: To access the LLAMA-1B model you may need to get a Hugging Face API token and request access to model make sure to set your hugging face token in `main.py`:
`os.environ['HF_TOKEN'] = [API_KEY_HERE]`

Command to run:
`torchrun --nproc_per_node=4 main.py --num_workers 2 --batchsize 8 --model llama --wandb  [WANDB_API_KEY];`

## Results

From running the commands and batchscript above you will have generated a final optimized version of the model called `fine_tuned_optimized_[MODEL_NAME]`, log files, wandb runs, and the `hpml_project_dis.out` file for that run.

Below are the results we got from running our code with the same aforementioned commands.

| Model Size  | Unoptimized (MB) | Optimized (MB) | Change (%) |
| ----------- | ---------------- | -------------- | ---------- |
| BERT        | 439.851          | 142.057        | -67.70     |
| GPT2        | 487.843          | 204.343        | -58.11     |
| LLAMA3.2-1B | 4717.557         | 1469.557       | -68.85     |

We see that after applying our optimizations we get very large decreases in model size. With approximately 2/3 of each model being reduced. With such small model sizes, these models would be able to be deployed on edge devices and platforms with more limited GPU and memory constraints without issue.

| Accuracy    | After LORA (%) | After Pruning + Quantization | Change (%) |
| ----------- | -------------- | ---------------------------- | ---------- |
| BERT        | 94.2           | 92.1                         | -2.23      |
| GPT2        | 91.23          | 89.82                        | -1.55      |
| LLAMA3.2-1B | 93.64          | 94.02                        | 0.41       |

However, it's interesting to note that even despite our large decreases in model size, the change in accuracy is very little. With LLAMA3.2-1B even getting a 0.41% increase in accuracy despite 68.85% of it model size being reduced

| Profiler    | CPU Time Total (s) | Model Inference (ms) | CPU Mem (MB) |
| ----------- | ------------------ | -------------------- | ------------ |
| BERT        | 12.566             | 53.2                 | 51.15        |
| GPT2        | 33.128             | 44.016               | 96.61        |
| LLAMA3.2-1B | 5.618              | 70.021               | 520.36       |

Finally, we see that the CPU time for each optimized model is extremely small. With, suprisingly, LLAMA3.2-1B being the fastest model CPU-wise, but the slowest in terms of model inference. LLAMA also still takes up alot of CPU memory.

This does prove however, that the optimizations made to the LLMs have an enourmous effect and can greatly improve model speed and reduce model size with minimal changes in accuracy.

## Repository Structure

Below is the outline of the structure of the repository. The key files and folders here are `main.py`, `utils.py`, `notebooks` folder, and all of the generated folders with names `fine_tuned_optimized_{Model_name}`, `logs_{Model_name}`, `lora_{Model_name}`, and `lora_pruned_{Model_name}`.

The generated folders `fine_tuned_optimized_{Model_name}`, `logs_{Model_name}`, `lora_4bit_{Model_name}`, and `lora_pruned_{Model_name}` hold the trained version of the model at different stages of the training pipeline, and the associated log files for each model.

```
Project
|   bert_gpt2_run.out
|   hpml_project_dis.sh
|   llama_run.out
|   main.py
|   output.txt
|   README.md
|   utils.py
|
+---Figures
|   +---bert
|   |   |   bert_profiling.PNG
|   |   |   c30d2cc9a85e_3592.1733886994979634727.pt.trace.json
|   |   |
|   |   +---lora
|   |   |       bert_lora_eval_acc.png
|   |   |       bert_lora_eval_loss.png
|   |   |       bert_lora_train_loss.png
|   |   |       bert_lora_train_lr.png
|   |   |
|   |   \---quantization_pruning
|   |           bert_quant_eval_acc.png
|   |           bert_quant_eval_loss.png
|   |           bert_quant_train_loss.png
|   |           bert_quant_train_lr.png
|   |
|   +---gpt2
|   |   |   c30d2cc9a85e_3592.1733887334243629990.pt.trace.json
|   |   |   gpt2_profiling.PNG
|   |   |
|   |   +---lora
|   |   |       gpt_lora_eval_acc.png
|   |   |       gpt_lora_eval_loss.png
|   |   |       gpt_lora_train_loss.png
|   |   |
|   |   \---quantization_pruning
|   |           gpt_quant_eval_acc.png
|   |           gpt_quant_eval_loss.png
|   |           gpt_quant_train_loss.png
|   |
|   \---llama
|       |   3f084057175a_479.1733941612213113239.pt.trace.json
|       |   llama_profiling.PNG
|       |
|       +---lora
|       |       llama_lora_eval_acc.png
|       |       llama_lora_eval_loss.png
|       |       llama_lora_train_loss.png
|       |
|       \---quantization_pruning
|               llama_quant_eval_acc.png
|               llama_quant_eval_loss.png
|               llama_quant_train_loss.png
|
+---fine_tuned_optimized_bert
|   |   adapter_config.json
|   |   adapter_model.safetensors
|   |
|   +---checkpoint-1000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-1500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-2000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-2500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-266
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-3000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-3183
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-532
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   \---checkpoint-798
|           adapter_config.json
|           adapter_model.safetensors
|           optimizer.pt
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|           vocab.txt
|
+---fine_tuned_optimized_gpt2
|   |   adapter_config.json
|   |   adapter_model.safetensors
|   |
|   +---checkpoint-1062
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       merges.txt
|   |       optimizer.pt
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.json
|   |
|   +---checkpoint-1593
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       merges.txt
|   |       optimizer.pt
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.json
|   |
|   \---checkpoint-531
|           adapter_config.json
|           adapter_model.safetensors
|           merges.txt
|           optimizer.pt
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|           vocab.json
|
+---fine_tuned_optimized_llama
|   |   adapter_config.json
|   |   adapter_model.safetensors
|   |   README.md
|   |
|   +---checkpoint-1062
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |
|   +---checkpoint-1593
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |
|   \---checkpoint-531
|           adapter_config.json
|           adapter_model.safetensors
|           optimizer.pt
|           README.md
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|
+---logs_bert
|       events.out.tfevents.1733890831.b-33-15.6438.0
|       events.out.tfevents.1733890981.b-33-15.6438.2
|       events.out.tfevents.1733891173.b-33-15.6438.3
|       events.out.tfevents.1733930841.b-33-11.6433.0
|       events.out.tfevents.1733930993.b-33-11.6433.2
|       events.out.tfevents.1733931186.b-33-11.6433.3
|
+---logs_gpt2
|       events.out.tfevents.1733891203.b-33-15.7684.0
|       events.out.tfevents.1733891377.b-33-15.7684.2
|       events.out.tfevents.1733891609.b-33-15.7684.3
|       events.out.tfevents.1733931217.b-33-11.7676.0
|       events.out.tfevents.1733931390.b-33-11.7676.2
|       events.out.tfevents.1733931624.b-33-11.7676.3
|
+---logs_llama
|       events.out.tfevents.1733814629.b-33-34.6469.0
|       events.out.tfevents.1733847945.b-33-25.6407.0
|       events.out.tfevents.1733852449.b-33-18.6442.0
|       events.out.tfevents.1733858485.b-33-16.6488.0
|       events.out.tfevents.1733865417.b-33-25.6430.0
|       events.out.tfevents.1733873460.b-33-23.6438.0
|       events.out.tfevents.1733882742.b-33-39.6391.0
|       events.out.tfevents.1733894749.b-33-11.6431.0
|       events.out.tfevents.1733897720.b-33-23.6460.0
|       events.out.tfevents.1733898811.b-33-23.6460.2
|       events.out.tfevents.1733900565.b-33-23.6460.3
|
+---lora_4bit_bert
|   +---checkpoint-1000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-1500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-2000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-2500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-266
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-3000
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-3183
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-500
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   +---checkpoint-532
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.txt
|   |
|   \---checkpoint-798
|           adapter_config.json
|           adapter_model.safetensors
|           optimizer.pt
|           README.md
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|           vocab.txt
|
+---lora_4bit_gpt2
|   +---checkpoint-1062
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       merges.txt
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.json
|   |
|   +---checkpoint-1593
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       merges.txt
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |       vocab.json
|   |
|   \---checkpoint-531
|           adapter_config.json
|           adapter_model.safetensors
|           merges.txt
|           optimizer.pt
|           README.md
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|           vocab.json
|
+---lora_4bit_llama
|   +---checkpoint-1062
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |
|   +---checkpoint-1593
|   |       adapter_config.json
|   |       adapter_model.safetensors
|   |       optimizer.pt
|   |       README.md
|   |       rng_state_0.pth
|   |       rng_state_1.pth
|   |       rng_state_2.pth
|   |       rng_state_3.pth
|   |       scheduler.pt
|   |       special_tokens_map.json
|   |       tokenizer.json
|   |       tokenizer_config.json
|   |       trainer_state.json
|   |       training_args.bin
|   |
|   \---checkpoint-531
|           adapter_config.json
|           adapter_model.safetensors
|           optimizer.pt
|           README.md
|           rng_state_0.pth
|           rng_state_1.pth
|           rng_state_2.pth
|           rng_state_3.pth
|           scheduler.pt
|           special_tokens_map.json
|           tokenizer.json
|           tokenizer_config.json
|           trainer_state.json
|           training_args.bin
|
+---lora_bert
|       adapter_config.json
|       adapter_model.safetensors
|       README.md
|       special_tokens_map.json
|       tokenizer.json
|       tokenizer_config.json
|       vocab.txt
|
+---lora_gpt2
|       adapter_config.json
|       adapter_model.safetensors
|       merges.txt
|       README.md
|       special_tokens_map.json
|       tokenizer.json
|       tokenizer_config.json
|       vocab.json
|
+---lora_llama
|       adapter_config.json
|       adapter_model.safetensors
|       README.md
|       special_tokens_map.json
|       tokenizer.json
|       tokenizer_config.json
|
+---lora_pruned_bert
|       adapter_config.json
|       adapter_model.safetensors
|       README.md
|
+---lora_pruned_gpt2
|       adapter_config.json
|       adapter_model.safetensors
|       README.md
|
+---lora_pruned_llama
|       adapter_config.json
|       adapter_model.safetensors
|       README.md
|
+---notebooks
|       Demo.ipynb
|       pytorch_profiling.ipynb
|
\---__pycache__
        utils.cpython-312.pyc
```
