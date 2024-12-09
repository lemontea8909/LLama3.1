# LLama3.1

# Hugging Face Dataset and Training Pipeline

This repository provides a pipeline for training a model, and performing inference.

## Dataset Format

The dataset should be structured as a JSON file with the following format:

```json
[
    {
        "content": "This paper shows that ..."
    },
    {
        "content": "This is second content ..."
    },
]
```
## Directory Structure
```
.
├── train.py          # Script to configure and run training
├── train.sh          # Shell script to execute training
├── inference.py      # Script for inference
├── dataset/          # Directory for the prepared dataset
└── README.md         # Documentation
```
## For training 
First replace the path in train.py
```python
dataset = "your dataset path"
output_dir = "your output path"
```

## Run Training
```python
python train.sh
```

## Run Inference
Replace the checkpoint path and user prompt
```python
NEW_MODEL = "your-output-dir/checkpoint-xxx"  # xxx is the checkpoint iterations  please check outputdir yourself
user = "your question / prompt to ask Llama"
```
And run the inference
```python
python inference.py
```



