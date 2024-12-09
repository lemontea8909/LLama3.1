import random
import numpy as np
import torch
from datasets import Dataset

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(0)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", 
    quantization_config=quantization_config,
    device_map="auto"
)

PAD_TOKEN = "<|pad|>"

tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

# we added a new padding token to the tokenizer, we have to extend the embddings
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

print(tokenizer.pad_token, tokenizer.pad_token_id)
# output: ('<|pad|>', 128256)

dataset = Dataset.load_from_disk("./your_huggingface_dataset")   # change to your dataset path (huggingface dataset)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training
)

# this is recommended by original lora paper: using lora, we should target the linear layers only
lora_config = LoraConfig(
    r=16,  # rank for matrix decomposition
    lora_alpha=16,
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj"
    ],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.CAUSAL_LM
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

print(model.print_trainable_parameters())
# output: trainable params: 83,886,080 || all params: 8,114,212,864 || trainable%: 1.0338
from trl import SFTConfig, SFTTrainer

OUTPUT_DIR = "./llama3-test"   # output dir for lora

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field='content',  # this is the final text example we formatted
    max_seq_length=4096,
    num_train_epochs=1,
    per_device_train_batch_size=2,  # training batch size
    gradient_accumulation_steps=4,  # by using gradient accum, we updating weights every: batch_size * gradient_accum_steps = 4 * 2 = 8 steps
    optim="paged_adamw_8bit",  # paged adamw
    save_steps=0.2,  # save every 20% of the trainig steps
    logging_steps=10,
    learning_rate=1e-4,
    fp16=True,  # also try bf16=True
    save_strategy='steps',
    warmup_ratio=0.1,  # learning rate warmup
    save_total_limit=2,
    lr_scheduler_type="cosine",  # scheduler
    save_safetensors=True,  # saving to safetensors
    dataset_kwargs={
        "add_special_tokens": False,  # we template with special tokens already
        "append_concat_token": False,  # no need to add additional sep token
    },
    seed=1
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)


trainer.train()



