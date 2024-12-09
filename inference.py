import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

NEW_MODEL="llama3-fake2-5-r32/checkpoint-3600"  # replace the output-dir/checkpoint-xx

# load trained/resized tokenizer
tokenizer = AutoTokenizer.from_pretrained(NEW_MODEL)

# here we are loading the raw model, if you can't load it on your GPU, you can just change device_map to cpu
# we won't need gpu here anyway
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map='auto',
)

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(model, NEW_MODEL)
model = model.merge_and_unload()

system = "You are a Artificial Intelligence assistant and willing to answer the question from the user."  
user = "user's question"   #  replace your question / prompt
# use chat template
text = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>  
<|start_header_id|>user<|end_header_id|>{user}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>\n'''   

device = "cuda" if torch.cuda.is_available() else "cpu"

inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=4096,)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
