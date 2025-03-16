from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel

# Load tokenizer and base model
base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
fine_tuned_dir = "./output"

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_dir)
model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModel.from_pretrained(model, fine_tuned_dir)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Inference
instruction = "How do I calibrate Machine H sensors?"
prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=35, do_sample=True)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 Response:\n", response)
