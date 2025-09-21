from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "/home/ubuntu/model/llava-hf/llava-v1.5-7b"

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)