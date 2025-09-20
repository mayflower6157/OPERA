from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import torch

# Choose the LLaVA checkpoint
model_id = "llava-hf/llava-1.5-7b-hf"

# Load processor + model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load image from Hugging Face (or any URL)
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
image = Image.open(requests.get(url, stream=True).raw)

# Prompt
prompt = "What do you see in this picture?"

# Preprocess + run
inputs = processor(prompt, image, return_tensors="pt").to("cuda")
output = model.generate(**inputs, max_new_tokens=128)

print(processor.batch_decode(output, skip_special_tokens=True)[0])