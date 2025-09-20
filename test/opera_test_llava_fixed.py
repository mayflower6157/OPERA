from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import requests
import torch

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Load processor with slow tokenizer to avoid ModelWrapper error
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Example image from Hugging Face dataset
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # Prompt
    prompt = "Describe this image in detail."

    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt").to("cuda")

    # Generate output
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\n=== Model Output ===")
    print(response)

if __name__ == "__main__":
    main()
