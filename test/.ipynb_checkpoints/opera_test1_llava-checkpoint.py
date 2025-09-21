from transformers import LlavaProcessor, LlamaTokenizer, LlavaForConditionalGeneration, CLIPImageProcessor
from PIL import Image
import requests
import torch

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Load components separately to avoid fast tokenizer issues
    tokenizer = LlamaTokenizer.from_pretrained(model_id, legacy=True)
    image_processor = CLIPImageProcessor.from_pretrained(model_id)
    
    # Load model
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Example image from Hugging Face dataset
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # Prompt (proper format for LLaVA)
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    # Process text and image separately 
    text_inputs = tokenizer(prompt, return_tensors="pt")
    image_inputs = image_processor(image, return_tensors="pt")
    
    # Combine inputs
    inputs = {
        'input_ids': text_inputs['input_ids'],
        'attention_mask': text_inputs['attention_mask'],
        'pixel_values': image_inputs['pixel_values']
    }
    
    # Move to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\n=== Model Output ===")
    print(response)

if __name__ == "__main__":
    main()