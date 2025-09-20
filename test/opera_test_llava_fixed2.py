from transformers import LlavaProcessor, LlamaTokenizer, LlavaForConditionalGeneration
from PIL import Image
import requests
import torch

def main():
    model_id = "llava-hf/llava-1.5-7b-hf"

    # Use specific classes instead of Auto classes to avoid tokenizer format issues
    try:
        # Try with LlavaProcessor first
        processor = LlavaProcessor.from_pretrained(model_id)
        print("✓ Successfully loaded LlavaProcessor")
    except Exception as e:
        print(f"✗ Failed to load LlavaProcessor: {e}")
        # Fallback: load components separately
        from transformers import CLIPImageProcessor
        tokenizer = LlamaTokenizer.from_pretrained(model_id, legacy=True)
        image_processor = CLIPImageProcessor.from_pretrained(model_id)
        print("✓ Loaded tokenizer and image processor separately")
        
        # Create a simple wrapper
        class SimpleProcessor:
            def __init__(self, tokenizer, image_processor):
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                
            def __call__(self, text, images, return_tensors="pt"):
                text_inputs = self.tokenizer(text, return_tensors=return_tensors)
                image_inputs = self.image_processor(images, return_tensors=return_tensors)
                return {**text_inputs, **image_inputs}
                
            def batch_decode(self, *args, **kwargs):
                return self.tokenizer.batch_decode(*args, **kwargs)
                
        processor = SimpleProcessor(tokenizer, image_processor)
    
    # Load model with specific class
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✓ Successfully loaded model")
    
    # Example image from Hugging Face dataset
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"
    image = Image.open(requests.get(url, stream=True).raw)

    # Prompt
    prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"

    # Prepare inputs
    inputs = processor(prompt, image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    # Generate output
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    print("\n=== Model Output ===")
    print(response)

if __name__ == "__main__":
    main()
