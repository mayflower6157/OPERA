from transformers import pipeline

# Use the multimodal pipeline
pipe = pipeline(
    "image-text-to-text", 
    model="llava-hf/llava-1.5-7b-hf", 
    device_map="auto"   # automatically picks MPS/GPU/CPU
)

# Inputs: image + text in chat format
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png"},
            {"type": "text", "text": "Describe the image briefly."}
        ],
    }
]

# Run inference
result = pipe(messages)

print(result[0]["generated_text"])