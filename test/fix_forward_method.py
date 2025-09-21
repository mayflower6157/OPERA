import re

# Read the current file
with open('/home/ubuntu/miniconda/envs/opera/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py', 'r') as f:
    content = f.read()

# Find and replace the specific line in the forward method
old_line = "            special_image_mask = self.get_placeholder_mask(\n                input_ids, inputs_embeds=inputs_embeds, image_features=image_features\n            )"

new_lines = """            result = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            if isinstance(result, tuple):
                # Handle the case where we get updated embeddings and input_ids
                special_image_mask, inputs_embeds, input_ids = result
            else:
                special_image_mask = result"""

# Replace the line
new_content = content.replace(old_line, new_lines)

# Write the updated file
with open('/home/ubuntu/miniconda/envs/opera/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py', 'w') as f:
    f.write(new_content)

print("Successfully updated forward method")
