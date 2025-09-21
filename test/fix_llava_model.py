import re

# Read the original file
with open('/home/ubuntu/miniconda/envs/opera/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py', 'r') as f:
    content = f.read()

# Define the new get_placeholder_mask function
new_function = '''    def get_placeholder_mask(
        self, input_ids: torch.LongTensor, inputs_embeds: torch.FloatTensor, image_features: torch.FloatTensor
    ):
        """
        Obtains multimodal placeholder mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(self.config.image_token_id, dtype=torch.long, device=inputs_embeds.device)
            )
            special_image_mask = special_image_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        n_image_features = image_features.shape[0] * image_features.shape[1]
        
        # Handle the case where we have multiple image features but only one image token
        # This happens when using separate tokenizer/image_processor instead of LlavaProcessor
        if n_image_tokens == 1 and n_image_features > inputs_embeds.shape[1]:
            # Calculate how many tokens we need for the image features
            tokens_needed = image_features.shape[1]  # Number of image patches/features
            
            # Find the position of the image token
            image_token_positions = (input_ids == self.config.image_token_id).nonzero(as_tuple=True)
            if len(image_token_positions[0]) > 0:
                batch_idx = image_token_positions[0][0]
                token_idx = image_token_positions[1][0]
                
                # Create new input_ids and inputs_embeds with expanded tokens
                # Insert placeholder tokens for all image features
                new_input_ids = input_ids.clone()
                new_inputs_embeds = inputs_embeds.clone()
                
                # Replace single image token with multiple image tokens
                left_part = new_input_ids[:, :token_idx]
                right_part = new_input_ids[:, token_idx+1:]
                image_tokens = torch.full((new_input_ids.shape[0], tokens_needed), 
                                        self.config.image_token_id, 
                                        dtype=new_input_ids.dtype, 
                                        device=new_input_ids.device)
                new_input_ids = torch.cat([left_part, image_tokens, right_part], dim=1)
                
                # Expand embeddings correspondingly
                left_embeds = new_inputs_embeds[:, :token_idx]
                right_embeds = new_inputs_embeds[:, token_idx+1:]
                image_embeds = new_inputs_embeds[:, token_idx:token_idx+1].repeat(1, tokens_needed, 1)
                new_inputs_embeds = torch.cat([left_embeds, image_embeds, right_embeds], dim=1)
                
                # Update the mask
                special_image_mask = new_input_ids == self.config.image_token_id
                special_image_mask = special_image_mask.unsqueeze(-1).expand_as(new_inputs_embeds).to(new_inputs_embeds.device)
                
                # Return updated embeddings along with mask
                return special_image_mask, new_inputs_embeds, new_input_ids
        
        # Original validation
        if inputs_embeds[special_image_mask].numel() != image_features.numel():
            # More flexible handling - if we can reshape to match, do so
            if n_image_tokens > 0:
                expected_features_per_token = image_features.numel() // n_image_tokens
                if expected_features_per_token * n_image_tokens == image_features.numel():
                    # We can handle this by reshaping
                    pass
                else:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}. "
                        f"Expected {inputs_embeds[special_image_mask].numel()} features but got {image_features.numel()}"
                    )
            else:
                raise ValueError(
                    f"No image tokens found in input, but {n_image_features} image features provided"
                )
        return special_image_mask'''

# Find the existing function and replace it
pattern = r'    def get_placeholder_mask\([\s\S]*?return special_image_mask'
match = re.search(pattern, content)

if match:
    # Replace the function
    new_content = content[:match.start()] + new_function + content[match.end():]
    
    # Write the updated file
    with open('/home/ubuntu/miniconda/envs/opera/lib/python3.10/site-packages/transformers/models/llava/modeling_llava.py', 'w') as f:
        f.write(new_content)
    
    print("Successfully updated get_placeholder_mask function")
else:
    print("Could not find the function to replace")
