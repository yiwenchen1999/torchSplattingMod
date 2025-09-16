#!/usr/bin/env python3
"""
Simple verification script to check if our CLIP global feature extraction
matches the standard CLIP encode_image method.
"""
import torch
import clip
from PIL import Image
import numpy as np

def test_clip_equivalence():
    """Test if our manual CLIP encoding matches standard encoding"""
    print("Testing CLIP Global Feature Equivalence")
    print("="*50)
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    
    # Create a test image
    test_img = Image.new('RGB', (224, 224), color=(128, 64, 192))
    
    print(f"Device: {device}")
    print(f"Test image size: {test_img.size}")
    
    with torch.no_grad():
        # Method 1: Standard CLIP encoding
        img_tensor = preprocess(test_img).unsqueeze(0).to(device)
        standard_features = model.encode_image(img_tensor)
        standard_features = torch.nn.functional.normalize(standard_features, dim=-1)
        
        print(f"Standard features shape: {standard_features.shape}")
        print(f"Standard features dtype: {standard_features.dtype}")
        
        # Method 2: Our manual method
        visual_encoder = model.visual
        
        # Forward pass through visual encoder
        x = visual_encoder.conv1(img_tensor)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        
        # Add class token
        x = torch.cat([
            visual_encoder.class_embedding.to(x.dtype) + 
            torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
            x
        ], dim=1)
        
        # Add positional embedding
        x = x + visual_encoder.positional_embedding.to(x.dtype)
        
        # Layer norm pre
        x = visual_encoder.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Transformer layers
        x = visual_encoder.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Extract class token (global feature)
        global_feature = x[:, 0, :]  # [batch_size, hidden_dim]
        
        # Apply final layer norm and projection (same as standard CLIP)
        global_feature = visual_encoder.ln_post(global_feature)
        
        # Apply projection if exists
        if hasattr(visual_encoder, 'proj'):
            global_feature = global_feature @ visual_encoder.proj
        
        # Normalize
        manual_features = torch.nn.functional.normalize(global_feature, dim=-1)
        
        print(f"Manual features shape: {manual_features.shape}")
        print(f"Manual features dtype: {manual_features.dtype}")
        
        # Compare
        diff = torch.abs(standard_features - manual_features)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        cos_sim = torch.nn.functional.cosine_similarity(standard_features, manual_features, dim=-1).item()
        
        print(f"\nComparison Results:")
        print(f"Max difference: {max_diff:.10f}")
        print(f"Mean difference: {mean_diff:.10f}")
        print(f"Cosine similarity: {cos_sim:.10f}")
        
        # Check if they're essentially identical
        tolerance = 1e-6
        is_identical = max_diff < tolerance and cos_sim > (1 - tolerance)
        
        print(f"\nAre they identical (tolerance={tolerance})? {is_identical}")
        
        if is_identical:
            print("✅ SUCCESS: Our implementation matches standard CLIP!")
        else:
            print("❌ FAILURE: Our implementation differs from standard CLIP!")
            
        return is_identical

if __name__ == "__main__":
    test_clip_equivalence()
