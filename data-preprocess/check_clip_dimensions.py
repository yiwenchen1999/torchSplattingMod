#!/usr/bin/env python3
import argparse
import torch
import clip
from PIL import Image
import numpy as np

def check_clip_model_dimensions(model_name: str, device="cuda"):
    """Check dimensions of a CLIP model"""
    print(f"\n{'='*60}")
    print(f"Checking CLIP Model: {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load model
        model, preprocess = clip.load(model_name, device=device)
        model.eval()
        
        # Create a dummy image and text for testing
        dummy_image = Image.new('RGB', (224, 224), color='red')
        dummy_text = "a test prompt"
        
        print(f"Model loaded successfully on device: {device}")
        
        # Check text encoder dimensions
        print(f"\n--- Text Encoder ---")
        text_tokens = clip.tokenize([dummy_text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_tokens)
            print(f"Text features shape: {text_features.shape}")
            print(f"Text features dtype: {text_features.dtype}")
        
        # Check image encoder dimensions
        print(f"\n--- Image Encoder ---")
        img_tensor = preprocess(dummy_image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            print(f"Image features shape: {image_features.shape}")
            print(f"Image features dtype: {image_features.dtype}")
        
        # Check visual encoder internal dimensions
        print(f"\n--- Visual Encoder Internal ---")
        visual_encoder = model.visual
        
        # Check conv1 layer
        print(f"Conv1 input channels: {visual_encoder.conv1.in_channels}")
        print(f"Conv1 output channels: {visual_encoder.conv1.out_channels}")
        print(f"Conv1 kernel size: {visual_encoder.conv1.kernel_size}")
        print(f"Conv1 stride: {visual_encoder.conv1.stride}")
        print(f"Conv1 padding: {visual_encoder.conv1.padding}")
        
        # Check transformer dimensions
        print(f"Transformer width: {visual_encoder.width}")
        print(f"Transformer layers: {visual_encoder.layers}")
        print(f"Transformer heads: {visual_encoder.heads}")
        
        # Check positional embedding
        print(f"Positional embedding shape: {visual_encoder.positional_embedding.shape}")
        print(f"Class embedding shape: {visual_encoder.class_embedding.shape}")
        
        # Check output projection
        if hasattr(visual_encoder, 'proj'):
            print(f"Output projection shape: {visual_encoder.proj.weight.shape}")
        else:
            print("No output projection found")
        
        # Check final projection layer
        if hasattr(model, 'visual_projection'):
            print(f"Visual projection shape: {model.visual_projection.weight.shape}")
        else:
            print("No visual projection found")
        
        # Test patch-level feature extraction (like in our encoding script)
        print(f"\n--- Patch-level Feature Extraction ---")
        with torch.no_grad():
            # Forward through visual encoder manually
            x = visual_encoder.conv1(img_tensor)  # shape = [*, width, grid, grid]
            print(f"After conv1: {x.shape}")
            
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            print(f"After reshape and permute: {x.shape}")
            
            # Add class token
            x = torch.cat([visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            print(f"After adding class token: {x.shape}")
            
            # Add positional embedding
            x = x + visual_encoder.positional_embedding.to(x.dtype)
            print(f"After adding positional embedding: {x.shape}")
            
            # Layer norm pre
            x = visual_encoder.ln_pre(x)
            print(f"After layer norm pre: {x.shape}")
            
            x = x.permute(1, 0, 2)  # NLD -> LND
            print(f"After permute to LND: {x.shape}")
            
            # Transformer layers
            x = visual_encoder.transformer(x)
            print(f"After transformer: {x.shape}")
            
            x = x.permute(1, 0, 2)  # LND -> NLD
            print(f"After permute to NLD: {x.shape}")
            
            # Get patch features (remove class token)
            patch_features = x[:, 1:, :]  # shape = [batch_size, num_patches, hidden_dim]
            print(f"Patch features (without class token): {patch_features.shape}")
            
            # Reshape to spatial format
            grid_size = int(np.sqrt(patch_features.shape[1]))
            spatial_features = patch_features.reshape(patch_features.shape[0], grid_size, grid_size, patch_features.shape[2])
            spatial_features = spatial_features.permute(0, 3, 1, 2)  # [batch, hidden_dim, grid, grid]
            print(f"Spatial features: {spatial_features.shape}")
        
        # Summary
        print(f"\n--- Summary ---")
        print(f"Text feature dimension: {text_features.shape[1]}")
        print(f"Image feature dimension: {image_features.shape[1]}")
        print(f"Visual encoder width (hidden dim): {visual_encoder.width}")
        print(f"Patch grid size: {grid_size}x{grid_size}")
        print(f"Number of patches: {grid_size * grid_size}")
        print(f"Patch-level feature dimension: {visual_encoder.width}")
        
        return {
            'model_name': model_name,
            'text_dim': text_features.shape[1],
            'image_dim': image_features.shape[1],
            'visual_width': visual_encoder.width,
            'grid_size': grid_size,
            'num_patches': grid_size * grid_size,
            'patch_dim': visual_encoder.width
        }
        
    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Check CLIP model dimensions")
    parser.add_argument("--models", nargs="+", default=[
        "ViT-B/32",
        "ViT-B/16", 
        "ViT-L/14",
        "ViT-L/14@336px"
    ], help="CLIP model names to check")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_results", action="store_true", help="Save results to JSON file")
    args = parser.parse_args()
    
    print("CLIP Model Dimension Checker")
    print("="*60)
    
    results = []
    
    for model_name in args.models:
        result = check_clip_model_dimensions(model_name, device=args.device)
        if result:
            results.append(result)
    
    # Print comparison table
    if results:
        print(f"\n{'='*80}")
        print("DIMENSION COMPARISON TABLE")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Text Dim':<10} {'Image Dim':<11} {'Visual Width':<13} {'Grid Size':<10} {'Patches':<8}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['model_name']:<20} {result['text_dim']:<10} {result['image_dim']:<11} {result['visual_width']:<13} {result['grid_size']:<10} {result['num_patches']:<8}")
    
    # Save results if requested
    if args.save_results and results:
        import json
        output_file = "clip_model_dimensions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
