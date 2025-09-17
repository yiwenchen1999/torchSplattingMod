#!/usr/bin/env python3
import argparse
import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
import json

def load_clip_model(model_name: str = "ViT-L/14", device="cuda"):
    """Load CLIP model and return both model and preprocess function"""
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

def encode_image_clip_standard(model, preprocess, img_pil: Image.Image, device="cuda"):
    """Encode image using standard CLIP encode_image method"""
    with torch.no_grad():
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        image_features = model.encode_image(img_tensor)
        image_features = torch.nn.functional.normalize(image_features, dim=-1)
    return image_features

def encode_image_clip_manual(model, preprocess, img_pil: Image.Image, device="cuda", dtype=torch.float32):
    """Encode image using manual method (from our encode_with_clip.py)"""
    # Preprocess image using CLIP's preprocessing
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device, dtype=dtype)  # (1, 3, 224, 224)
    
    # Get features from the visual encoder
    with torch.no_grad():
        # Get the visual encoder directly
        visual_encoder = model.visual
        
        # Check conv1 layer specifically
        conv1_weight_dtype = visual_encoder.conv1.weight.dtype
        
        # Forward through the visual encoder to get intermediate features
        if img_tensor.dtype != conv1_weight_dtype:
            img_tensor = img_tensor.to(dtype=conv1_weight_dtype)
        x = visual_encoder.conv1(img_tensor)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        
        # Add class token
        x = torch.cat([visual_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        
        # Add positional embedding
        x = x + visual_encoder.positional_embedding.to(x.dtype)
        
        # Layer norm pre
        x = visual_encoder.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        
        # Transformer layers
        x = visual_encoder.transformer(x)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # Get the last layer features (before final layer norm and projection)
        last_layer_features = x  # shape = [batch_size, num_patches + 1, hidden_dim]
        
        # Extract global feature (class token)
        global_feature = last_layer_features[:, 0, :]  # shape = [batch_size, hidden_dim]

        print(f"Global feature shape before ln_post: {global_feature.shape}")
        
        # Apply final layer norm and projection (like standard CLIP)
        global_feature = visual_encoder.ln_post(global_feature)

        print(f"Global feature shape after ln_post: {global_feature.shape}")
        
        # Apply projection if it exists
        if hasattr(visual_encoder, 'proj'):
            global_feature = global_feature @ visual_encoder.proj

        print(f"Global feature shape after proj: {global_feature.shape}")
        
        # Normalize
        global_feature = torch.nn.functional.normalize(global_feature, dim=-1)

        print(f"Global feature shape after normalize: {global_feature.shape}")
    
    return global_feature

def test_global_feature_equivalence(model_name: str = "ViT-L/14", device="cuda", num_tests: int = 5):
    """Test if global features match standard CLIP encoded features"""
    print(f"Testing CLIP Global Feature Equivalence")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Number of tests: {num_tests}")
    print("="*60)
    
    # Load model
    model, preprocess = load_clip_model(model_name, device=device)
    
    # Create test images
    test_images = []
    for i in range(num_tests):
        # Create different colored images
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        color = colors[i % len(colors)]
        img = Image.new('RGB', (224, 224), color=color)
        test_images.append(img)
    
    results = []
    
    for i, img in enumerate(test_images):
        print(f"\nTest {i+1}/{num_tests}:")
        print(f"Image color: {img.getpixel((112, 112))}")
        
        # Method 1: Standard CLIP encoding
        standard_features = encode_image_clip_standard(model, preprocess, img, device)
        
        # Method 2: Manual encoding (our method)
        manual_features = encode_image_clip_manual(model, preprocess, img, device)
        
        # Compare features
        diff = torch.abs(standard_features - manual_features)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(standard_features, manual_features, dim=-1).item()
        
        print(f"  Standard features shape: {standard_features.shape}")
        print(f"  Manual features shape: {manual_features.shape}")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        print(f"  Cosine similarity: {cos_sim:.8f}")
        
        # Check if they're essentially the same
        is_same = max_diff < 1e-5 and cos_sim > 0.9999
        print(f"  Are they the same? {is_same}")
        
        results.append({
            "test_id": i+1,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "cosine_similarity": cos_sim,
            "is_same": is_same
        })
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    all_same = all(r["is_same"] for r in results)
    avg_max_diff = np.mean([r["max_diff"] for r in results])
    avg_mean_diff = np.mean([r["mean_diff"] for r in results])
    avg_cos_sim = np.mean([r["cosine_similarity"] for r in results])
    
    print(f"All tests passed: {all_same}")
    print(f"Average max difference: {avg_max_diff:.8f}")
    print(f"Average mean difference: {avg_mean_diff:.8f}")
    print(f"Average cosine similarity: {avg_cos_sim:.8f}")
    
    if all_same:
        print("✅ SUCCESS: Global features match standard CLIP encoded features!")
    else:
        print("❌ FAILURE: Global features do not match standard CLIP encoded features!")
    
    return results, all_same

def test_saved_global_features(features_dir: str, device="cuda", num_samples: int = 3):
    """Test if saved global features match standard CLIP encoding"""
    print(f"\n{'='*60}")
    print("TESTING SAVED GLOBAL FEATURES")
    print(f"{'='*60}")
    
    # Find feature files
    feature_files = list(Path(features_dir).glob("*.npy"))
    global_files = list(Path(features_dir).glob("*_global.npy"))
    
    if not global_files:
        print("No global feature files found!")
        return
    
    print(f"Found {len(global_files)} global feature files")
    
    # Load CLIP model
    model, preprocess = load_clip_model("ViT-L/14", device=device)
    
    # Test a few samples
    test_files = global_files[:num_samples]
    
    for global_file in test_files:
        print(f"\nTesting: {global_file.name}")
        
        # Find corresponding image file
        base_name = global_file.stem.replace("_global", "")
        image_file = global_file.parent / f"{base_name}.png"
        
        if not image_file.exists():
            print(f"  Image file not found: {image_file}")
            continue
        
        # Load and encode image with standard CLIP
        img = Image.open(image_file)
        standard_features = encode_image_clip_standard(model, preprocess, img, device)
        
        # Load saved global features
        saved_features = torch.from_numpy(np.load(global_file)).to(device)
        
        # Compare
        diff = torch.abs(standard_features - saved_features)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        cos_sim = torch.nn.functional.cosine_similarity(standard_features, saved_features, dim=-1).item()
        
        print(f"  Standard features shape: {standard_features.shape}")
        print(f"  Saved features shape: {saved_features.shape}")
        print(f"  Max difference: {max_diff:.8f}")
        print(f"  Mean difference: {mean_diff:.8f}")
        print(f"  Cosine similarity: {cos_sim:.8f}")
        
        is_same = max_diff < 1e-5 and cos_sim > 0.9999
        print(f"  Are they the same? {is_same}")

def main():
    parser = argparse.ArgumentParser(description="Test CLIP global feature equivalence")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_tests", type=int, default=5, help="Number of test images")
    parser.add_argument("--features_dir", help="Directory containing saved global features to test")
    parser.add_argument("--save_results", action="store_true", help="Save test results to JSON")
    args = parser.parse_args()
    
    # Test 1: Compare manual vs standard encoding
    results, all_same = test_global_feature_equivalence(
        model_name=args.model,
        device=args.device,
        num_tests=args.num_tests
    )
    
    # Test 2: Test saved global features if directory provided
    if args.features_dir:
        test_saved_global_features(args.features_dir, device=args.device)
    
    # Save results if requested
    if args.save_results:
        output_file = "clip_global_feature_test_results.json"
        test_data = {
            "model": args.model,
            "device": args.device,
            "num_tests": args.num_tests,
            "all_tests_passed": all_same,
            "results": results
        }
        with open(output_file, "w") as f:
            json.dump(test_data, f, indent=2)
        print(f"\nTest results saved to: {output_file}")

if __name__ == "__main__":
    main()
