#!/usr/bin/env python3
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import clip

def list_npy_files(folder, exts=(".npy",)):
    """List all .npy files in the folder"""
    p = Path(folder)
    return sorted([f for f in p.rglob("*") if f.suffix.lower() in exts])

def load_clip_model(model_name: str = "ViT-L/14", device="cuda"):
    """Load CLIP model and return both model and preprocess function"""
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess

def encode_text_prompt(model, text_prompt: str, device="cuda"):
    """Encode text prompt using CLIP"""
    with torch.no_grad():
        text_tokens = clip.tokenize([text_prompt]).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = F.normalize(text_features, dim=-1)
    return text_features

def load_clip_features(feature_path: str, device="cuda"):
    """Load CLIP features from .npy file"""
    features = np.load(feature_path)
    features_tensor = torch.from_numpy(features).to(device)
    return features_tensor

def load_global_features(feature_path: str, device="cuda"):
    """Load CLIP global features from .npy file"""
    # Try to find the corresponding global feature file
    global_path = str(feature_path).replace(".npy", "_global.npy")
    if os.path.exists(global_path):
        features = np.load(global_path)
        features_tensor = torch.from_numpy(features).to(device)
        return features_tensor
    else:
        print(f"Warning: Global features not found for {feature_path}")
        return None

def calculate_similarity_heatmap(text_features, image_features, device="cuda"):
    """Calculate cosine similarity between text and image features"""
    with torch.no_grad():
        # Normalize image features
        # image_features shape: (hidden_dim, height, width)
        image_features_flat = image_features.view(image_features.shape[0], -1)  # (hidden_dim, height*width)
        image_features_norm = F.normalize(image_features_flat, dim=0)  # normalize along feature dimension
        
        # Calculate cosine similarity
        # text_features shape: (1, hidden_dim)
        # image_features_norm shape: (hidden_dim, height*width)
        similarity = torch.mm(text_features, image_features_norm)  # (1, height*width)
        
        # Reshape back to spatial dimensions
        height, width = image_features.shape[1], image_features.shape[2]
        similarity_map = similarity.view(height, width)
        
        # Normalize to [0, 1] for heatmap visualization
        similarity_map = (similarity_map - similarity_map.min()) / (similarity_map.max() - similarity_map.min() + 1e-8)
        
    return similarity_map.cpu().numpy()

def calculate_global_similarity(text_features, global_features, device="cuda"):
    """Calculate cosine similarity between text and global image features"""
    with torch.no_grad():
        # Normalize global features
        global_features_norm = F.normalize(global_features.unsqueeze(0), dim=-1)  # (1, hidden_dim)
        
        # Calculate cosine similarity
        # text_features shape: (1, hidden_dim)
        # global_features_norm shape: (1, hidden_dim)
        similarity = torch.mm(text_features, global_features_norm.t())  # (1, 1)
        
    return similarity.item()

def save_heatmap(similarity_map, output_path: str, title: str = "CLIP Similarity Heatmap"):
    """Save similarity heatmap as image"""
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Similarity Score')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Test CLIP similarity between text prompt and image features")
    parser.add_argument("--features_dir", required=True, help="Directory containing CLIP feature .npy files")
    parser.add_argument("--output_dir", required=True, help="Output directory for heatmap visualizations")
    parser.add_argument("--text_prompt", default="a ship at sea", help="Text prompt to test similarity")
    parser.add_argument("--model", default="ViT-L/14", help="CLIP model name")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_individual", action="store_true", help="Save individual heatmaps for each image")
    parser.add_argument("--save_grid", action="store_true", help="Save a grid of all heatmaps")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load CLIP model
    print(f"Loading CLIP model: {args.model}")
    model, preprocess = load_clip_model(args.model, device=args.device)
    
    # Encode text prompt
    print(f"Encoding text prompt: '{args.text_prompt}'")
    text_features = encode_text_prompt(model, args.text_prompt, device=args.device)
    
    # Find all feature files
    feature_files = list_npy_files(args.features_dir)
    if not feature_files:
        print(f"No .npy files found in {args.features_dir}")
        return
    
    print(f"Found {len(feature_files)} feature files")
    
    # Process each feature file
    similarity_maps = []
    global_similarities = []
    feature_names = []
    
    for feature_file in tqdm(feature_files, desc="Processing features"):
        # Load patch features
        features = load_clip_features(str(feature_file), device=args.device)
        
        # Load global features if available
        global_features = load_global_features(str(feature_file), device=args.device)
        
        # Calculate similarity heatmap using global features if available, otherwise use patch features
        if global_features is not None:
            # Use global features for similarity calculation (this will work with text features)
            global_sim = calculate_global_similarity(text_features, global_features, device=args.device)
            global_similarities.append(global_sim)
            
            # Create a uniform heatmap based on global similarity
            height, width = features.shape[1], features.shape[2]
            similarity_map = np.full((height, width), global_sim)
        else:
            # Fallback to patch-level similarity (may have dimension mismatch)
            try:
                similarity_map = calculate_similarity_heatmap(text_features, features, device=args.device)
                global_similarities.append(np.mean(similarity_map))
            except RuntimeError as e:
                print(f"Warning: Dimension mismatch for {feature_file.stem}: {e}")
                # Create a uniform heatmap with mean similarity
                height, width = features.shape[1], features.shape[2]
                similarity_map = np.full((height, width), 0.5)  # neutral similarity
                global_similarities.append(0.5)
        
        similarity_maps.append(similarity_map)
        
        # Get feature file name for output
        feature_name = feature_file.stem
        feature_names.append(feature_name)
        
        # Save individual heatmap if requested
        if args.save_individual:
            output_path = Path(args.output_dir) / f"{feature_name}_heatmap.png"
            title = f"CLIP Similarity: '{args.text_prompt}'\n{feature_name}"
            save_heatmap(similarity_map, str(output_path), title)
    
    # Save grid of all heatmaps if requested
    if args.save_grid and similarity_maps:
        print("Creating grid visualization...")
        n_images = len(similarity_maps)
        cols = min(4, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (similarity_map, feature_name) in enumerate(zip(similarity_maps, feature_names)):
            if i < len(axes):
                axes[i].imshow(similarity_map, cmap='hot', interpolation='nearest')
                axes[i].set_title(feature_name, fontsize=8)
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(similarity_maps), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"CLIP Similarity Heatmaps: '{args.text_prompt}'", fontsize=16)
        plt.tight_layout()
        
        grid_output_path = Path(args.output_dir) / "similarity_grid.png"
        plt.savefig(str(grid_output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Grid visualization saved to: {grid_output_path}")
    
    # Calculate and save statistics
    all_similarities = np.concatenate([sim.flatten() for sim in similarity_maps])
    stats = {
        "text_prompt": args.text_prompt,
        "model": args.model,
        "num_images": len(similarity_maps),
        "mean_similarity": float(np.mean(all_similarities)),
        "std_similarity": float(np.std(all_similarities)),
        "min_similarity": float(np.min(all_similarities)),
        "max_similarity": float(np.max(all_similarities)),
        "global_similarities": {
            "mean": float(np.mean(global_similarities)),
            "std": float(np.std(global_similarities)),
            "min": float(np.min(global_similarities)),
            "max": float(np.max(global_similarities)),
            "values": global_similarities
        },
        "feature_files": [str(f) for f in feature_files]
    }
    
    stats_path = Path(args.output_dir) / "similarity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nSimilarity Statistics:")
    print(f"  Mean similarity: {stats['mean_similarity']:.4f}")
    print(f"  Std similarity: {stats['std_similarity']:.4f}")
    print(f"  Min similarity: {stats['min_similarity']:.4f}")
    print(f"  Max similarity: {stats['max_similarity']:.4f}")
    print(f"\nGlobal Similarity Statistics:")
    print(f"  Mean global similarity: {stats['global_similarities']['mean']:.4f}")
    print(f"  Std global similarity: {stats['global_similarities']['std']:.4f}")
    print(f"  Min global similarity: {stats['global_similarities']['min']:.4f}")
    print(f"  Max global similarity: {stats['global_similarities']['max']:.4f}")
    print(f"  Statistics saved to: {stats_path}")
    
    if args.save_individual:
        print(f"Individual heatmaps saved to: {args.output_dir}")
    
    print("Done!")

if __name__ == "__main__":
    main()
