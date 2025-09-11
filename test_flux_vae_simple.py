#!/usr/bin/env python3
"""
简单的FLUX VAE测试脚本
测试修改后的编码和解码脚本的基本功能
"""

import os
import sys
import torch
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

def test_flux_vae_loading():
    """测试FLUX VAE加载功能"""
    print("=== 测试FLUX VAE加载 ===")
    
    try:
        from data_preprocess.encode_with_vae import load_vae
        
        # 测试FLUX VAE加载
        print("加载FLUX VAE...")
        vae = load_vae(flux_repo="black-forest-labs/FLUX.1-dev", device="cpu", dtype=torch.float32)
        
        print(f"VAE配置:")
        print(f"  - scaling_factor: {vae.config.scaling_factor}")
        print(f"  - shift_factor: {vae.config.shift_factor}")
        print(f"  - 设备: {next(vae.parameters()).device}")
        print(f"  - 数据类型: {next(vae.parameters()).dtype}")
        
        return True
        
    except Exception as e:
        print(f"FLUX VAE加载失败: {e}")
        return False

def test_sd_vae_loading():
    """测试SD VAE加载功能"""
    print("\n=== 测试SD VAE加载 ===")
    
    try:
        from data_preprocess.encode_with_vae import load_vae
        
        # 测试SD VAE加载
        print("加载SD VAE...")
        vae = load_vae(sd_repo="runwayml/stable-diffusion-v1-5", device="cpu", dtype=torch.float32)
        
        print(f"VAE配置:")
        print(f"  - scaling_factor: {vae.config.scaling_factor}")
        print(f"  - shift_factor: {getattr(vae.config, 'shift_factor', 'None')}")
        print(f"  - 设备: {next(vae.parameters()).device}")
        print(f"  - 数据类型: {next(vae.parameters()).dtype}")
        
        return True
        
    except Exception as e:
        print(f"SD VAE加载失败: {e}")
        return False

def test_encoding_functions():
    """测试编码函数"""
    print("\n=== 测试编码函数 ===")
    
    try:
        from data_preprocess.encode_with_vae import load_vae, encode_image
        import torch
        
        # 创建测试图像张量
        test_image = torch.randn(1, 3, 512, 512)
        
        # 测试FLUX编码
        print("测试FLUX VAE编码...")
        vae_flux = load_vae(flux_repo="black-forest-labs/FLUX.1-dev", device="cpu", dtype=torch.float32)
        latents_flux = encode_image(vae_flux, test_image, use_flux_scaling=True)
        print(f"FLUX编码结果形状: {latents_flux.shape}")
        
        # 测试SD编码
        print("测试SD VAE编码...")
        vae_sd = load_vae(sd_repo="runwayml/stable-diffusion-v1-5", device="cpu", dtype=torch.float32)
        latents_sd = encode_image(vae_sd, test_image, use_flux_scaling=False)
        print(f"SD编码结果形状: {latents_sd.shape}")
        
        return True
        
    except Exception as e:
        print(f"编码函数测试失败: {e}")
        return False

def test_decoding_functions():
    """测试解码函数"""
    print("\n=== 测试解码函数 ===")
    
    try:
        from data_preprocess.decode_with_vae import load_vae, decode_latents
        import torch
        
        # 创建测试潜在表示
        test_latents = torch.randn(1, 4, 64, 64)
        
        # 测试FLUX解码
        print("测试FLUX VAE解码...")
        vae_flux = load_vae(flux_repo="black-forest-labs/FLUX.1-dev", device="cpu", dtype=torch.float32)
        image_flux = decode_latents(vae_flux, test_latents, use_flux_scaling=True)
        print(f"FLUX解码结果形状: {image_flux.shape}")
        
        # 测试SD解码
        print("测试SD VAE解码...")
        vae_sd = load_vae(sd_repo="runwayml/stable-diffusion-v1-5", device="cpu", dtype=torch.float32)
        image_sd = decode_latents(vae_sd, test_latents, use_flux_scaling=False)
        print(f"SD解码结果形状: {image_sd.shape}")
        
        return True
        
    except Exception as e:
        print(f"解码函数测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试修改后的FLUX VAE支持...")
    
    tests = [
        test_flux_vae_loading,
        test_sd_vae_loading,
        test_encoding_functions,
        test_decoding_functions,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试 {test.__name__} 出现异常: {e}")
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✅ 所有测试通过！FLUX VAE支持已成功添加。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
