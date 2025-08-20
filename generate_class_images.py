#!/usr/bin/env python3
"""
Generate class images for DreamBooth training
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

def generate_class_images():
    """Generate class images for training"""
    
    print("🚀 Starting enhanced class image generation...")
    
    # Load the base model
    print("📥 Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("✅ Using CUDA")
    else:
        print("⚠️ CUDA not available, using CPU")
    
    # Enable memory optimization
    pipe.enable_attention_slicing()
    
    # Class prompt (what we want to generate)
    class_prompt = "man portrait, close up face"
    
    # Number of class images to generate (increased to 30)
    num_class_images = 30
    
    # Create output directory
    os.makedirs("./dataset/class-images", exist_ok=True)
    
    print(f"🎨 Generating {num_class_images} class images with prompt: '{class_prompt}'")
    
    # Generate images
    for i in range(num_class_images):
        print(f"📸 Generating image {i+1}/{num_class_images}...")
        
        # Generate image
        image = pipe(
            prompt=class_prompt,
            height=512,
            width=512,
            num_inference_steps=30,
            guidance_scale=7.5
        ).images[0]
        
        # Save image
        output_path = f"./dataset/class-images/class_image_{i+1:02d}.png"
        image.save(output_path)
        print(f"💾 Saved: {output_path}")
    
    print("✅ Enhanced class image generation complete!")
    print(f"📁 Generated {num_class_images} images in ./dataset/class-images/")

if __name__ == "__main__":
    generate_class_images()
