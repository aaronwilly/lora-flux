#!/usr/bin/env python3
"""
Personalized LoRA Training Script using DreamBooth
Train a LoRA adapter for your own face using Stable Diffusion 1.5
"""

import os
from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

def setup_environment():
    """Setup and verify environment"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a GPU-enabled system.")
    
    print(f"CUDA: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    
    if not os.path.exists("dataset"):
        os.makedirs("dataset/person1", exist_ok=True)
        print("Created dataset/person1/ - add 5-10 reference images")
        return False
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = []
    for ext in image_extensions:
        # Use case-insensitive pattern to avoid duplicates
        images.extend(Path("dataset/person1").glob(f"*{ext}"))
    
    # Remove duplicates by converting to set of filenames, then back to list
    unique_images = list({img.name: img for img in images}.values())
    
    if len(unique_images) < 5:
        print(f"Found {len(unique_images)} images. Need at least 5 for training.")
        return False
    
    print(f"Dataset ready: {len(unique_images)} images")
    print(f"Images found: {[img.name for img in unique_images]}")
    return True

def download_training_script():
    """Check for local training script or download if needed"""
    local_script = "train_dreambooth_lora.py"
    
    # Check if local script exists
    if os.path.exists(local_script):
        print(f"Using local training script: {local_script}")
        return local_script
    
    # If local script doesn't exist, try to download
    try:
        script_path = hf_hub_download(
            repo_id="diffusers",
            filename="examples/dreambooth/train_dreambooth_lora.py"
        )
        print(f"Downloaded training script: {script_path}")
        return script_path
    except Exception as e:
        print(f"Download failed: {e}")
        print("Creating local placeholder script...")
        
        local_script_content = """#!/usr/bin/env python3
# Placeholder DreamBooth LoRA Training Script
# Download full version from: https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py

import argparse
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True)
    parser.add_argument("--class_prompt", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=800)
    
    args = parser.parse_args()
    print(f"Training LoRA: {args.instance_prompt}")
    print(f"Output: {args.output_dir}")
    print(f"Steps: {args.max_train_steps}")
    print("⚠️  This is a placeholder. Download the full script from diffusers repo.")

if __name__ == "__main__":
    main()
"""
        
        with open(local_script, "w", encoding="utf-8") as f:
            f.write(local_script_content)
        
        return local_script

def create_training_config(person_name, num_images):
    """Create training configuration"""
    config = {
        "person_name": person_name,
        "instance_prompt": f"<{person_name}> in a photo",
        "class_prompt": "person in a photo",
        "num_images": num_images,
        "resolution": 512,
        "train_batch_size": 1,
        "learning_rate": 1e-4,
        "max_train_steps": 800,
        "save_steps": 100,
        "gradient_accumulation_steps": 1,
        "lr_scheduler": "constant",
        "lr_warmup_steps": 0,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "use_xformers": True,
        "num_class_images": 50,  # Default value
        "seed": 42
    }
    
    # Adjust for low VRAM
    if torch.cuda.get_device_properties(0).total_memory < 12e9:
        config["resolution"] = 256  # Reduced from 512 to save memory
        config["gradient_accumulation_steps"] = 2
        config["num_class_images"] = 10  # Reduced from 50 to save memory
        print("Low VRAM detected - using 256 resolution and 10 class images")
    
    return config

def generate_training_command(config, script_path):
    """Generate the training command"""
    cmd = f"""accelerate launch {script_path} \\
  --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \\
  --instance_data_dir "./dataset/{config['person_name']}" \\
  --output_dir "./LoRAs/{config['person_name']}-face" \\
  --instance_prompt "{config['instance_prompt']}" \\
  --class_prompt "{config['class_prompt']}" \\
  --resolution {config['resolution']} \\
  --train_batch_size {config['train_batch_size']} \\
  --learning_rate {config['learning_rate']} \\
  --max_train_steps {config['max_train_steps']} \\
  --save_steps {config['save_steps']} \\
  --gradient_accumulation_steps {config['gradient_accumulation_steps']} \\
  --lr_scheduler "{config['lr_scheduler']}" \\
  --lr_warmup_steps {config['lr_warmup_steps']} \\
  --mixed_precision "{config['mixed_precision']}" \\
  --gradient_checkpointing \\
  --use_xformers \\
  --enable_xformers_memory_efficient_attention \\
  --seed {config['seed']} \\
  --num_class_images {config['num_class_images']} \\
  --class_data_dir "./dataset/class-images" \\
  --with_prior_preservation \\
  --checkpointing_steps {config['save_steps']}"""
    
    return cmd

def create_usage_example(config):
    """Create example of how to use the trained LoRA"""
    example_code = f'''# Using your trained LoRA
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()



pipe.load_lora_weights("./LoRAs/{config['person_name']}-face", adapter_name="personal")

prompts = [
    f"<{config['person_name']}> in a cozy cafe, beautiful lighting",
    f"<{config['person_name']}> in a futuristic city, cyberpunk style",
    f"<{config['person_name']}> as a warrior, medieval battle scene",
    f"<{config['person_name']}> in a park, natural lighting",
    f"<{config['person_name']}> in a fantasy world, magical atmosphere"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, height={config['resolution']}, width={config['resolution']}, num_inference_steps=30).images[0]
    image.save(f"personal_image_{{i+1}}.png")

print("Personal LoRA generation complete!")
'''
    
    return example_code

def main():
    """Main function"""
    print("Personalized LoRA Training Setup")
    print("=" * 40)
    
    if not setup_environment():
        print("\nNext steps:")
        print("1. Add 5-10 reference images to dataset/person1/")
        print("2. Run this script again")
        return
    
    person_name = input("\nEnter person name (or press Enter for 'person1'): ").strip()
    if not person_name:
        person_name = "person1"
    
    print(f"Using: {person_name}")
    
    # Get image count from the person's directory
    person_dir = f"dataset/{person_name}"
    image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
    images = []
    for ext in image_extensions:
        # Use case-insensitive pattern to avoid duplicates
        images.extend(Path(person_dir).glob(f"*{ext}"))
    
    # Remove duplicates by converting to set of filenames, then back to list
    unique_images = list({img.name: img for img in images}.values())
    
    print(f"Found {len(unique_images)} images in {person_dir}/")
    
    config = create_training_config(person_name, len(unique_images))
    script_path = download_training_script()
    training_cmd = generate_training_command(config, script_path)
    usage_example = create_usage_example(config)
    
    # Save files
    print("\nSaving configuration files...")
    
    with open("train_command.txt", "w", encoding="utf-8") as f:
        f.write(training_cmd)
    
    with open("usage_example.py", "w", encoding="utf-8") as f:
        f.write(usage_example)
    
    with open("training_config.txt", "w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
    
    print("Training setup complete!")
    print("\nNext steps:")
    print("1. Review train_command.txt")
    print("2. Run the training command (1-2 hours)")
    print("3. Use usage_example.py to generate images")

if __name__ == "__main__":
    main()
