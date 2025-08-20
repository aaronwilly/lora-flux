from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
print(torch.version.cuda)       # should print 12.1 or 11.8
print(torch.cuda.is_available())
# -----------------------------
# Step 1: Load Base Model
# -----------------------------
# Using the same model your LoRA was trained on
base_model = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
)

# Enable memory optimization techniques
pipe.enable_model_cpu_offload()        # Move unused parts to CPU
pipe.enable_attention_slicing()        # Process attention in chunks
pipe.enable_sequential_cpu_offload()   # Alternative offloading method

# Enable xformers for better memory efficiency
try:
    pipe.enable_xformers_memory_efficient_attention()
    print("xformers enabled for memory efficiency")
except Exception as e:
    print(f"xformers not available: {e}")

# Note: With sequential offloading, we don't move to CUDA
# The pipeline will automatically manage GPU/CPU memory

 

# -----------------------------
# Step 2: Load Personalized LoRA (Face Identity)
# -----------------------------
# personal_lora_repo = "your-username/your-personal-lora"  # Replace with your DreamBooth/face LoRA
# personal_lora_file = "personal-face.safetensors"
# hf_hub_download(repo_id=personal_lora_repo, filename=personal_lora_file, local_dir="./LoRAs")
# pipe.load_lora_weights(f"./LoRAs/{personal_lora_file}", adapter_name="lora")

# -----------------------------
# Step 3: Load Style LoRA (Optional)
# -----------------------------
try:
    # Load the locally trained eugene-lora from the new training
    lora_path = "./LoRAs/eugene-face-new"  # New LoRA directory

    # Load from the new LoRA directory
    pipe.load_lora_weights(lora_path, adapter_name="eugene0901_face")
    print("✅ Eugene LoRA loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load Eugene LoRA: {e}")
    print("Continuing without LoRA...")

# -----------------------------                                                           
# Step 4: Define Prompts for Multiple Scenes
# -----------------------------
prompts = [
    "<eugene0901_face> man portrait, close-up face, smiling, short black hair, brown eyes, soft studio lighting",
    "<eugene0901_face> realistic face, looking at camera, medium light, casual clothes, natural background",
    "<eugene0901_face> cinematic portrait, sharp facial features, short hair, warm sunlight, detailed skin texture"
]

# -----------------------------
# Step 5: Generate Images
# -----------------------------
generated_images = []
for prompt in prompts:
    # Lower resolution to save VRAM (512x512 instead of 1024x1024)
    image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=30).images[0]
    generated_images.append(image)

# -----------------------------
# Step 6: Save Images
# -----------------------------
for idx, img in enumerate(generated_images):
    img.save(f"generated_image_{idx+1}.png")

print("✅ Generation complete! Images saved as generated_image_1.png ... generated_image_N.png")
