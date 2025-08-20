from huggingface_hub import hf_hub_download
from diffusers import FluxPipeline
import torch
from PIL import Image
print(torch.version.cuda)       # should print 12.1 or 11.8
print(torch.cuda.is_available())
# -----------------------------
# Step 1: Load Base Model
# -----------------------------
# Using a reliable, publicly available model
base_model = "black-forest-labs/FLUX.1-dev"
pipe = FluxPipeline.from_pretrained(
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
    # Load the locally trained eugene-lora
    lora_file = "checkpoint-1000/pytorch_lora_weights.safetensors"  # Your trained LoRA file

    # Load from local LoRAs folder
    pipe.load_lora_weights(f"./LoRAs/eugene-face/{lora_file}", adapter_name="eugene")
    print("✅ Eugene LoRA loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load Eugene LoRA: {e}")
    print("Continuing without LoRA...")

# -----------------------------                                                           
# Step 4: Define Prompts for Multiple Scenes
# -----------------------------
prompts = [
    "Close-up portrait of <eugene>, in a cozy cafe, Ghibli style, beautiful lighting",
    "Full-body portrait of <eugene>, futuristic city, cyberpunk style, neon lights",
    # "Action pose of <eugene> as a warrior, medieval battle, oil painting style, dramatic lighting",
    "Casual sitting pose of <eugene>, park scenery, photorealistic style, natural lighting",
    # "Fantasy scene with <eugene>, dragon flying in background, cinematic lighting, epic composition"
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
