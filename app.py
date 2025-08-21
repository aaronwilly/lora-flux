from huggingface_hub import hf_hub_download
from diffusers import StableDiffusionPipeline, FluxPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from PIL import Image
print(torch.version.cuda)       # should print 12.1 or 11.8
print(torch.cuda.is_available())
# -----------------------------
# Step 1: Load Base Model
# -----------------------------
# Using the same model your LoRA was trained on

# black-forest-labs/FLUX.1-dev
# runwayml/stable-diffusion-v1-5
# stabilityai/stable-diffusion-2-1
# stabilityai/stable-diffusion-xl-base-1.0
# stabilityai/stable-diffusion-3-medium-diffusers

base_model = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
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
try:
    # Load the enhanced eugene-lora from checkpoint-2000 (best checkpoint)
    lora_path = "./LoRAs/eugene-face-enhanced/checkpoint-2000"  # Best checkpoint

    # Load from the enhanced LoRA directory
    pipe.load_lora_weights(lora_path, adapter_name="eugene0901_face")
    print("✅ Enhanced Eugene LoRA (checkpoint-2000) loaded successfully!")
except Exception as e:
    print(f"⚠️ Warning: Could not load Enhanced Eugene LoRA: {e}")
    print("Continuing without LoRA...")

# -----------------------------                                                           
# Step 4: Define Prompts for Multiple Scenes
# -----------------------------
prompts = [
    "A portrait of a <eugene0901_face>, mid-30s, short brown hair, light stubble, wearing a casual shirt, sitting outdoors in natural daylight, soft focus background, cinematic lighting, ultra-realistic, high detail, professional photography",
    "Ultra-realistic photo of a <eugene0901_face> in his 30s, dark hair, short beard, wearing a gray sweater, standing on a city street during golden hour, natural sunlight illuminating his face, shallow depth of field, sharp focus, 8k, professional photography",
    "A front-facing portrait of a <eugene0901_face>, early 30s, short brown hair, neatly trimmed beard, wearing a casual jacket, looking directly at the camera with natural expression, soft sunlight illuminating his face, subtle background bokeh, hyper-realistic, detailed skin texture, professional photography"
]

# Comprehensive negative prompt to avoid common issues
negative_prompt = """blurry, lowres, bad anatomy, extra limbs, mutated hands, disfigured, deformed, cropped, out of frame, watermark, text, logo, jpeg artifacts, unrealistic proportions, bad eyes, bad hands, two faces, oversaturated, cartoonish, poor lighting"""

# -----------------------------
# Step 5: Generate Images
# -----------------------------
generated_images = []
for prompt in prompts:

    # Lower resolution to save VRAM (512x512 instead of 1024x1024)
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        height=512, 
        width=512, 
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]
    generated_images.append(image)

# -----------------------------
# Step 6: Save Images
# -----------------------------
for idx, img in enumerate(generated_images):
    img.save(f"generated_image_{idx+1}.png")

print("✅ Generation complete! Images saved as generated_image_1.png ... generated_image_N.png")
