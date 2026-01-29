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
prompts= [
"A portrait of a <eugene0901_face>, mid-20s, short brown hair, light stubble, wearing a casual shirt, sitting outdoors in natural daylight, soft focus background, cinematic lighting, ultra-realistic, high detail, professional photography",
"Ultra-realistic photo of a <eugene0901_face> in 25s, dark hair, wearing a gray sweater, standing on a city street during golden hour, natural sunlight illuminating his face, shallow depth of field, sharp focus, 8k, professional photography",
"<eugene0901_face>, solo portrait of one man, close-up, upper body, looking directly at camera, realistic photo, natural sunlight, bokeh background",
"<eugene0901_face>, solo portrait of one man, early morning park, soft mist in the background, natural daylight, casual hoodie, realistic skin details, cinematic tones, DSLR 50mm lens, ultra-realistic, professional photography",
"<eugene0901_face>, close-up portrait, sitting in a cozy coffee shop by the window, warm indoor lighting, shallow depth of field, soft bokeh of lights, relaxed expression, photorealistic, cinematic photography",
"<eugene0901_face>, professional headshot, wearing a business suit, clean haircut, neutral gray background, studio lighting, ultra-sharp details, high-resolution DSLR photo, professional portrait photography",
"<eugene0901_face>, casual portrait, standing on a beach at sunset, golden light reflecting on skin, wind in hair, natural smile, cinematic horizon background, ultra-realistic, vibrant details",
"<eugene0901_face>, solo portrait of one man, standing near neon city lights at night, colorful reflections on face, moody atmosphere, cinematic cyberpunk style, sharp facial features, detailed skin texture, professional low-light photography",
"<eugene0901_face>, outdoor hiking portrait, wearing casual jacket, forest background, sunlight filtering through trees, natural rugged look, realistic skin pores, cinematic lens flare, ultra-photorealistic",
"<eugene0901_face>, close-up, in a modern office environment, soft natural daylight from large windows, casual smart-casual outfit, professional vibe, subtle bokeh, high detail, DSLR photo",
"<eugene0901_face>, street-style portrait, casual outfit, leaning against a brick wall, urban background, cinematic lighting, realistic textures, solo composition, sharp focus, ultra-realistic",
"<eugene0901_face>, portrait in snowy outdoor environment, wearing winter coat and scarf, snow falling in background, breath visible in cold air, cinematic tones, realistic lighting, professional photography",
"<eugene0901_face>, cinematic travel photo, standing near a train station with luggage, overcast daylight, casual outfit, solo subject, natural expression, high detail, hyper-realistic photo"
]

# Comprehensive negative prompt to avoid common issues
negative_prompt = """blurry, lowres, bad anatomy, extra limbs, mutated hands, disfigured, deformed, cropped, out of frame, watermark, text, logo, jpeg artifacts, poorly drawn face, two faces, bad eyes, bad hands, unrealistic lighting, cartoonish, oversaturated"""

# -----------------------------
# Step 5: Generate Images
# -----------------------------
generated_images = []
for prompt in prompts:

    # Lower resolution to save VRAM (512x512 instead of 1024x1024)
    image = pipe(
        prompt=prompt, 
        negative_prompt=negative_prompt,
        height=768, 
        width=768, 
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]
    generated_images.append(image)

# -----------------------------
# Step 6: Save Images
# -----------------------------
for idx, img in enumerate(generated_images):
    img.save(f"generated_image_{idx+1}.png")

print("✅ Generation complete! Images saved as generated_image_1.png ... generated_image_N.png")
