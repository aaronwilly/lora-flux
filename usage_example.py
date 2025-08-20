# Using your trained LoRA
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



pipe.load_lora_weights("./LoRAs/eugene-face", adapter_name="personal")

prompts = [
    f"<eugene> in a cozy cafe, beautiful lighting",
    f"<eugene> in a futuristic city, cyberpunk style",
    f"<eugene> as a warrior, medieval battle scene",
    f"<eugene> in a park, natural lighting",
    f"<eugene> in a fantasy world, magical atmosphere"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(f"personal_image_{i+1}.png")

print("Personal LoRA generation complete!")
