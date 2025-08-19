# Using your trained LoRA
from diffusers import FluxPipeline
import torch

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16
)

# Enable memory optimizations
pipe.enable_model_cpu_offload()
pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload()



pipe.load_lora_weights("./LoRAs/person1-face", adapter_name="personal")

prompts = [
    f"<person1> in a cozy cafe, beautiful lighting",
    f"<person1> in a futuristic city, cyberpunk style",
    f"<person1> as a warrior, medieval battle scene",
    f"<person1> in a park, natural lighting",
    f"<person1> in a fantasy world, magical atmosphere"
]

for i, prompt in enumerate(prompts):
    image = pipe(prompt=prompt, height=512, width=512, num_inference_steps=30).images[0]
    image.save(f"personal_image_{i+1}.png")

print("Personal LoRA generation complete!")
