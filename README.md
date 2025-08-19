# LoRA Pro - Personalized AI Image Generation

This project allows you to create personalized AI-generated images using LoRA (Low-Rank Adaptation) with FLUX.1-dev, and train your own face LoRA using DreamBooth.

## 🚀 Quick Start

### 1. Generate Images with Pre-trained LoRAs

```bash
# Activate virtual environment
.\venv\Scripts\activate

# Run the main app
python app.py
```

This will:
- Download FLUX.1-dev model
- Load a Ghibli-style LoRA
- Generate 5 different scenes
- Save images as `generated_image_1.png` through `generated_image_5.png`

### 2. Train Your Own Personal LoRA

```bash
# Run the training setup script
python train_personal_lora.py
```

## 📁 Project Structure

```
lora-pro/
├── app.py                    # Main image generation script
├── train_personal_lora.py    # Personal LoRA training setup
├── dataset/                  # Your reference images (create this)
│   └── person1/             # Add 5-10 photos here
├── LoRAs/                   # Downloaded and trained LoRAs
├── venv/                    # Virtual environment
└── README.md               # This file
```

## 🎭 Training Your Own Face LoRA

### Prerequisites

- **GPU**: At least 8GB VRAM (16GB+ recommended)
- **Images**: 5-10 high-quality photos of the person
- **Time**: Training takes 1-2 hours

### Step 1: Prepare Your Dataset

1. Create the dataset folder:
   ```bash
   mkdir -p dataset/person1
   ```

2. Add 5-10 reference images to `dataset/person1/`:
   - Use different angles (front, side, 3/4 view)
   - Include various expressions
   - Ensure good lighting
   - Avoid group photos
   - Supported formats: JPG, PNG, WebP

### Step 2: Run Training Setup

```bash
python train_personal_lora.py
```

This script will:
- ✅ Check your environment
- ✅ Verify your dataset
- ✅ Download training scripts
- ✅ Generate optimal training commands
- ✅ Create usage examples

### Step 3: Start Training

The script creates `train_command.txt` with the exact command to run:

```bash
# Copy and paste the command from train_command.txt
accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path "black-forest-labs/FLUX.1-dev" \
  --instance_data_dir "./dataset/person1" \
  --output_dir "./LoRAs/person1-face" \
  --instance_prompt "<person1> in a photo" \
  --resolution 1024 \
  --train_batch_size 1 \
  --learning_rate 1e-4 \
  --max_train_steps 800
```

### Step 4: Use Your Trained LoRA

After training completes, use `usage_example.py` to generate images with your face:

```bash
python usage_example.py
```

## 🔧 Memory Optimization

The scripts automatically optimize for your GPU memory:

- **High VRAM (16GB+)**: 1024×1024 resolution, batch size 1
- **Medium VRAM (8-12GB)**: 512×512 resolution, gradient accumulation
- **Low VRAM (<8GB)**: 512×512 resolution, more aggressive optimization

## 📊 Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 1e-4 | Standard for LoRA training |
| Steps | 800 | Good balance of quality/time |
| Resolution | 1024/512 | Based on available VRAM |
| Batch Size | 1 | Memory-efficient training |

## 🎨 Prompt Engineering

### Basic Format
```
<person_name> in [scene], [style], [lighting]
```

### Examples
- `<alice> in a cozy cafe, Ghibli style, beautiful lighting`
- `<bob> in a futuristic city, cyberpunk style, neon lights`
- `<charlie> as a warrior, medieval battle, dramatic lighting`

### Tips
- Always use `<person_name>` as the first token
- Be specific about the scene and style
- Include lighting and atmosphere details
- Use descriptive adjectives

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce resolution to 512×512
   - Enable gradient accumulation
   - Use memory optimization techniques

2. **LoRA Loading Error**
   - Install PEFT: `pip install peft`
   - Check LoRA file path
   - Verify LoRA compatibility

3. **Poor Training Results**
   - Use more diverse reference images
   - Increase training steps to 1000-1200
   - Adjust learning rate to 5e-5

4. **Model Access Denied**
   - Request access to FLUX.1-dev on Hugging Face
   - Use alternative base models temporarily

### Getting Help

- Check the generated log files
- Verify your dataset structure
- Ensure all dependencies are installed
- Check GPU memory usage

## 📚 Advanced Usage

### Combining Multiple LoRAs

```python
# Load personal LoRA
pipe.load_lora_weights("./LoRAs/person1-face", adapter_name="personal")

# Load style LoRA
pipe.load_lora_weights("./LoRAs/ghibli-style", adapter_name="style")

# Use both in generation
image = pipe(
    prompt="<person1> in a magical forest, Ghibli style",
    height=512, width=512
).images[0]
```

### Custom Training Parameters

Edit `training_config.txt` to adjust:
- Learning rate
- Training steps
- Resolution
- Batch size
- Scheduler type

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Share your trained LoRAs
- Contribute code

## 📄 License

This project is for educational and personal use. Please respect the licenses of the base models and LoRAs you use.

## 🙏 Acknowledgments

- **FLUX.1-dev** by Black Forest Labs
- **Diffusers** library by Hugging Face
- **DreamBooth** research paper
- **LoRA** technique by Microsoft Research

---

**Happy LoRA Training! 🎨✨**
