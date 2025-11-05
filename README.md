# PokeForge: Fine-Tuning Diffusion Models with LoRA

A deep learning project exploring parameter-efficient fine-tuning of Stable Diffusion using Low-Rank Adaptation (LoRA) for domain-specific image generation.

## Overview

This project demonstrates modern transfer learning techniques by fine-tuning a pre-trained diffusion model on a specialized image dataset. Using LoRA adapters, the model achieves domain-specific generation with **95% fewer trainable parameters** compared to full fine-tuning, making it feasible to train on consumer hardware (Apple M3 with 16GB RAM).

## Technical Stack

- **Framework:** PyTorch 2.9.0 with MPS (Metal Performance Shaders) optimization
- **Base Model:** Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5)
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation) via HuggingFace PEFT
- **Architecture:** UNet2DConditionModel with attention layer adapters
- **Dataset:** 905 curated images with text captions

## Key Features

### 1. Parameter-Efficient Fine-Tuning
- **LoRA Rank:** 4 (optimized for memory constraints)
- **Trainable Parameters:** 797K out of 860M total (0.09%)
- **Target Modules:** Attention layers (to_q, to_k, to_v, to_out.0)

### 2. Optimized Training Pipeline
- Custom PyTorch Dataset with preprocessing pipeline
- DDPM noise scheduling for diffusion training
- Gradient clipping and MPS memory management
- Automatic checkpoint saving every 5 epochs

### 3. Production-Ready Generation
- DPM-Solver++ multistep scheduler for fast inference
- Configurable guidance scale and inference steps
- Support for negative prompting
- Batch generation capabilities

## Project Structure

```
PokeForge/
├── collect_pokemon_data.py    # Data collection from PokeAPI
├── preprocess_images.py        # Image preprocessing & dataset creation
├── train_pokemon_lora.py       # LoRA fine-tuning script
├── generate_pokemon.py         # Inference script
├── test_setup.py              # Environment validation
├── requirements.txt           # Python dependencies
└── data/
    ├── raw/                   # Original images & metadata
    └── processed/             # Preprocessed images & CSV
```

## Results

### Training Performance
- **Epochs:** 20
- **Training Time:** ~12 hours on Apple M3
- **Final Loss:** 0.0494 (converged from 0.0521)
- **Batch Size:** 1 (optimized for 16GB RAM)
- **Learning Rate:** 1e-4 with AdamW optimizer

### Generation Quality
The fine-tuned model successfully:
- Generates novel designs not present in training data
- Maintains consistent art style and aesthetic
- Generalizes to unseen type combinations
- Produces high-quality 512×512 images in ~45 seconds

## Installation

### Requirements
- Python 3.8+
- CUDA GPU, Apple Silicon (MPS), or CPU
- 16GB+ RAM recommended

### Setup
```bash
# Clone repository
git clone https://github.com/yourusername/PokeForge.git
cd PokeForge

# Install dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

## Usage

### 1. Data Collection
```bash
python collect_pokemon_data.py
```

### 2. Preprocessing
```bash
python preprocess_images.py
```

### 3. Training
```bash
python train_pokemon_lora.py \
  --num_epochs 20 \
  --batch_size 1 \
  --lora_rank 4 \
  --learning_rate 1e-4
```

### 4. Generation
```bash
python generate_pokemon.py \
  --lora_path output/lora_model/checkpoint-epoch-20 \
  --prompt "fire-type dragon pokemon" \
  --num_images 4 \
  --num_inference_steps 50 \
  --guidance_scale 9.0
```

## Technical Insights

### Why LoRA?
Traditional fine-tuning of large diffusion models requires:
- 100% of model parameters to be trainable (~860M for SD 1.5)
- Significant GPU memory (24GB+ VRAM)
- Long training times

LoRA enables:
- **0.09% trainable parameters** via low-rank decomposition
- Training on consumer hardware (16GB RAM)
- Fast adaptation to new domains
- Easy merging/switching between adaptations

### Memory Optimization
For Apple Silicon (MPS) training:
- Float32 precision (MPS doesn't support mixed precision yet)
- Per-step cache clearing to prevent memory leaks
- Frozen VAE and text encoder (only UNet is trained)
- Batch size of 1 with gradient accumulation

## Skills Demonstrated

- Deep learning with PyTorch
- Diffusion models and denoising techniques
- Transfer learning and parameter-efficient fine-tuning
- GPU optimization and memory management
- Data pipeline development
- Hyperparameter tuning and training monitoring

## Future Enhancements

- [ ] Implement Streamlit/Gradio web interface
- [ ] Experiment with higher LoRA ranks (8, 16)
- [ ] Train on additional epochs (30-40)
- [ ] Add ControlNet for pose/composition control
- [ ] Implement DreamBooth for style consistency

## License

This project is for educational purposes. The Stable Diffusion model is subject to the CreativeML Open RAIL-M license.

## Acknowledgments

- HuggingFace for Diffusers and PEFT libraries
- RunwayML for Stable Diffusion v1.5
- PokeAPI for dataset metadata
