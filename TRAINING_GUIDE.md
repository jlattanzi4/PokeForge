# Pokemon LoRA Training Guide

## Quick Start

### 1. Start Training

```bash
python train_pokemon_lora.py
```

This will train with default settings:
- 10 epochs
- Batch size: 1
- LoRA rank: 4
- Learning rate: 1e-4
- Using Stable Diffusion 1.5

### 2. Monitor Training

The script will show progress bars and loss values:
```
Epoch 1/10
████████████████████ 905/905 [15:30<00:00, loss: 0.0845]
```

### 3. Generate Pokemon

After training:
```bash
# Generate fire-type dragon
python generate_pokemon.py --prompt "fire-type dragon pokemon"

# Generate water-type turtle
python generate_pokemon.py --prompt "water-type turtle pokemon" --num_images 8

# Generate with custom LoRA
python generate_pokemon.py --lora_path output/lora_model/checkpoint-epoch-5
```

## Training Options

### Basic Training
```bash
# Quick test (1 epoch)
python train_pokemon_lora.py --num_epochs 1

# Full training (20 epochs)
python train_pokemon_lora.py --num_epochs 20

# Larger batch size (if you have memory)
python train_pokemon_lora.py --batch_size 2
```

### Advanced Options
```bash
python train_pokemon_lora.py \
  --num_epochs 15 \
  --batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 4 \
  --save_interval 3 \
  --output_dir output/my_model
```

### Memory Optimization
If you run out of memory:
```bash
# Reduce LoRA rank
python train_pokemon_lora.py --lora_rank 2

# Smaller image size
python train_pokemon_lora.py --image_size 384
```

## Generation Options

### Example Prompts
```bash
# By type
python generate_pokemon.py --prompt "fire-type pokemon"
python generate_pokemon.py --prompt "water and flying-type pokemon"
python generate_pokemon.py --prompt "psychic-type pokemon"

# By characteristics
python generate_pokemon.py --prompt "electric-type mouse pokemon"
python generate_pokemon.py --prompt "grass and poison-type seed pokemon"
python generate_pokemon.py --prompt "fire-type lizard pokemon"

# Creative combinations
python generate_pokemon.py --prompt "steel and dragon-type legendary pokemon"
python generate_pokemon.py --prompt "fairy-type butterfly pokemon"
```

### Generation Parameters

```bash
python generate_pokemon.py \
  --prompt "fire-type dragon pokemon" \
  --num_images 4 \
  --num_inference_steps 50 \
  --guidance_scale 7.5
```

- `num_inference_steps`: More steps = higher quality (but slower)
- `guidance_scale`: Higher = more prompt adherence (7.5 is good default)

## Training Timeline

On Apple M3:
- **Per epoch**: ~15-20 minutes (905 images)
- **10 epochs**: ~3 hours
- **20 epochs**: ~6 hours

## Checkpoints

Models are saved in `output/lora_model/`:
- `checkpoint-epoch-5/` - Checkpoint after 5 epochs
- `checkpoint-epoch-10/` - Checkpoint after 10 epochs
- `final_model/` - Final trained model

## Tips

### For Best Results:
1. Train for at least 10 epochs
2. Try different checkpoints (epoch 5, 10, 15) to see which works best
3. Use specific prompts that match the training data format
4. Experiment with guidance_scale (6.0-9.0)

### If Training is Too Slow:
1. Reduce epochs for testing (e.g., 3 epochs)
2. Close other applications
3. Let it run overnight for full training

### If You Get Errors:
1. Make sure you have 16GB RAM available
2. Close memory-intensive apps
3. Reduce batch_size to 1
4. Reduce lora_rank to 2

## Expected Results

After training, you should be able to generate:
- Pokemon in various styles
- New type combinations
- Creative creature designs
- Pokemon that match specific descriptions

The model learns:
- Pokemon art style
- Type-based characteristics (fire, water, etc.)
- Creature anatomy and proportions
- Color palettes and patterns

## File Structure

```
PokeForge/
├── data/
│   ├── raw/              # Original Pokemon images
│   └── processed/        # Preprocessed 512x512 images
├── output/
│   ├── lora_model/       # Trained models
│   │   ├── checkpoint-epoch-5/
│   │   ├── checkpoint-epoch-10/
│   │   └── final_model/
│   └── generated/        # Generated Pokemon images
├── train_pokemon_lora.py # Training script
└── generate_pokemon.py   # Generation script
```

## Next Steps

After successful training:
1. Try different checkpoints to find the best one
2. Experiment with creative prompts
3. Generate a variety of Pokemon types
4. Share your results!

## Troubleshooting

**Model not downloading?**
- Check internet connection
- Model will be cached after first download (~4GB)

**Out of memory?**
- Reduce batch_size to 1
- Reduce lora_rank to 2
- Close other applications

**Generated images look weird?**
- Train for more epochs
- Try different checkpoints
- Adjust guidance_scale (try 6.0-9.0)
- Use more specific prompts

**Training seems stuck?**
- It's normal! Each epoch takes 15-20 minutes
- Check that loss is decreasing over time
