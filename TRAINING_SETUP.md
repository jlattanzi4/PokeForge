# Training Environment Setup

## System Specifications

**Hardware:**
- Model: MacBook Air (Apple M3)
- RAM: 16 GB
- GPU: Apple M3 (Neural Engine + GPU cores)

**Accelerator:**
- MPS (Metal Performance Shaders) - Available ✓
- Device: Apple Silicon GPU acceleration

## Installed Dependencies

### Core ML Libraries
- PyTorch: 2.9.0 (with MPS support)
- TorchVision: 0.24.0

### Diffusion Model Libraries
- Diffusers: 0.35.2
- Transformers: 4.57.1
- Accelerate: 1.11.0
- PEFT (LoRA): 0.17.1

### Data Handling
- Datasets: 4.3.0
- Pandas: 2.2.2
- PyArrow: 22.0.0

### Utilities
- SafeTensors: 0.6.2
- Hugging Face Hub: 0.36.0
- Tokenizers: 0.22.1

## Training Capabilities

### What Works on Apple M3
✓ **LoRA Fine-tuning** - Efficient parameter fine-tuning (recommended)
✓ **DreamBooth** - Subject-driven generation
✓ **Textual Inversion** - Learning new concepts
✓ **MPS Acceleration** - GPU-accelerated training

### Memory Considerations
With 16GB RAM:
- LoRA training: ✓ Works well
- Full fine-tuning: ⚠️ May require optimization (gradient checkpointing, mixed precision)
- Batch size: Start with 1-4 and increase as memory allows

## Recommended Training Settings for M3

```python
# Optimal settings for Apple M3 with 16GB RAM
training_args = {
    "device": "mps",
    "mixed_precision": "fp16",  # Use half precision
    "gradient_checkpointing": True,  # Save memory
    "batch_size": 1,  # Start small
    "gradient_accumulation_steps": 4,  # Effective batch size of 4
    "use_8bit_adam": False,  # Not available on MPS
    "lora_rank": 4,  # Lower rank = less memory
}
```

## Next Steps

1. **Choose Training Method:**
   - LoRA (Recommended) - Fast, memory-efficient
   - DreamBooth - More powerful but slower

2. **Download Base Model:**
   - Stable Diffusion 1.5 (~4GB)
   - Stable Diffusion 2.1 (~5GB)
   - SDXL (~7GB) - May be challenging on 16GB RAM

3. **Configure Training Script:**
   - Set hyperparameters
   - Configure LoRA settings
   - Set up logging/checkpointing

4. **Start Training:**
   - Monitor memory usage
   - Adjust batch size if needed
   - Save checkpoints regularly

## Troubleshooting

### If you encounter memory issues:
1. Reduce batch size to 1
2. Enable gradient checkpointing
3. Lower LoRA rank (e.g., 4 or 8 instead of 16)
4. Use smaller image resolution (384x384 instead of 512x512)
5. Close other applications

### If training is slow:
1. Verify MPS is being used (check device)
2. Reduce dataset size for testing
3. Use mixed precision (fp16)
4. Increase batch size if memory allows

## Environment Ready! ✓

Your training environment is fully configured and ready to fine-tune Stable Diffusion models on your Pokemon dataset.
