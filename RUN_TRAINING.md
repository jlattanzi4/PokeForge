# How to Run Training

## Quick Test (1 epoch - 20 minutes)

1. **Open Terminal**
   - Navigate to project directory:
     ```bash
     cd /Users/josephlattanzi/Scripts/PokeForge
     ```

2. **Start Training**
   ```bash
   python train_pokemon_lora.py --num_epochs 1
   ```

3. **What You'll See**
   ```
   ============================================================
   Pokemon LoRA Training
   ============================================================
   Dataset: data/processed/pokemon_dataset.csv
   Output: output/lora_model
   Epochs: 1
   ...
   
   Setting up training environment...
   Using device: MPS (Apple Silicon GPU)
   
   Loading pretrained model: runwayml/stable-diffusion-v1-5
   [First time: Downloads ~4GB model - takes 5-10 minutes]
   [Subsequent runs: Uses cached model - instant]
   
   Configuring LoRA (rank=4)...
   trainable params: 983,040 || all params: 860,000,000 || trainable%: 0.11%
   
   Loading dataset from pokemon_dataset.csv...
   Dataset size: 905 images
   
   Starting training for 1 epochs...
   
   ============================================================
   Epoch 1/1
   ============================================================
   Epoch 1/1: 100%|███████| 905/905 [15:30<00:00, loss: 0.0845]
   
   Step 0: loss = 0.1234
   Step 50: loss = 0.1056
   Step 100: loss = 0.0987
   ...
   
   Epoch 1 completed. Average loss: 0.0845
   
   Saving final model to output/lora_model/final_model...
   Training complete!
   ```

4. **Training Progress**
   - You'll see a progress bar for each epoch
   - Loss values are logged every 50 steps
   - Lower loss = better training
   - First epoch typically: loss 0.15 → 0.08

5. **When Complete**
   - Model saved to: `output/lora_model/final_model/`
   - Training stats saved to: `output/lora_model/final_model/training_stats.json`
   - Total time: ~20 minutes (first run includes model download)

## What to Do While Training

- **Let it run** - Don't close terminal
- **Don't sleep computer** - Training will pause if Mac sleeps
- **Monitor progress** - Watch loss values decrease
- **Check temperature** - Mac might get warm (normal for M3)

## If Something Goes Wrong

**Out of Memory Error:**
```bash
# Already using minimal settings, but if needed:
python train_pokemon_lora.py --num_epochs 1 --lora_rank 2
```

**Model Download Fails:**
- Check internet connection
- Try again - download resumes automatically

**Training Stalls:**
- Check Activity Monitor - should see Python using CPU/GPU
- If frozen >5 min, Ctrl+C and restart

**Mac Gets Too Hot:**
- Close other applications
- Ensure good ventilation
- Mac will throttle if needed (training slower but safe)

## After Test Completes

Come back and we'll:
1. Review the training results
2. Test generation with the trained model
3. Decide if you want to run full training

## Full Training Commands (For Later)

**10 epochs (recommended):**
```bash
python train_pokemon_lora.py --num_epochs 10
# Time: ~3 hours
```

**20 epochs (best quality):**
```bash
python train_pokemon_lora.py --num_epochs 20
# Time: ~6 hours
# Recommendation: Run overnight
```

## Tips

- **First run**: Model download adds 5-10 minutes
- **Subsequent runs**: Use cached model, start immediately
- **Save power**: Plug in your MacBook (training is intensive)
- **Check progress**: Loss should steadily decrease
- **Be patient**: Training takes time but results are worth it!

## Need Help?

If you encounter any issues, save the error message and come back - I can help troubleshoot!

