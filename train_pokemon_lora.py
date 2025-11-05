"""
Pokemon LoRA Training Script
Fine-tunes Stable Diffusion on Pokemon dataset using LoRA
Optimized for Apple M3 with 16GB RAM
"""

import os
import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
import argparse
import json
from datetime import datetime

# Configuration
BASE_DIR = Path(__file__).parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "output"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class PokemonDataset(Dataset):
    """
    Custom Dataset for Pokemon images with captions
    """
    def __init__(self, csv_file, image_size=512, tokenizer=None):
        self.data = pd.read_csv(csv_file)
        self.image_size = image_size
        self.tokenizer = tokenizer

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load and transform image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        # Get caption
        caption = row['caption']

        # Tokenize caption if tokenizer provided
        if self.tokenizer:
            tokens = self.tokenizer(
                caption,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            input_ids = tokens.input_ids[0]
        else:
            input_ids = caption

        return {
            'pixel_values': image,
            'input_ids': input_ids,
            'caption': caption
        }


def setup_training(args):
    """
    Set up models, LoRA config, and optimizer
    """
    print("Setting up training environment...")

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU (WARNING: Training will be slow)")

    # Load pretrained model
    print(f"\nLoading pretrained model: {args.pretrained_model}")
    print("This may take a few minutes...")

    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model,
        subfolder="tokenizer"
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model,
        subfolder="text_encoder"
    )

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model,
        subfolder="vae"
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model,
        subfolder="unet"
    )

    # Freeze VAE and text encoder (only train UNet with LoRA)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Configure LoRA for UNet
    print(f"\nConfiguring LoRA (rank={args.lora_rank})...")
    lora_config = LoraConfig(
        r=args.lora_rank,  # LoRA rank
        lora_alpha=args.lora_rank,  # LoRA alpha (typically same as rank)
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],  # Attention layers
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # Move models to device
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)

    # Set to training mode
    unet.train()
    vae.eval()
    text_encoder.eval()

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8,
    )

    return {
        'device': device,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'unet': unet,
        'optimizer': optimizer,
    }


def train_epoch(models, dataloader, noise_scheduler, args, epoch):
    """
    Train for one epoch
    """
    device = models['device']
    unet = models['unet']
    vae = models['vae']
    text_encoder = models['text_encoder']
    optimizer = models['optimizer']

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
    total_loss = 0

    for step, batch in enumerate(progress_bar):
        # Move batch to device
        pixel_values = batch['pixel_values'].to(device)
        input_ids = batch['input_ids'].to(device)

        # Encode images to latent space
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

        # Sample noise
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        # Sample timestep
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            device=device
        )
        timesteps = timesteps.long()

        # Add noise to latents
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Get text embeddings
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0]

        # Predict noise
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Calculate loss
        loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        # Clear MPS cache every step to prevent memory leaks (MPS has memory issues)
        if device.type == "mps":
            torch.mps.empty_cache()

        # Update progress bar
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

        # Log every N steps
        if step % args.log_interval == 0:
            print(f"\nStep {step}: loss = {loss.item():.4f}")

    return total_loss / len(dataloader)


def main(args):
    """
    Main training function
    """
    print("=" * 60)
    print("Pokemon LoRA Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset_csv}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print("=" * 60)
    print()

    # Setup models
    models = setup_training(args)

    # Load dataset
    print(f"\nLoading dataset from {args.dataset_csv}...")
    dataset = PokemonDataset(
        csv_file=args.dataset_csv,
        image_size=args.image_size,
        tokenizer=models['tokenizer']
    )

    print(f"Dataset size: {len(dataset)} images")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # MPS works best with 0 workers
    )

    # Noise scheduler
    from diffusers import DDPMScheduler
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler"
    )

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print()

    training_stats = {
        'start_time': datetime.now().isoformat(),
        'args': vars(args),
        'epoch_losses': []
    }

    for epoch in range(args.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*60}")

        avg_loss = train_epoch(models, dataloader, noise_scheduler, args, epoch)

        print(f"\nEpoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        training_stats['epoch_losses'].append(avg_loss)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print(f"Saving checkpoint to {checkpoint_dir}...")
            models['unet'].save_pretrained(checkpoint_dir)

            # Save training stats
            with open(checkpoint_dir / "training_stats.json", 'w') as f:
                json.dump(training_stats, f, indent=2)

    # Save final model
    final_dir = Path(args.output_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving final model to {final_dir}...")
    models['unet'].save_pretrained(final_dir)

    # Save training stats
    training_stats['end_time'] = datetime.now().isoformat()
    with open(final_dir / "training_stats.json", 'w') as f:
        json.dump(training_stats, f, indent=2)

    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final model saved to: {final_dir}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pokemon LoRA model")

    # Dataset args
    parser.add_argument(
        "--dataset_csv",
        type=str,
        default=str(PROCESSED_DATA_DIR / "pokemon_dataset.csv"),
        help="Path to dataset CSV file"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Image size (default: 512)"
    )

    # Model args
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Pretrained model name or path"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="LoRA rank (default: 4, lower=less memory)"
    )

    # Training args
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(OUTPUT_DIR / "lora_model"),
        help="Output directory for model checkpoints"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=50,
        help="Log every N steps"
    )

    args = parser.parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args)
