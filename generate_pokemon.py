"""
Pokemon Generation Script
Uses trained LoRA model to generate new Pokemon
"""

import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
import argparse
from datetime import datetime

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
GENERATED_DIR = OUTPUT_DIR / "generated"

# Create output directory
GENERATED_DIR.mkdir(parents=True, exist_ok=True)


def load_lora_pipeline(base_model, lora_path, device):
    """
    Load Stable Diffusion pipeline with trained LoRA weights
    """
    print(f"Loading base model: {base_model}")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float32,  # Use float32 for MPS
        safety_checker=None,  # Disable safety checker for faster generation
    )

    # Load LoRA weights
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA weights from: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    else:
        print("WARNING: No LoRA weights loaded. Using base model only.")

    # Use faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Move to device
    pipe = pipe.to(device)

    return pipe


def generate_pokemon(pipe, prompt, negative_prompt, num_images, args):
    """
    Generate Pokemon images from prompts
    """
    print(f"\nGenerating {num_images} Pokemon...")
    print(f"Prompt: {prompt}")
    if negative_prompt:
        print(f"Negative prompt: {negative_prompt}")
    print()

    images = []

    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")

        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.image_size,
            width=args.image_size,
        ).images[0]

        images.append(image)

        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pokemon_{timestamp}_{i+1}.png"
        output_path = GENERATED_DIR / filename
        image.save(output_path)
        print(f"Saved: {output_path}")

    return images


def main(args):
    print("=" * 60)
    print("Pokemon Generation")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"LoRA path: {args.lora_path}")
    print(f"Number of images: {args.num_images}")
    print("=" * 60)
    print()

    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")

    print()

    # Load pipeline
    pipe = load_lora_pipeline(args.base_model, args.lora_path, device)

    # Generate Pokemon
    images = generate_pokemon(
        pipe,
        args.prompt,
        args.negative_prompt,
        args.num_images,
        args
    )

    print("\n" + "=" * 60)
    print(f"Generation complete! Images saved to: {GENERATED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Pokemon with trained LoRA model")

    # Model args
    parser.add_argument(
        "--base_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=str(OUTPUT_DIR / "lora_model" / "final_model"),
        help="Path to trained LoRA weights"
    )

    # Generation args
    parser.add_argument(
        "--prompt",
        type=str,
        default="fire-type dragon pokemon",
        help="Text prompt for generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="blurry, low quality, distorted, deformed",
        help="Negative prompt"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale (higher = more prompt adherence)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="Generated image size"
    )

    args = parser.parse_args()

    main(args)
