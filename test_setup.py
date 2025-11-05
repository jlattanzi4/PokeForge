"""
Test script to validate training setup
Tests dataset loading and model initialization
"""

import torch
import pandas as pd
from pathlib import Path
from train_pokemon_lora import PokemonDataset
from transformers import CLIPTokenizer

BASE_DIR = Path(__file__).parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
DATASET_CSV = PROCESSED_DATA_DIR / "pokemon_dataset.csv"

def test_environment():
    """Test PyTorch and device availability"""
    print("=" * 60)
    print("Testing Environment")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\n✓ Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\n✓ Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\n⚠  Using device: CPU (training will be slow)")

    return device


def test_dataset():
    """Test dataset loading"""
    print("\n" + "=" * 60)
    print("Testing Dataset")
    print("=" * 60)

    if not DATASET_CSV.exists():
        print(f"✗ Dataset not found at: {DATASET_CSV}")
        return False

    print(f"Loading dataset from: {DATASET_CSV}")

    # Load CSV
    df = pd.read_csv(DATASET_CSV)
    print(f"✓ Dataset loaded: {len(df)} images")

    # Check columns
    required_columns = ['image_path', 'caption']
    for col in required_columns:
        if col not in df.columns:
            print(f"✗ Missing column: {col}")
            return False
    print(f"✓ Required columns present")

    # Check sample image
    sample_row = df.iloc[0]
    image_path = Path(sample_row['image_path'])

    if not image_path.exists():
        print(f"✗ Sample image not found: {image_path}")
        return False

    print(f"✓ Sample image exists: {image_path.name}")
    print(f"  Caption: {sample_row['caption']}")

    # Test dataset class
    print("\nTesting PokemonDataset class...")
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        dataset = PokemonDataset(
            csv_file=DATASET_CSV,
            image_size=512,
            tokenizer=tokenizer
        )

        print(f"✓ Dataset class initialized: {len(dataset)} samples")

        # Test loading one sample
        sample = dataset[0]
        print(f"✓ Sample loaded:")
        print(f"  Image shape: {sample['pixel_values'].shape}")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Caption: {sample['caption']}")

        return True

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return False


def test_model_loading():
    """Test model loading (without downloading full model)"""
    print("\n" + "=" * 60)
    print("Testing Model Components")
    print("=" * 60)

    try:
        from diffusers import StableDiffusionPipeline
        from peft import LoraConfig

        print("✓ Diffusers library imported")
        print("✓ PEFT library imported")

        # Test LoRA config
        lora_config = LoraConfig(
            r=4,
            lora_alpha=4,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        print("✓ LoRA config created")

        print("\nNote: Actual model download will happen during training")
        print("Expected download size: ~4GB for Stable Diffusion 1.5")

        return True

    except Exception as e:
        print(f"✗ Error with model components: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("POKEFORGE SETUP TEST")
    print("=" * 60)
    print()

    results = {}

    # Test environment
    device = test_environment()
    results['environment'] = True

    # Test dataset
    results['dataset'] = test_dataset()

    # Test model components
    results['model'] = test_model_loading()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.capitalize():20s} {status}")

    print("=" * 60)

    if all_passed:
        print("\n✓ All tests passed! Ready to train!")
        print("\nTo start training:")
        print("  python train_pokemon_lora.py")
        print("\nFor a quick test (1 epoch):")
        print("  python train_pokemon_lora.py --num_epochs 1")
    else:
        print("\n✗ Some tests failed. Please fix issues before training.")

    print()


if __name__ == "__main__":
    main()
