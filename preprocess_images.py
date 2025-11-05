"""
Pokemon Image Preprocessing Script
Resizes images to 512x512, standardizes backgrounds, and prepares dataset for training
"""

import os
import json
from pathlib import Path
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Configuration
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"
PROCESSED_IMAGES_DIR = PROCESSED_DATA_DIR / "images"
METADATA_FILE = RAW_DATA_DIR / "metadata" / "all_pokemon.json"

# Create output directories
PROCESSED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Image settings
TARGET_SIZE = 512
BACKGROUND_COLOR = (255, 255, 255)  # White background


def resize_with_padding(image: Image.Image, target_size: int, bg_color: tuple) -> Image.Image:
    """
    Resize image to target_size x target_size with padding to maintain aspect ratio
    Also converts transparent backgrounds to solid color

    Args:
        image: PIL Image object
        target_size: Target width and height (square)
        bg_color: RGB tuple for background color

    Returns:
        Resized PIL Image with padding
    """
    # Convert RGBA to RGB if needed (handle transparency)
    if image.mode == 'RGBA':
        # Create white background
        background = Image.new('RGB', image.size, bg_color)
        # Paste the image on the background using alpha channel as mask
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Calculate scaling to fit within target size
    width, height = image.size
    scale = min(target_size / width, target_size / height)

    # Resize maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create new image with target size and background color
    new_image = Image.new('RGB', (target_size, target_size), bg_color)

    # Calculate position to paste (center the image)
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    # Paste resized image onto background
    new_image.paste(image, (paste_x, paste_y))

    return new_image


def preprocess_images():
    """
    Main preprocessing function
    Processes all Pokemon images and creates training dataset CSV
    """
    print("Pokemon Image Preprocessing")
    print("=" * 60)
    print(f"Source: {RAW_IMAGES_DIR}")
    print(f"Output: {PROCESSED_IMAGES_DIR}")
    print(f"Target size: {TARGET_SIZE}x{TARGET_SIZE}")
    print(f"Background: RGB{BACKGROUND_COLOR}")
    print("-" * 60)

    # Load metadata
    with open(METADATA_FILE, 'r') as f:
        all_metadata = json.load(f)

    print(f"Loaded metadata for {len(all_metadata)} Pokemon")

    # Create mapping of pokemon ID to metadata
    metadata_map = {m['id']: m for m in all_metadata}

    # Prepare dataset records
    dataset_records = []

    # Get all image files
    image_files = sorted(RAW_IMAGES_DIR.glob("*.png"))
    print(f"Found {len(image_files)} images to process")
    print()

    successful = 0
    failed = 0

    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Extract Pokemon ID from filename (e.g., "001_bulbasaur.png" -> 1)
            pokemon_id = int(image_path.stem.split('_')[0])

            # Get metadata for this Pokemon
            metadata = metadata_map.get(pokemon_id)
            if not metadata:
                print(f"Warning: No metadata found for {image_path.name}")
                failed += 1
                continue

            # Load and preprocess image
            image = Image.open(image_path)
            processed_image = resize_with_padding(image, TARGET_SIZE, BACKGROUND_COLOR)

            # Save processed image
            output_filename = f"{str(pokemon_id).zfill(3)}_{metadata['name']}.png"
            output_path = PROCESSED_IMAGES_DIR / output_filename
            processed_image.save(output_path, 'PNG', optimize=True)

            # Add to dataset
            dataset_records.append({
                'image_id': pokemon_id,
                'image_filename': output_filename,
                'image_path': str(output_path),
                'name': metadata['name'],
                'caption': metadata['caption'],
                'types': ', '.join(metadata['types']),
                'genus': metadata['genus'],
                'generation': metadata['generation']
            })

            successful += 1

        except Exception as e:
            print(f"\nError processing {image_path.name}: {e}")
            failed += 1

    print()
    print("=" * 60)
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # Create dataset CSV
    df = pd.DataFrame(dataset_records)
    dataset_csv_path = PROCESSED_DATA_DIR / "pokemon_dataset.csv"
    df.to_csv(dataset_csv_path, index=False)

    print(f"\nDataset CSV saved to: {dataset_csv_path}")
    print(f"Total records: {len(df)}")

    # Display sample
    print("\nSample records:")
    print(df[['image_filename', 'caption']].head(10).to_string(index=False))

    return df


if __name__ == "__main__":
    print()
    input("Press Enter to start preprocessing...")
    print()

    df = preprocess_images()

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print(f"Images saved to: {PROCESSED_IMAGES_DIR}")
    print(f"Dataset file: {PROCESSED_DATA_DIR / 'pokemon_dataset.csv'}")
