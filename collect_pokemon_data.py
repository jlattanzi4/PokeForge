"""
Pokemon Data Collection Script
Collects images from HybridShivam/Pokemon repository
and metadata from PokeAPI and Bulbapedia
"""

import os
import requests
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import urllib.request

# Configuration
BASE_DIR = Path(__file__).parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
IMAGES_DIR = RAW_DATA_DIR / "images"
METADATA_DIR = RAW_DATA_DIR / "metadata"

# Create directories
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# URLs
HYBRIDSHIVAM_BASE_URL = "https://raw.githubusercontent.com/HybridShivam/Pokemon/master/assets/images"
POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"

# Pokemon generations (Gen 1-9 = 1025 total Pokemon as of 2025)
MAX_POKEMON_ID = 1025


def download_image(pokemon_id: int, pokemon_name: str) -> bool:
    """
    Download Pokemon image from HybridShivam repository
    Returns True if successful, False otherwise
    """
    # HybridShivam uses zero-padded 3-digit IDs (001, 002, etc.)
    padded_id = str(pokemon_id).zfill(3)
    image_url = f"{HYBRIDSHIVAM_BASE_URL}/{padded_id}.png"
    image_path = IMAGES_DIR / f"{padded_id}_{pokemon_name}.png"

    try:
        urllib.request.urlretrieve(image_url, image_path)
        print(f"✓ Downloaded image for {pokemon_name} (ID: {padded_id})")
        return True
    except Exception as e:
        print(f"✗ Failed to download image for {pokemon_name} (ID: {padded_id}): {e}")
        return False


def get_pokemon_metadata(pokemon_id: int) -> Optional[Dict]:
    """
    Fetch Pokemon metadata from PokeAPI
    Returns metadata dict or None if failed
    """
    try:
        # Get main Pokemon data
        response = requests.get(f"{POKEAPI_BASE_URL}/pokemon/{pokemon_id}")
        response.raise_for_status()
        pokemon_data = response.json()

        # Get species data for flavor text
        species_response = requests.get(f"{POKEAPI_BASE_URL}/pokemon-species/{pokemon_id}")
        species_response.raise_for_status()
        species_data = species_response.json()

        # Extract relevant information
        types = [t["type"]["name"] for t in pokemon_data["types"]]

        # Get English flavor text (description)
        flavor_text = ""
        for entry in species_data.get("flavor_text_entries", []):
            if entry["language"]["name"] == "en":
                flavor_text = entry["flavor_text"].replace("\n", " ").replace("\f", " ")
                break

        # Get English genus (e.g., "Mouse Pokemon", "Seed Pokemon")
        genus = ""
        for entry in species_data.get("genera", []):
            if entry["language"]["name"] == "en":
                genus = entry["genus"]
                break

        metadata = {
            "id": pokemon_id,
            "name": pokemon_data["name"],
            "types": types,
            "height": pokemon_data["height"],
            "weight": pokemon_data["weight"],
            "base_stats": {
                stat["stat"]["name"]: stat["base_stat"]
                for stat in pokemon_data["stats"]
            },
            "flavor_text": flavor_text,
            "genus": genus,
            "generation": species_data.get("generation", {}).get("name", "")
        }

        print(f"✓ Fetched metadata for {metadata['name']}")
        return metadata

    except Exception as e:
        print(f"✗ Failed to fetch metadata for Pokemon ID {pokemon_id}: {e}")
        return None


def generate_caption(metadata: Dict) -> str:
    """
    Generate a text caption for the Pokemon based on metadata
    Format: "fire-type lizard pokemon" or "electric and steel-type mouse pokemon"
    """
    if not metadata:
        return ""

    # Get types
    types = metadata.get("types", [])
    if len(types) == 1:
        type_str = f"{types[0]}-type"
    else:
        type_str = f"{' and '.join(types)}-type"

    # Get genus (e.g., "Mouse Pokemon" -> "mouse")
    genus = metadata.get("genus", "").lower()
    # Remove "pokemon" from genus if present
    genus = genus.replace(" pokemon", "").replace(" pokémon", "").strip()

    # Build caption
    caption = f"{type_str} {genus} pokemon"

    return caption


def collect_pokemon_data(start_id: int = 1, end_id: int = MAX_POKEMON_ID,
                        delay: float = 0.1) -> None:
    """
    Main function to collect Pokemon images and metadata

    Args:
        start_id: Starting Pokemon ID (default: 1)
        end_id: Ending Pokemon ID (default: MAX_POKEMON_ID)
        delay: Delay between API calls to be respectful (seconds)
    """
    print(f"Starting Pokemon data collection (ID {start_id} to {end_id})")
    print(f"Images will be saved to: {IMAGES_DIR}")
    print(f"Metadata will be saved to: {METADATA_DIR}")
    print("-" * 60)

    successful_downloads = 0
    failed_downloads = 0
    all_metadata = []

    for pokemon_id in range(start_id, end_id + 1):
        print(f"\n[{pokemon_id}/{end_id}] Processing Pokemon ID: {pokemon_id}")

        # Get metadata from PokeAPI
        metadata = get_pokemon_metadata(pokemon_id)

        if metadata:
            pokemon_name = metadata["name"]

            # Download image from HybridShivam
            if download_image(pokemon_id, pokemon_name):
                # Generate caption
                caption = generate_caption(metadata)
                metadata["caption"] = caption

                # Save individual metadata file
                padded_id = str(pokemon_id).zfill(3)
                metadata_path = METADATA_DIR / f"{padded_id}_{pokemon_name}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                all_metadata.append(metadata)
                successful_downloads += 1

                print(f"  Caption: '{caption}'")
            else:
                failed_downloads += 1
        else:
            failed_downloads += 1

        # Be respectful to the APIs
        time.sleep(delay)

    # Save combined metadata file
    combined_metadata_path = METADATA_DIR / "all_pokemon.json"
    with open(combined_metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Successful: {successful_downloads}")
    print(f"Failed: {failed_downloads}")
    print(f"Total: {successful_downloads + failed_downloads}")
    print(f"\nCombined metadata saved to: {combined_metadata_path}")


if __name__ == "__main__":
    # Resume from Pokemon 906
    START_ID = 906
    END_ID = 1025  # All Pokemon through Gen 9

    print("Pokemon Data Collection Script")
    print("=" * 60)
    print(f"This script will download Pokemon images and metadata")
    print(f"for Pokemon IDs {START_ID} to {END_ID}")
    print()
    input("Press Enter to continue...")

    collect_pokemon_data(START_ID, END_ID)
