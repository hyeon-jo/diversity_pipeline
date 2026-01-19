"""Command-line interface for BEV diversity analysis."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bev_diversity.config import load_config
from bev_diversity.embedder.extractor import ImageEmbedder
from bev_diversity.metrics.vendi_family import VendiScoreFamily
from bev_diversity.utils import load_embeddings


def print_banner():
    """Print welcome banner."""
    print("=" * 80)
    print("BEV Diversity Analysis")
    print("Video Diversity Assessment using Vendi Score Family")
    print("=" * 80)
    print()


def print_results(scores: dict, num_images: int, kernel: str):
    """
    Print results in a formatted way.

    Args:
        scores: Dictionary of Vendi scores
        num_images: Number of images processed
        kernel: Kernel type used
    """
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    print(f"Images Processed: {num_images:,}")
    print(f"Kernel: {kernel}")
    print()
    print("Vendi Score Family:")
    print("-" * 80)

    # Sort scores by q value (for nice display)
    score_items = []
    for key, value in scores.items():
        if key == "VS_q=inf":
            q_val = float("inf")
        else:
            q_val = float(key.split("=")[1])
        score_items.append((q_val, key, value))

    score_items.sort()

    # Print scores with descriptions
    descriptions = {
        0.1: "[Most sensitive to rare items]",
        0.5: "",
        1.0: "[Original Vendi Score - Shannon entropy]",
        2.0: "",
        float("inf"): "[Most sensitive to common items/duplicates]",
    }

    for q_val, key, value in score_items:
        q_str = "inf" if np.isinf(q_val) else f"{q_val}"
        desc = descriptions.get(q_val, "")
        print(f"  VS (q={q_str:>4}):  {value:>8.2f}  {desc}")

    print()
    print("-" * 80)
    print()
    print("Interpretation:")
    print("  - Higher scores indicate greater diversity")
    print('  - Scores represent "effective number of unique scenarios"')

    # Add interpretation based on score ranges
    vs_min = min(v for k, v in scores.items() if k != "VS_q=inf")
    vs_max = max(v for k, v in scores.items())

    if "VS_q=inf" in scores:
        vs_inf = scores["VS_q=inf"]
        vs_01 = scores.get("VS_q=0.1", vs_max)
        ratio = vs_inf / vs_01 if vs_01 > 0 else 1.0

        if ratio < 0.3:
            print(
                f"  - VS_inf being much lower than VS_0.1 ({ratio:.2%}) suggests "
                "significant duplicates/very similar items"
            )
        elif ratio > 0.7:
            print("  - Scores are relatively uniform across q values, indicating balanced diversity")

    print()


def save_results(scores: dict, output_path: Path, embeddings: np.ndarray, config_path: str):
    """
    Save results to JSON file.

    Args:
        scores: Dictionary of Vendi scores
        output_path: Path to output JSON file
        embeddings: Embedding matrix (for metadata)
        config_path: Path to config file used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    output_data = {
        "vendi_scores": scores,
        "metadata": {
            "num_images": len(embeddings),
            "embedding_dim": embeddings.shape[1],
            "config_file": str(config_path),
        },
    }

    # Write JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")
    print("=" * 80)
    print()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BEV Diversity Analysis - Compute Vendi Score Family from frame images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings and compute Vendi scores
  python -m bev_diversity -c config.yaml

  # Compute Vendi scores from pre-computed embeddings
  python -m bev_diversity -c config_embedding_only.yaml

  # With verbose output
  python -m bev_diversity -c config.yaml -v

  # Save embeddings for later use
  python -m bev_diversity -c config.yaml --save-embeddings
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Override output directory from config",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Save extracted embeddings to .npy file",
    )

    args = parser.parse_args()

    try:
        # Print banner
        print_banner()

        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
        print("Configuration loaded successfully")
        print()

        # Override output directory if specified
        if args.output:
            config.output_dir = args.output

        # Check if running in embedding-only mode
        if config.is_embedding_only_mode():
            # Embedding-only mode: Load pre-computed embeddings
            print("Running in EMBEDDING-ONLY mode")
            print(f"Loading pre-computed embeddings from {config.embedding_path}...")
            embeddings = load_embeddings(config.embedding_path)
            print()

            if args.verbose:
                print("Configuration Summary:")
                print(f"  Embedding Path: {config.embedding_path}")
                print(f"  Output Directory: {config.output_dir}")
                print(f"  Kernel: {config.vendi.kernel}")
                print(f"  Q Values: {config.vendi.q_values}")
                print()

        else:
            # Standard mode: Extract embeddings from images
            print("Running in EXTRACTION mode")
            print()

            # Print configuration summary
            if args.verbose:
                print("Configuration Summary:")
                print(f"  Model Path: {config.model.model_path}")
                print(f"  Input Directory: {config.input_dir}")
                print(f"  Output Directory: {config.output_dir}")
                print(f"  Kernel: {config.vendi.kernel}")
                print(f"  Q Values: {config.vendi.q_values}")
                print()

            # Initialize embedder
            print("Initializing embedder...")
            embedder = ImageEmbedder(
                model_config=config.model,
                embedding_config=config.embedding,
            )
            print()

            # Extract embeddings
            print("Extracting embeddings from images...")
            embeddings = embedder.extract_from_directory(
                root_dir=Path(config.input_dir),
                extensions=config.image_extensions,
                show_progress=True,
            )
            print(f"Extracted {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")
            print()

            # Save embeddings if requested
            if args.save_embeddings:
                embeddings_path = Path(config.output_dir) / "embeddings.npy"
                embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(embeddings_path, embeddings)
                print(f"Embeddings saved to: {embeddings_path}")
                print()

        # Compute Vendi scores
        print("Computing Vendi Score Family...")
        vendi = VendiScoreFamily(
            kernel=config.vendi.kernel,
            rbf_gamma=config.vendi.rbf_gamma,
        )

        scores = vendi.compute_all(
            embeddings=embeddings,
            q_values=config.vendi.q_values,
            include_infinity=config.vendi.include_infinity,
        )

        # Print results
        print_results(scores, len(embeddings), config.vendi.kernel)

        # Save results
        output_path = Path(config.output_dir) / "vendi_scores.json"
        save_results(scores, output_path, embeddings, args.config)

        return 0

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
