"""Command-line interface for BEV diversity analysis."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from bev_diversity.config import load_config
from bev_diversity.embedder.extractor import ImageEmbedder
from bev_diversity.embedder.video_embedder import VideoEmbedder, load_embeddings_from_dir
from bev_diversity.metrics.vendi_family import VendiScoreFamily
from bev_diversity.utils import load_embeddings


def print_banner(mode: str = ""):
    """Print welcome banner."""
    print("=" * 80)
    print("BEV Diversity Analysis")
    if mode:
        print(f"Mode: {mode}")
    print("Video Diversity Assessment using Vendi Score Family")
    print("=" * 80)
    print()


def print_results(scores: dict, num_items: int, kernel: str, item_type: str = "Images"):
    """
    Print results in a formatted way.

    Args:
        scores: Dictionary of Vendi scores
        num_items: Number of items processed
        kernel: Kernel type used
        item_type: Type of items ("Images" or "Videos")
    """
    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)
    print()
    print(f"{item_type} Processed: {num_items:,}")
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


def save_results(
    scores: dict,
    output_path: Path,
    embeddings: np.ndarray,
    config_path: str,
    mode: str = "image",
    video_names: list = None,
):
    """
    Save results to JSON file.

    Args:
        scores: Dictionary of Vendi scores
        output_path: Path to output JSON file
        embeddings: Embedding matrix (for metadata)
        config_path: Path to config file used
        mode: Processing mode ("image" or "video")
        video_names: List of video names (for video mode)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare output data
    metadata = {
        "num_items": len(embeddings),
        "embedding_dim": embeddings.shape[1],
        "config_file": str(config_path),
        "mode": mode,
    }

    if video_names:
        metadata["video_names"] = video_names

    output_data = {
        "vendi_scores": scores,
        "metadata": metadata,
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
        description="BEV Diversity Analysis - Compute Vendi Score Family from images or videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # VIDEO MODE: Extract video embeddings (frame pooling) and compute Vendi scores
  python -m bev_diversity -c config_video.yaml

  # IMAGE MODE: Extract image embeddings and compute Vendi scores
  python -m bev_diversity -c config_image.yaml

  # EMBEDDING-ONLY MODE: Compute Vendi scores from pre-computed embeddings
  python -m bev_diversity -c config_embedding_only.yaml

  # With verbose output
  python -m bev_diversity -c config.yaml -v

  # Save combined embeddings file (individual files are always saved in video mode)
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
        help="Save extracted embeddings to .npy file (combined)",
    )

    args = parser.parse_args()

    try:
        # Load configuration first to determine mode
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)
        print("Configuration loaded successfully")
        print()

        # Override output directory if specified
        if args.output:
            config.output_dir = args.output

        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_names = None
        item_type = "Images"
        mode = "image"

        # Determine mode and process accordingly
        if config.is_embedding_only_mode():
            # EMBEDDING-ONLY MODE
            print_banner("EMBEDDING-ONLY")

            if config.embedding_dir:
                # Load from directory of individual embedding files
                print(f"Loading embeddings from directory: {config.embedding_dir}...")
                embeddings, video_names = load_embeddings_from_dir(Path(config.embedding_dir))
                item_type = "Videos"
                mode = "video"
            else:
                # Load from single embedding file
                print(f"Loading embeddings from file: {config.embedding_path}...")
                embeddings = load_embeddings(config.embedding_path)

            if args.verbose:
                print()
                print("Configuration Summary:")
                if config.embedding_dir:
                    print(f"  Embedding Directory: {config.embedding_dir}")
                else:
                    print(f"  Embedding Path: {config.embedding_path}")
                print(f"  Output Directory: {config.output_dir}")
                print(f"  Kernel: {config.vendi.kernel}")
                print(f"  Q Values: {config.vendi.q_values}")
                print()

        elif config.is_video_mode():
            # VIDEO MODE: Extract video embeddings with frame pooling
            print_banner("VIDEO EXTRACTION")

            if args.verbose:
                print("Configuration Summary:")
                print(f"  Model Path: {config.model.model_path}")
                print(f"  Input Directory: {config.input_dir}")
                print(f"  Output Directory: {config.output_dir}")
                print(f"  Num Frames: {config.video.num_frames}")
                print(f"  Sampling Strategy: {config.video.frame_sample_strategy}")
                print(f"  Pooling Strategy: {config.video.pooling_strategy}")
                print(f"  Kernel: {config.vendi.kernel}")
                print(f"  Q Values: {config.vendi.q_values}")
                print()

            # Initialize video embedder
            print("Initializing video embedder...")
            embedder = VideoEmbedder(
                model_config=config.model,
                video_config=config.video,
                input_config=config.input_config,
                output_config=config.output_config,
            )
            print()

            # Determine input mode (video_folders or single_folder)
            input_mode = "video_folders"
            if config.input_config:
                input_mode = config.input_config.mode

            # Extract embeddings
            if input_mode == "video_folders":
                print("Extracting video embeddings (video_folders mode)...")
                embedding_paths, video_names = embedder.extract_from_video_folders(
                    root_dir=Path(config.input_dir),
                    output_dir=output_dir,
                    extensions=config.image_extensions,
                    show_progress=True,
                )
            else:
                print("Extracting video embeddings (single_folder mode)...")
                frames_per_video = None
                if config.input_config:
                    frames_per_video = config.input_config.frames_per_video
                embedding_paths, video_names = embedder.extract_from_single_folder(
                    root_dir=Path(config.input_dir),
                    output_dir=output_dir,
                    frames_per_video=frames_per_video,
                    extensions=config.image_extensions,
                    show_progress=True,
                )

            print(f"Extracted {len(video_names)} video embeddings")
            print()

            # Load all embeddings for Vendi score computation
            embedding_dir = output_dir / (config.output_config.embedding_subdir if config.output_config else "embeddings")
            embeddings, _ = load_embeddings_from_dir(embedding_dir, show_progress=False)

            item_type = "Videos"
            mode = "video"

            # Save combined embeddings if requested
            if args.save_embeddings:
                combined_path = output_dir / "embeddings_all.npy"
                np.save(combined_path, embeddings)
                print(f"Combined embeddings saved to: {combined_path}")
                print()

        else:
            # IMAGE MODE: Extract individual image embeddings
            print_banner("IMAGE EXTRACTION")

            if args.verbose:
                print("Configuration Summary:")
                print(f"  Model Path: {config.model.model_path}")
                print(f"  Input Directory: {config.input_dir}")
                print(f"  Output Directory: {config.output_dir}")
                print(f"  Kernel: {config.vendi.kernel}")
                print(f"  Q Values: {config.vendi.q_values}")
                print()

            # Initialize image embedder
            print("Initializing image embedder...")
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
                embeddings_path = output_dir / "embeddings.npy"
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
        print_results(scores, len(embeddings), config.vendi.kernel, item_type)

        # Save results
        output_path = output_dir / "vendi_scores.json"
        save_results(scores, output_path, embeddings, args.config, mode, video_names)

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
