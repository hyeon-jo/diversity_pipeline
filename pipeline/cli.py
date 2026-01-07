import argparse
import sys
from pathlib import Path
import numpy as np

from .config import VideoEmbedderConfig, ClusteringConfig, AnalysisConfig
from .pipeline import VideoCurationPipeline
from .embedder import VideoEmbedder
from .demo import generate_synthetic_embeddings, print_results_summary

def progress_callback(current_dir: str, count: int):
    print(f"\r🔍 Scanning... {count} directories checked | Current: {current_dir[-30:] if len(current_dir) > 30 else current_dir:<30}", end="", flush=True)

def main():
    parser = argparse.ArgumentParser(
        description="Video-Centric Data Curation Pipeline"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with synthetic embeddings"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory containing video files to process"
    )
    parser.add_argument(
        "--frame-dir",
        type=str,
        default=None,
        help="Base directory containing frame directories (trainlake structure)"
    )
    parser.add_argument(
        "--frame-pattern",
        type=str,
        default="**/CMR_GT_Frame",
        help="Glob pattern to find frame directories (default: **/CMR_GT_Frame)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save embeddings (default: output)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of synthetic samples for demo mode"
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for k-NN graph"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Leiden resolution parameter"
    )
    parser.add_argument(
        "--skip-clustering",
        action="store_true",
        help="Only extract embeddings, skip clustering analysis"
    )
    # New options
    parser.add_argument(
        "--cache-dirs",
        action="store_true",
        help="Enable caching of found frame directories"
    )
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="Force rescan of directories ignoring cache"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for embedding extraction"
    )

    args = parser.parse_args()

    # Configuration
    embedder_config = VideoEmbedderConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )

    clustering_config = ClusteringConfig(
        k_neighbors=args.k_neighbors,
        resolution=args.resolution
    )

    analysis_config = AnalysisConfig(
        micro_cluster_threshold=3,
        low_density_percentile=15.0,
        outlier_distance_percentile=95.0
    )

    # Create pipeline
    pipeline = VideoCurationPipeline(
        embedder_config=embedder_config,
        clustering_config=clustering_config,
        analysis_config=analysis_config
    )

    if args.frame_dir is not None:
        # Frame directory mode: process pre-extracted frames
        print(f"Running in FRAME DIRECTORY MODE...")
        print(f"Base directory: {args.frame_dir}")
        print(f"Output directory: {args.output_dir}")

        # Find all frame directories
        if args.cache_dirs or args.rescan:
            cache_file = Path(args.output_dir) / "frame_dirs_cache.txt"
            print(f"Using cache file: {cache_file}")
            frame_dirs = VideoEmbedder.find_frame_directories_cached(
                args.frame_dir,
                cache_file=cache_file,
                force_rescan=args.rescan,
                progress_callback=progress_callback
            )
            print() # Newline after progress
        else:
            # Check if we should use optimized search (if structure matches expectation)
            # Or fall back to pattern match if pattern is custom
            if args.frame_pattern == "**/CMR_GT_Frame":
                print("Using structure-based optimized search...")
                frame_dirs = VideoEmbedder.find_frame_directories_optimized(
                    args.frame_dir,
                    progress_callback=progress_callback
                )
                print() # Newline
            else:
                print(f"Using pattern-based search: {args.frame_pattern}")
                frame_dirs = VideoEmbedder.find_frame_directories(
                    args.frame_dir,
                    pattern=args.frame_pattern
                )

        if not frame_dirs:
            print(f"No frame directories found in {args.frame_dir}")
            sys.exit(1)

        print(f"Found {len(frame_dirs)} frame directories to process")

        # Extract embeddings
        embeddings, video_names, failed_indices = pipeline.embedder.extract_embeddings_from_frame_dirs(
            frame_dirs,
            save_embeddings=True,
            show_progress=True
        )

        print(f"\nEmbedding extraction complete!")
        print(f"  Successful: {len(video_names)}")
        print(f"  Failed: {len(failed_indices)}")
        print(f"  Output directory: {args.output_dir}")

        if not args.skip_clustering and len(embeddings) > 10:
            # Run clustering analysis
            print("\nRunning clustering analysis...")
            results = pipeline.run_from_embeddings(
                embeddings=embeddings,
                video_paths=video_names,
                generate_captions=True
            )

            # Print results
            print_results_summary(results)
        elif args.skip_clustering:
            print("\nSkipping clustering analysis (--skip-clustering flag set)")
        else:
            print("\nSkipping clustering analysis (not enough samples)")

    elif args.demo or args.video_dir is None:
        # Demo mode: use synthetic embeddings
        print("Running in DEMO MODE with synthetic embeddings...")
        print(f"Generating {args.n_samples} synthetic video embeddings...")

        embeddings, ground_truth = generate_synthetic_embeddings(
            n_samples=args.n_samples,
            n_clusters=12
        )

        print(f"Ground truth clusters: {len(np.unique(ground_truth))}")

        # Run pipeline
        results = pipeline.run_from_embeddings(
            embeddings=embeddings,
            generate_captions=True
        )

        # Print results
        print_results_summary(results)

        # Evaluate clustering quality (since we have ground truth)
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(ground_truth, results.cluster_labels)
        nmi = normalized_mutual_info_score(ground_truth, results.cluster_labels)

        print("\nCLUSTERING QUALITY (vs. ground truth):")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Info: {nmi:.4f}")

    else:
        # Real mode: process actual video files
        from pathlib import Path

        video_dir = Path(args.video_dir)
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

        video_paths = [
            str(p) for p in video_dir.rglob("*")
            if p.suffix.lower() in video_extensions
        ]

        if not video_paths:
            print(f"No video files found in {video_dir}")
            sys.exit(1)

        print(f"Found {len(video_paths)} videos in {video_dir}")

        # Run pipeline
        results = pipeline.run(
            video_paths=video_paths,
            generate_captions=True
        )

        # Print results
        print_results_summary(results)

if __name__ == "__main__":
    main()
