"""
Unified diversity evaluator combining all metrics.

Provides:
- Comprehensive diversity evaluation
- Comparison against reference datasets
- Percentile-based diversity scores
- Detailed reports with interpretations
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray

from .metrics import (
    vendi_score,
    coverage_score,
    density_score,
    feature_space_coverage,
    scenario_entropy,
    rarity_score,
    intra_cluster_diversity,
)
from .reference_datasets import ReferenceDatasetLoader

if TYPE_CHECKING:
    from video_curation_pipeline import PipelineResults, VideoEmbedder

logger = logging.getLogger(__name__)


@dataclass
class DiversityScore:
    """A single diversity metric with value and interpretation."""
    name: str
    value: float
    unit: str = "%"
    description: str = ""
    interpretation: str = ""
    reference_comparison: Optional[dict[str, float]] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "description": self.description,
            "interpretation": self.interpretation,
            "reference_comparison": self.reference_comparison
        }


@dataclass
class DiversityReport:
    """
    Complete diversity evaluation report.

    Contains all computed metrics with interpretations and
    comparisons against reference datasets.
    """
    # Dataset info
    dataset_name: str
    n_samples: int
    n_clusters: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Core metrics
    overall_diversity_score: float = 0.0  # Composite score (0-100%)
    vendi_score: Optional[DiversityScore] = None
    scenario_entropy: Optional[DiversityScore] = None
    balance_score: Optional[DiversityScore] = None

    # Coverage metrics
    absolute_coverage: Optional[DiversityScore] = None
    relative_coverage_bdd: Optional[DiversityScore] = None
    relative_coverage_nuimages: Optional[DiversityScore] = None

    # Quality metrics
    rarity_score: Optional[DiversityScore] = None
    intra_cluster_diversity: Optional[DiversityScore] = None
    density_score: Optional[DiversityScore] = None

    # Raw metrics data
    raw_metrics: dict[str, Any] = field(default_factory=dict)

    # Reference comparisons
    reference_datasets: list[str] = field(default_factory=list)

    def get_summary(self) -> dict[str, float]:
        """Get a summary of key metrics as percentages."""
        summary = {
            "Overall Diversity Score": self.overall_diversity_score,
        }

        for score in [
            self.vendi_score,
            self.scenario_entropy,
            self.balance_score,
            self.absolute_coverage,
            self.rarity_score,
            self.intra_cluster_diversity,
        ]:
            if score is not None:
                summary[score.name] = score.value

        return summary

    def to_dict(self) -> dict:
        """Convert report to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "n_samples": self.n_samples,
            "n_clusters": self.n_clusters,
            "timestamp": self.timestamp,
            "overall_diversity_score": self.overall_diversity_score,
            "metrics": {
                "vendi_score": self.vendi_score.to_dict() if self.vendi_score else None,
                "scenario_entropy": self.scenario_entropy.to_dict() if self.scenario_entropy else None,
                "balance_score": self.balance_score.to_dict() if self.balance_score else None,
                "absolute_coverage": self.absolute_coverage.to_dict() if self.absolute_coverage else None,
                "relative_coverage_bdd": self.relative_coverage_bdd.to_dict() if self.relative_coverage_bdd else None,
                "relative_coverage_nuimages": self.relative_coverage_nuimages.to_dict() if self.relative_coverage_nuimages else None,
                "rarity_score": self.rarity_score.to_dict() if self.rarity_score else None,
                "intra_cluster_diversity": self.intra_cluster_diversity.to_dict() if self.intra_cluster_diversity else None,
                "density_score": self.density_score.to_dict() if self.density_score else None,
            },
            "raw_metrics": self.raw_metrics,
            "reference_datasets": self.reference_datasets,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save report to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Report saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DiversityReport":
        """Load report from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class DiversityEvaluator:
    """
    Unified diversity evaluator for autonomous driving datasets.

    Computes comprehensive diversity metrics and provides interpretable
    scores as percentages. Supports comparison against reference datasets.

    Example:
        evaluator = DiversityEvaluator(embedder=embedder)

        # Evaluate with reference datasets
        report = evaluator.evaluate(
            results,
            reference_datasets=["bdd100k", "nuimages"],
            bdd_path="/path/to/bdd100k",
            nuimages_path="/path/to/nuimages"
        )

        # Print summary
        print(report.get_summary())

        # Save detailed report
        report.save("diversity_report.json")
    """

    def __init__(
        self,
        embedder: Optional["VideoEmbedder"] = None,
        cache_dir: Union[str, Path] = "./reference_embeddings",
        k_neighbors: int = 5
    ):
        """
        Initialize the evaluator.

        Args:
            embedder: VideoEmbedder for extracting reference embeddings.
            cache_dir: Directory for caching reference embeddings.
            k_neighbors: k parameter for k-NN based metrics.
        """
        self.embedder = embedder
        self.ref_loader = ReferenceDatasetLoader(embedder, cache_dir)
        self.k_neighbors = k_neighbors

    def evaluate(
        self,
        results: "PipelineResults",
        dataset_name: str = "target_dataset",
        reference_datasets: Optional[list[str]] = None,
        bdd_path: Optional[Union[str, Path]] = None,
        nuimages_path: Optional[Union[str, Path]] = None,
        max_reference_samples: int = 10000
    ) -> DiversityReport:
        """
        Perform comprehensive diversity evaluation.

        Args:
            results: Pipeline results containing embeddings and cluster labels.
            dataset_name: Name for the evaluated dataset.
            reference_datasets: List of reference datasets to compare against.
            bdd_path: Path to BDD100K images (for extraction if not cached).
            nuimages_path: Path to nuImages data (for extraction if not cached).
            max_reference_samples: Maximum samples from reference datasets.

        Returns:
            DiversityReport with all computed metrics.
        """
        logger.info(f"Starting diversity evaluation for {dataset_name}")
        logger.info(f"  Samples: {len(results.embeddings)}")
        logger.info(f"  Clusters: {len(results.cluster_info)}")

        embeddings = results.embeddings
        cluster_labels = results.cluster_labels

        # Initialize report
        report = DiversityReport(
            dataset_name=dataset_name,
            n_samples=len(embeddings),
            n_clusters=len(results.cluster_info)
        )

        # Load reference datasets
        reference_embeddings = {}
        if reference_datasets:
            for ref_name in reference_datasets:
                try:
                    ref_path = None
                    if ref_name == "bdd100k":
                        ref_path = bdd_path
                    elif ref_name == "nuimages":
                        ref_path = nuimages_path

                    ref_emb = self.ref_loader.load_or_extract(
                        ref_name,
                        data_dir=ref_path,
                        max_samples=max_reference_samples
                    )
                    reference_embeddings[ref_name] = ref_emb
                    report.reference_datasets.append(ref_name)
                    logger.info(f"  Loaded reference: {ref_name} ({len(ref_emb)} samples)")

                except Exception as e:
                    logger.warning(f"Failed to load reference {ref_name}: {e}")

        # Compute metrics
        logger.info("Computing diversity metrics...")

        # 1. Vendi Score
        logger.info("  Computing Vendi Score...")
        vs = vendi_score(embeddings)
        # Normalize to percentage (relative to max possible = n_samples)
        vs_percentage = (vs / len(embeddings)) * 100
        report.vendi_score = DiversityScore(
            name="Vendi Score (Effective Diversity)",
            value=round(vs_percentage, 2),
            unit="%",
            description="Effective number of unique scenarios as percentage of dataset size",
            interpretation=self._interpret_vendi_score(vs, len(embeddings))
        )
        report.raw_metrics["vendi_score_raw"] = vs

        # 2. Scenario Entropy
        logger.info("  Computing Scenario Entropy...")
        entropy_result = scenario_entropy(cluster_labels)
        report.scenario_entropy = DiversityScore(
            name="Scenario Distribution Entropy",
            value=round(entropy_result["normalized_entropy"], 2),
            unit="%",
            description="How uniformly samples are distributed across scenarios",
            interpretation=self._interpret_entropy(entropy_result["normalized_entropy"])
        )
        report.balance_score = DiversityScore(
            name="Cluster Balance Score",
            value=round(entropy_result["balance_score"], 2),
            unit="%",
            description="Equality of scenario representation (1 - Gini coefficient)",
            interpretation=self._interpret_balance(entropy_result["balance_score"])
        )
        report.raw_metrics["scenario_entropy"] = entropy_result

        # 3. Feature Space Coverage
        logger.info("  Computing Feature Space Coverage...")
        fsc_result = feature_space_coverage(embeddings)
        report.absolute_coverage = DiversityScore(
            name="Feature Space Coverage",
            value=round(fsc_result["absolute_coverage"], 2),
            unit="%",
            description="Proportion of feature space occupied",
            interpretation=self._interpret_coverage(fsc_result["absolute_coverage"])
        )
        report.raw_metrics["feature_space_coverage"] = fsc_result

        # 4. Intra-cluster Diversity
        logger.info("  Computing Intra-cluster Diversity...")
        icd_result = intra_cluster_diversity(embeddings, cluster_labels)
        report.intra_cluster_diversity = DiversityScore(
            name="Within-Cluster Diversity",
            value=round(icd_result["mean_diversity"], 2),
            unit="%",
            description="Average diversity within each scenario cluster",
            interpretation=self._interpret_intra_diversity(icd_result["mean_diversity"])
        )
        report.raw_metrics["intra_cluster_diversity"] = icd_result

        # 5. Rarity Score
        logger.info("  Computing Rarity Score...")
        rarity_result = rarity_score(embeddings, k=self.k_neighbors)
        report.rarity_score = DiversityScore(
            name="Rare Scenario Ratio",
            value=round(rarity_result["rare_sample_ratio"], 2),
            unit="%",
            description="Percentage of samples classified as rare/unique",
            interpretation=self._interpret_rarity(rarity_result["rare_sample_ratio"])
        )
        report.raw_metrics["rarity"] = {
            k: v for k, v in rarity_result.items()
            if k not in ["rare_indices", "rarity_scores"]  # Exclude large arrays
        }

        # 6. Reference-relative metrics
        if reference_embeddings:
            combined_ref = np.vstack(list(reference_embeddings.values()))

            # Coverage relative to combined reference
            logger.info("  Computing Coverage relative to references...")
            cov = coverage_score(embeddings, combined_ref, k=self.k_neighbors)
            # Add to raw metrics
            report.raw_metrics["coverage_vs_combined"] = cov * 100

            # Per-reference coverage
            for ref_name, ref_emb in reference_embeddings.items():
                logger.info(f"    Coverage vs {ref_name}...")
                ref_cov = coverage_score(embeddings, ref_emb, k=self.k_neighbors) * 100

                # FSC relative to reference
                fsc_rel = feature_space_coverage(
                    embeddings,
                    reference_embeddings=ref_emb
                )

                score = DiversityScore(
                    name=f"Coverage vs {ref_name.upper()}",
                    value=round(fsc_rel["relative_coverage"] if fsc_rel["relative_coverage"] else ref_cov, 2),
                    unit="%",
                    description=f"Proportion of {ref_name} feature space covered",
                    interpretation=self._interpret_relative_coverage(ref_cov, ref_name)
                )

                if ref_name == "bdd100k":
                    report.relative_coverage_bdd = score
                elif ref_name == "nuimages":
                    report.relative_coverage_nuimages = score

                report.raw_metrics[f"coverage_vs_{ref_name}"] = {
                    "coverage": ref_cov,
                    "fsc_relative": fsc_rel
                }

            # Density relative to reference
            logger.info("  Computing Density relative to references...")
            dens = density_score(embeddings, combined_ref, k=self.k_neighbors) * 100
            report.density_score = DiversityScore(
                name="Reference Density Score",
                value=round(dens, 2),
                unit="%",
                description="How well samples align with reference distribution",
                interpretation=self._interpret_density(dens)
            )
            report.raw_metrics["density_vs_reference"] = dens

        # 7. Compute overall diversity score
        report.overall_diversity_score = self._compute_overall_score(report)

        logger.info(f"Evaluation complete. Overall Diversity Score: {report.overall_diversity_score:.1f}%")

        return report

    def _compute_overall_score(self, report: DiversityReport) -> float:
        """
        Compute a weighted overall diversity score.

        Weights are assigned based on importance for autonomous driving:
        - Scenario coverage (most important for safety)
        - Distribution balance (important for training)
        - Effective diversity (quality of diversity)
        """
        weights = {
            "vendi": 0.20,
            "entropy": 0.15,
            "balance": 0.15,
            "coverage": 0.25,
            "relative_coverage": 0.15,
            "rarity": 0.10,
        }

        scores = []
        total_weight = 0

        if report.vendi_score:
            scores.append(report.vendi_score.value * weights["vendi"])
            total_weight += weights["vendi"]

        if report.scenario_entropy:
            scores.append(report.scenario_entropy.value * weights["entropy"])
            total_weight += weights["entropy"]

        if report.balance_score:
            scores.append(report.balance_score.value * weights["balance"])
            total_weight += weights["balance"]

        if report.absolute_coverage:
            scores.append(report.absolute_coverage.value * weights["coverage"])
            total_weight += weights["coverage"]

        if report.relative_coverage_bdd:
            scores.append(report.relative_coverage_bdd.value * weights["relative_coverage"])
            total_weight += weights["relative_coverage"]
        elif report.relative_coverage_nuimages:
            scores.append(report.relative_coverage_nuimages.value * weights["relative_coverage"])
            total_weight += weights["relative_coverage"]

        if report.rarity_score:
            # Rarity is inverse (more rare = less common scenarios captured)
            # But some rarity is good (capturing edge cases)
            # Optimal is around 5-15%
            rarity_val = report.rarity_score.value
            # Transform: 0% -> 50%, 5-15% -> 100%, >30% -> 50%
            if rarity_val < 5:
                rarity_score_adj = 50 + rarity_val * 10
            elif rarity_val <= 15:
                rarity_score_adj = 100
            else:
                rarity_score_adj = max(50, 100 - (rarity_val - 15) * 2)
            scores.append(rarity_score_adj * weights["rarity"])
            total_weight += weights["rarity"]

        if total_weight > 0:
            return round(sum(scores) / total_weight, 1)
        return 0.0

    def _interpret_vendi_score(self, vs: float, n_samples: int) -> str:
        """Interpret Vendi Score."""
        ratio = vs / n_samples
        if ratio > 0.5:
            return "Excellent: High effective diversity - most samples are unique"
        elif ratio > 0.3:
            return "Good: Moderate effective diversity - reasonable variety"
        elif ratio > 0.1:
            return "Fair: Limited effective diversity - some redundancy"
        else:
            return "Low: Many similar samples - consider expanding data collection"

    def _interpret_entropy(self, entropy: float) -> str:
        """Interpret scenario entropy."""
        if entropy > 90:
            return "Excellent: Nearly uniform distribution across scenarios"
        elif entropy > 75:
            return "Good: Well-balanced scenario distribution"
        elif entropy > 50:
            return "Moderate: Some scenarios are overrepresented"
        else:
            return "Imbalanced: Significant scenario skew - rebalancing recommended"

    def _interpret_balance(self, balance: float) -> str:
        """Interpret balance score."""
        if balance > 85:
            return "Excellent: Very equal representation across clusters"
        elif balance > 70:
            return "Good: Reasonably balanced distribution"
        elif balance > 50:
            return "Moderate: Some clusters dominate"
        else:
            return "Imbalanced: Long-tail distribution - consider upsampling rare scenarios"

    def _interpret_coverage(self, coverage: float) -> str:
        """Interpret feature space coverage."""
        if coverage > 80:
            return "Excellent: Broad coverage of feature space"
        elif coverage > 60:
            return "Good: Reasonable feature space coverage"
        elif coverage > 40:
            return "Moderate: Some regions underrepresented"
        else:
            return "Limited: Concentrated in few regions - expand data sources"

    def _interpret_relative_coverage(self, coverage: float, ref_name: str) -> str:
        """Interpret coverage relative to reference dataset."""
        if coverage > 70:
            return f"Excellent: Covers most of {ref_name}'s scenario space"
        elif coverage > 50:
            return f"Good: Covers majority of {ref_name}'s common scenarios"
        elif coverage > 30:
            return f"Moderate: Covers core {ref_name} scenarios, missing some"
        else:
            return f"Limited: Significant gaps compared to {ref_name}"

    def _interpret_intra_diversity(self, diversity: float) -> str:
        """Interpret intra-cluster diversity."""
        if diversity > 40:
            return "High: Diverse samples within clusters - consider finer clustering"
        elif diversity > 20:
            return "Moderate: Reasonable within-cluster variation"
        elif diversity > 10:
            return "Good: Clusters are relatively homogeneous"
        else:
            return "Excellent: Very tight clusters - good scenario separation"

    def _interpret_rarity(self, rarity: float) -> str:
        """Interpret rarity score."""
        if rarity > 20:
            return "High: Many unique/outlier samples - verify data quality"
        elif rarity > 10:
            return "Good: Healthy amount of rare scenarios captured"
        elif rarity > 5:
            return "Moderate: Some rare scenarios present"
        else:
            return "Low: Few rare scenarios - may miss edge cases"

    def _interpret_density(self, density: float) -> str:
        """Interpret density score relative to reference."""
        if density > 80:
            return "High: Samples concentrated in well-represented regions"
        elif density > 50:
            return "Good: Samples align well with reference distribution"
        elif density > 30:
            return "Moderate: Some samples in less common regions"
        else:
            return "Low: Many samples in rare regions - good for edge cases"

    def compare_datasets(
        self,
        results_list: list["PipelineResults"],
        dataset_names: list[str],
        **kwargs
    ) -> list[DiversityReport]:
        """
        Compare diversity across multiple datasets.

        Args:
            results_list: List of pipeline results to compare.
            dataset_names: Names for each dataset.
            **kwargs: Additional arguments passed to evaluate().

        Returns:
            List of DiversityReports for comparison.
        """
        reports = []
        for results, name in zip(results_list, dataset_names):
            report = self.evaluate(results, dataset_name=name, **kwargs)
            reports.append(report)
        return reports

    def quick_evaluate(
        self,
        embeddings: NDArray[np.float32],
        cluster_labels: NDArray[np.int32]
    ) -> dict[str, float]:
        """
        Quick evaluation without reference datasets.

        Returns a simple dictionary of key metrics.
        """
        return {
            "vendi_score": vendi_score(embeddings),
            "normalized_entropy": scenario_entropy(cluster_labels)["normalized_entropy"],
            "balance_score": scenario_entropy(cluster_labels)["balance_score"],
            "n_effective_clusters": len(np.unique(cluster_labels)),
        }
