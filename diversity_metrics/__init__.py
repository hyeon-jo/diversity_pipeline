"""
Diversity Metrics for Autonomous Driving Data Evaluation

Quantitative evaluation of dataset diversity using state-of-the-art metrics:

- Vendi Score: Effective number of unique scenarios (ICLR 2023)
- Coverage: Proportion of reference distribution covered
- Density: Manifold density estimation
- Feature Space Coverage (FSC): Grid-based coverage in embedding space
- Scenario Entropy: Distribution entropy across scenario types
- Rarity Score: Detection of rare/underrepresented scenarios

Supports relative evaluation against reference datasets:
- BDD100K
- nuImages
- Custom reference embeddings

References:
    See DIVERSITY_METRICS.md for detailed methodology and citations.
"""

from .metrics import (
    vendi_score,
    coverage_score,
    density_score,
    feature_space_coverage,
    scenario_entropy,
    rarity_score,
    intra_cluster_diversity,
)
from .evaluator import DiversityEvaluator, DiversityReport
from .reference_datasets import ReferenceDatasetLoader

__all__ = [
    "DiversityEvaluator",
    "DiversityReport",
    "ReferenceDatasetLoader",
    "vendi_score",
    "coverage_score",
    "density_score",
    "feature_space_coverage",
    "scenario_entropy",
    "rarity_score",
    "intra_cluster_diversity",
]

__version__ = "1.0.0"
