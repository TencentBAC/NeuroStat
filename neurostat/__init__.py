"""NeuroStat: An end-to-end framework bridging the statistical and semantic gap
for robust machine-generated text detection.
"""

from .features import (
    compute_tf_features,
    TFFeatureExtractor,
    TBFeatureExtractor,
    AttentionPooling,
)
from .losses import SupConLoss, orthogonal_penalty
from .model import (
    NeuroStatDetector,
    FusionClassifier,
    build_neurostat_detector,
    load_causal_lm_backbone,
)

__all__ = [
    "NeuroStatDetector",
    "FusionClassifier",
    "build_neurostat_detector",
    "load_causal_lm_backbone",
    "compute_tf_features",
    "TFFeatureExtractor",
    "TBFeatureExtractor",
    "AttentionPooling",
    "SupConLoss",
    "orthogonal_penalty",
]
