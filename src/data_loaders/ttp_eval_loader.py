"""
TTP-Eval dataset loader.

The TTP-Eval dataset contains annotated web pages for evaluating
the Topical and Toxic Prompt (TTP) classification system.
"""

import csv
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from ..utils.taxonomy import HarmLabel, Dimension


@dataclass
class TTPEvalSample:
    """Single sample from TTP-Eval dataset."""

    # Core fields
    url: str
    lang: str
    body: str

    # Ground truth labels (from human annotation)
    hate_violence_label: Dimension
    ideological_label: Dimension
    sexual_label: Dimension
    illegal_label: Dimension
    self_inflicted_label: Dimension

    # Optional fields from performance evaluation
    belonging_threats: Optional[str] = None
    ttp_output: Optional[str] = None
    ttp_h: Optional[str] = None
    ttp_ih: Optional[str] = None
    ttp_se: Optional[str] = None
    ttp_il: Optional[str] = None
    ttp_si: Optional[str] = None
    true_label: Optional[str] = None
    ttp_label: Optional[str] = None

    def get_harm_label(self) -> HarmLabel:
        """Get HarmLabel object with all harm dimensions."""
        return HarmLabel(
            hate_violence=self.hate_violence_label,
            ideological=self.ideological_label,
            sexual=self.sexual_label,
            illegal=self.illegal_label,
            self_inflicted=self.self_inflicted_label,
        )

    def is_toxic(self) -> bool:
        """Check if sample is toxic in any harm category."""
        return self.get_harm_label().is_toxic()

    def is_topical(self) -> bool:
        """Check if sample is topical (but not toxic)."""
        return self.get_harm_label().is_topical()

    def is_safe(self) -> bool:
        """Check if sample is safe in all harm categories."""
        return self.get_harm_label().is_safe()


class TTPEvalLoader:
    """
    Loader for TTP-Eval dataset.

    Example usage:
        loader = TTPEvalLoader("data/TTP-Eval/TTPEval.tsv")
        samples = loader.load()
        print(f"Loaded {len(samples)} samples")

        # Filter toxic samples
        toxic_samples = loader.filter_by_toxicity(toxic=True)
    """

    # Mapping from short harm category names to attribute names
    HARM_CATEGORY_MAP = {
        "H": "hate_violence_label",
        "IH": "ideological_label",
        "SE": "sexual_label",
        "IL": "illegal_label",
        "SI": "self_inflicted_label",
    }

    def __init__(self, filepath: str):
        """
        Initialize loader.

        Args:
            filepath: Path to TTPEval.tsv file
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"TTP-Eval file not found: {filepath}")

        self._samples: Optional[List[TTPEvalSample]] = None

    def load(self, force_reload: bool = False) -> List[TTPEvalSample]:
        """
        Load all samples from the dataset.

        Args:
            force_reload: If True, reload even if already loaded

        Returns:
            List of TTPEvalSample objects
        """
        if self._samples is not None and not force_reload:
            return self._samples

        samples = []

        with open(self.filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')

            for line_number, row in enumerate(reader, start=2):
                try:
                    # Parse labels - handle both "None" and "Intent" formats
                    sample = TTPEvalSample(
                        url=row['URL'],
                        lang=row['Lang'],
                        body=row['Body'],
                        hate_violence_label=Dimension.from_label(row['Hate&V Lab']),
                        ideological_label=Dimension.from_label(row['Ideologi Lab']),
                        sexual_label=Dimension.from_label(row['Sexual Lab']),
                        illegal_label=Dimension.from_label(row['Illegal Lab']),
                        self_inflicted_label=Dimension.from_label(row['Self-Infl Lab']),
                        belonging_threats=row.get('BelongingThreats'),
                        ttp_output=row.get('TTP_Out'),
                        ttp_h=row.get('TTP_H'),
                        ttp_ih=row.get('TTP_IH'),
                        ttp_se=row.get('TTP_SE'),
                        ttp_il=row.get('TTP_IL'),
                        ttp_si=row.get('TTP_SI'),
                        true_label=row.get('TrueLabel'),
                        ttp_label=row.get('TTPLabel'),
                    )
                    samples.append(sample)
                except Exception as e:
                    url = row.get('URL', 'Unknown URL')
                    raise ValueError(f"Failed to parse row {line_number} (URL: {url}): {e}") from e

        self._samples = samples
        return samples

    def filter_by_toxicity(
        self,
        toxic: Optional[bool] = None,
        topical: Optional[bool] = None,
        safe: Optional[bool] = None
    ) -> List[TTPEvalSample]:
        """
        Filter samples by overall toxicity status.

        Args:
            toxic: If True, return only toxic samples
            topical: If True, return only topical samples
            safe: If True, return only safe samples

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        filtered = []
        for sample in self._samples:
            if toxic is not None and sample.is_toxic() != toxic:
                continue
            if topical is not None and sample.is_topical() != topical:
                continue
            if safe is not None and sample.is_safe() != safe:
                continue
            filtered.append(sample)

        return filtered

    def filter_by_harm(
        self,
        harm_category: str,
        dimension: Optional[Dimension] = None
    ) -> List[TTPEvalSample]:
        """
        Filter samples by specific harm category.

        Args:
            harm_category: One of "H", "IH", "SE", "IL", "SI"
            dimension: Optional dimension filter (SAFE, TOPICAL, TOXIC).
                     When None, returns samples that contain some harm (i.e., label != SAFE).

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        if harm_category not in TTPEvalLoader.HARM_CATEGORY_MAP:
            raise ValueError(f"Invalid harm category: {harm_category}")

        attr_name = TTPEvalLoader.HARM_CATEGORY_MAP[harm_category]
        filtered = []

        for sample in self._samples:
            label = getattr(sample, attr_name)
            if (dimension is None and label != Dimension.SAFE) or (dimension is not None and label == dimension):
                filtered.append(sample)

        return filtered

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary with various statistics
        """
        if self._samples is None:
            self.load()

        total = len(self._samples)
        toxic_count = len([s for s in self._samples if s.is_toxic()])
        topical_count = len([s for s in self._samples if s.is_topical()])
        safe_count = len([s for s in self._samples if s.is_safe()])

        # Per-harm statistics
        harm_stats = {}
        for short_name, attr_name in TTPEvalLoader.HARM_CATEGORY_MAP.items():
            toxic = len([s for s in self._samples if getattr(s, attr_name) == Dimension.TOXIC])
            topical = len([s for s in self._samples if getattr(s, attr_name) == Dimension.TOPICAL])
            safe = len([s for s in self._samples if getattr(s, attr_name) == Dimension.SAFE])
            harm_stats[short_name] = {
                "toxic": toxic,
                "topical": topical,
                "safe": safe,
            }

        return {
            "total_samples": total,
            "toxic_samples": toxic_count,
            "topical_samples": topical_count,
            "safe_samples": safe_count,
            "per_harm_statistics": harm_stats,
        }

    def __len__(self) -> int:
        """Get number of samples."""
        if self._samples is None:
            self.load()
        return len(self._samples)

    def __getitem__(self, idx: int) -> TTPEvalSample:
        """Get sample by index."""
        if self._samples is None:
            self.load()
        return self._samples[idx]
