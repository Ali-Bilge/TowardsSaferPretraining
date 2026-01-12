"""
HAVOC dataset loader.

HAVOC (Harmful Abstractions and Violations in Open Completions) is a benchmark
for evaluating LLM safety across multiple harm dimensions.
"""

import csv
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from collections import Counter
import ast
import logging

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

# Model configuration mapping: model_key -> (response_col, judge_col, label_col)
MODEL_CONFIGS = {
    'llama_3b': ('PrefixLlama3BResponse', 'Llama3BJudge', 'Llama3BLab'),
    'llama_1b': ('PrefixLlama1BResponse', 'Llama1BJudge', 'Llama1BLab'),
    'mistral_7b': ('PrefixMistral7BResponse', 'Mistral7BJudge', 'Mistral7BLab'),
    'gemma_2b': ('PrefixGemma2BResponse', 'Gemma2BJudge', 'Gemma2BLab'),
    'gemma_9b': ('PrefixGemma9BResponse', 'Gemma9BJudge', 'Gemma9BLab'),
    'gemma_27b': ('PrefixGemma27BResponse', 'Gemma27BJudge', 'Gemma27BLab'),
}


@dataclass
class ModelEvaluation:
    """Model evaluation result with response, judge, and label."""
    response: Optional[str] = None
    judge: Optional[str] = None
    label: Optional[str] = None


@dataclass
class HAVOCSample:
    """Single sample from HAVOC dataset."""

    # Core fields
    prefix: str
    suffix: str
    prefix_label: HarmLabel
    separator: str = " "

    # Optional fields from model evaluation
    snippet: Optional[str] = None
    prefix_gen_gpt_response: Optional[str] = None

    # Model responses and judgments
    model_evaluations: Dict[str, ModelEvaluation] = field(default_factory=dict)

    def get_full_text(self) -> str:
        """Get concatenated prefix + suffix."""
        # Only insert separator when both parts are present, and trim boundary
        # whitespace to avoid double spaces/unwanted separators.
        if self.prefix.strip() and self.suffix.strip():
            prefix = self.prefix.rstrip()
            suffix = self.suffix.lstrip()
            return prefix + self.separator + suffix if self.separator else prefix + suffix

        return self.prefix + self.suffix

    def is_toxic_prefix(self) -> bool:
        """Check if prefix is toxic."""
        return self.prefix_label.is_toxic()

    def is_topical_prefix(self) -> bool:
        """Check if prefix is topical."""
        return self.prefix_label.is_topical()

    def is_safe_prefix(self) -> bool:
        """Check if prefix is safe."""
        return self.prefix_label.is_safe()

    def get_leakage_type(self) -> str:
        """
        Determine leakage type based on prefix label.

        Returns:
            "provocative" if prefix is toxic
            "passive" if prefix is topical
            "neutral" if prefix is safe
        """
        if self.is_toxic_prefix():
            return "provocative"
        elif self.is_topical_prefix():
            return "passive"
        else:
            return "neutral"


class HAVOCLoader:
    """
    Loader for HAVOC dataset.

    Example usage:
        loader = HAVOCLoader("data/HAVOC/havoc.tsv")
        samples = loader.load()
        print(f"Loaded {len(samples)} samples")

        # Filter by leakage type
        provocative = loader.filter_by_leakage_type("provocative")
    """

    def __init__(self, filepath: str, modeleval_filepath: Optional[str] = None):
        """
        Initialize loader.

        Args:
            filepath: Path to havoc.tsv file
            modeleval_filepath: Optional path to havoc_modeleval.tsv
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"HAVOC file not found: {filepath}")

        self.modeleval_filepath = Path(modeleval_filepath) if modeleval_filepath else None
        if self.modeleval_filepath and not self.modeleval_filepath.exists():
            raise FileNotFoundError(f"Model evaluation file not found: {modeleval_filepath}")
        self._samples: Optional[List[HAVOCSample]] = None

    def _parse_label_list(self, label_str: str) -> HarmLabel:
        """
        Parse label from string list format.

        Args:
            label_str: String like "['topical', 'none', 'none', 'none', 'none']"

        Returns:
            HarmLabel object
        """
        label_str = (label_str or "").strip()
        # Some rows contain "[]" placeholders; treat as unknown and default safe.
        if label_str in {"[]", ""}:
            # This is common in the released TSVs; don't spam warnings.
            logger.debug(f"Invalid label list '{label_str}', defaulting to all safe")
            return HarmLabel()

        try:
            # Use ast.literal_eval for safe parsing
            labels = ast.literal_eval(label_str)

            # Handle empty or malformed labels
            if not labels or len(labels) != 5:
                logger.debug(f"Invalid label list '{label_str}', defaulting to all safe")
                return HarmLabel()  # Returns all safe

            return HarmLabel.from_list(labels)
        except Exception as e:
            # Keep as warning: truly malformed labels are unexpected.
            logger.warning(f"Failed to parse label '{label_str}': {e}. Defaulting to all safe", exc_info=True)
            return HarmLabel()  # Returns all safe

    def load(self, force_reload: bool = False) -> List[HAVOCSample]:
        """
        Load all samples from the dataset.

        Args:
            force_reload: If True, reload even if already loaded

        Returns:
            List of HAVOCSample objects
        """
        if self._samples is not None and not force_reload:
            return self._samples

        samples: List[HAVOCSample] = []

        # Use csv.reader instead of DictReader for robustness: havoc.tsv can contain
        # a small number of malformed rows (e.g., extra tabs). We reconstruct
        # Prefix/Suffix/PrefixLab in a best-effort way and skip rows we can't parse.
        # NOTE: we keep Python's default CSV quoting behavior here because the released TSVs
        # contain quote characters and the corresponding `havoc_modeleval.tsv` is parsed with
        # the same behavior. A very small number of rows contain unbalanced quotes; we handle
        # those with best-effort reconstruction below.
        with open(self.filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            try:
                header = next(reader)
            except StopIteration:
                self._samples = []
                return []

            if header[:3] != ["Prefix", "Suffix", "PrefixLab"]:
                raise ValueError(
                    f"Unexpected HAVOC header {header[:3]} (expected ['Prefix','Suffix','PrefixLab'])."
                )

            for line_number, row in enumerate(reader, start=2):
                # Expected row = [Prefix, Suffix, PrefixLab]
                if not row:
                    continue

                if len(row) == 3:
                    prefix, suffix, prefix_lab_str = row
                elif len(row) > 3:
                    # Assume stray tabs landed in the suffix; keep first column as Prefix,
                    # last column as PrefixLab, and stitch the middle back into Suffix.
                    prefix = row[0]

                    # Validate if the last column looks like a valid label (starts with '[')
                    potential_label = row[-1].strip()
                    if potential_label.startswith('['):
                        try:
                            # Try to parse as a valid label list to confirm it's not trailing garbage
                            ast.literal_eval(potential_label)
                            prefix_lab_str = row[-1]
                            suffix = "\t".join(row[1:-1])
                            logger.warning(
                                f"HAVOC TSV row {line_number}: expected 3 fields, got {len(row)}; "
                                "reconstructed suffix by joining middle fields."
                            )
                        except (ValueError, SyntaxError):
                            # Last column starts with '[' but is not valid Python list - treat as garbage
                            prefix_lab_str = ""
                            suffix = "\t".join(row[1:])
                            logger.warning(
                                f"HAVOC TSV row {line_number}: expected 3 fields, got {len(row)}; "
                                f"last column '{potential_label}' appears to be label-like but is invalid, "
                                "treating entire tail as suffix."
                            )
                    else:
                        # Last column doesn't look like a label - treat entire tail as suffix
                        prefix_lab_str = ""
                        suffix = "\t".join(row[1:])
                        logger.warning(
                            f"HAVOC TSV row {line_number}: expected 3 fields, got {len(row)}; "
                            f"last column '{potential_label}' doesn't match expected label pattern, "
                            "treating entire tail as suffix."
                        )
                else:  # len(row) == 1 or 2
                    prefix = row[0]
                    suffix = row[1] if len(row) == 2 else ""
                    prefix_lab_str = ""
                    logger.warning(
                        f"HAVOC TSV row {line_number}: expected 3 fields, got {len(row)}; "
                        "using empty label."
                    )

                prefix_label = self._parse_label_list(prefix_lab_str)

                samples.append(
                    HAVOCSample(
                        prefix=prefix,
                        suffix=suffix,
                        prefix_label=prefix_label,
                    )
                )

        # Load model evaluations if available
        if self.modeleval_filepath:
            self._load_model_evaluations(samples)

        self._samples = samples
        return samples

    def _create_sample_key(self, prefix: str, suffix: str) -> str:
        """Create a canonicalized key for sample matching.

        Uses '\0' as a delimiter between prefix and suffix to prevent collisions.
        Format: normalized_prefix + '\0' + normalized_suffix
        """
        # Normalize whitespace and concatenate with null byte delimiter
        normalized_prefix = re.sub(r'\s+', ' ', prefix.strip())
        normalized_suffix = re.sub(r'\s+', ' ', suffix.strip())
        return normalized_prefix + '\0' + normalized_suffix

    def _populate_model_evaluations(self, sample: HAVOCSample, eval_row: Dict[str, str]):
        """Populate model evaluation data for a single sample."""
        sample.snippet = eval_row.get('Snippet')
        sample.prefix_gen_gpt_response = eval_row.get('PrefixGenGPTResponse')

        for model_key, (response_col, judge_col, label_col) in MODEL_CONFIGS.items():
            sample.model_evaluations[model_key] = ModelEvaluation(
                response=eval_row.get(response_col),
                judge=eval_row.get(judge_col),
                label=eval_row.get(label_col)
            )

    def _load_model_evaluations(self, samples: List[HAVOCSample]):
        """
        Load model evaluation data and merge with samples.

        Uses explicit matching by canonicalized prefix+suffix content rather than
        fragile index-based matching. Falls back to index-based matching only if
        explicit matching fails, with clear warnings.
        """
        # First pass: read model evaluations into lookup dict and cache rows
        assert self.modeleval_filepath is not None, "modeleval_filepath should not be None"
        model_eval_lookup = {}
        duplicate_keys = 0
        cached_eval_rows = []
        # For `havoc_modeleval.tsv`, keep the default CSV quoting behavior. This file contains
        # JSON-like strings in some columns, and disabling quoting can misalign columns due to
        # embedded tabs/newlines.
        with open(self.modeleval_filepath, 'r', encoding='utf-8', errors='replace', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                cached_eval_rows.append(row)
                prefix = (row.get('Prefix') or '').strip()
                suffix = (row.get('Suffix') or '').strip()
                if prefix or suffix:  # Skip completely empty rows
                    key = self._create_sample_key(prefix, suffix)
                    if key in model_eval_lookup:
                        # Keep the first occurrence for determinism; count duplicates for summary.
                        duplicate_keys += 1
                        continue
                    model_eval_lookup[key] = row

        if duplicate_keys:
            logger.warning(
                f"Duplicate key(s) found in model evaluations: {duplicate_keys} duplicates; "
                "kept first occurrence for deterministic matching."
            )

        # Second pass: match and merge with samples
        matched_count = 0
        for sample in samples:
            sample_key = self._create_sample_key(sample.prefix, sample.suffix)
            eval_row = model_eval_lookup.get(sample_key)

            if eval_row is not None:
                self._populate_model_evaluations(sample, eval_row)
                matched_count += 1
            else:
                # No explicit match found - this is a problem
                logger.warning(
                    f"No matching model evaluation found for sample with prefix: "
                    f"{sample.prefix[:50]}... and suffix: {sample.suffix[:50]}..."
                )

        # Log summary statistics
        total_samples = len(samples)
        logger.info(
            f"Model evaluation matching: {matched_count}/{total_samples} samples matched "
            f"({len(model_eval_lookup)} evaluation rows available)"
        )

        # Fallback: index-based matching for any remaining unmatched samples
        # This assumes the files are in the same order and should only be used
        # as a last resort when explicit matching fails
        if matched_count < total_samples:
            logging.warning(
                f"Falling back to index-based matching for {total_samples - matched_count} "
                "unmatched samples. This assumes files are in identical order."
            )

            # Use cached eval rows for index-based fallback
            eval_rows = cached_eval_rows

            for idx, sample in enumerate(samples):
                if idx >= len(eval_rows):
                    break

                # Only apply fallback if this sample wasn't already matched
                sample_key = self._create_sample_key(sample.prefix, sample.suffix)
                if sample_key not in model_eval_lookup:
                    eval_row = eval_rows[idx]
                    logging.debug(
                        f"Index-based fallback: matching sample at index {idx} "
                        f"with eval row {idx}"
                    )
                    self._populate_model_evaluations(sample, eval_row)

    def filter_by_leakage_type(self, leakage_type: str) -> List[HAVOCSample]:
        """
        Filter samples by leakage type.

        Args:
            leakage_type: One of "provocative", "passive", "neutral"

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        assert self._samples is not None, "Samples should be loaded after load() call"

        valid_types = ["provocative", "passive", "neutral"]
        if leakage_type not in valid_types:
            raise ValueError(f"Invalid leakage type. Must be one of: {valid_types}")

        return [s for s in self._samples if s.get_leakage_type() == leakage_type]

    def filter_by_harm(
        self,
        harm_category: str,
        dimension: Optional[Dimension] = None
    ) -> List[HAVOCSample]:
        """
        Filter samples by specific harm category in prefix.

        Args:
            harm_category: One of "H", "IH", "SE", "IL", "SI"
            dimension: Optional dimension filter (SAFE, TOPICAL, TOXIC)

        Returns:
            Filtered list of samples
        """
        if self._samples is None:
            self.load()

        assert self._samples is not None, "Samples should be loaded after load() call"

        harm_map = {
            "H": "hate_violence",
            "IH": "ideological",
            "SE": "sexual",
            "IL": "illegal",
            "SI": "self_inflicted",
        }

        if harm_category not in harm_map:
            raise ValueError(f"Invalid harm category: {harm_category}")

        attr_name = harm_map[harm_category]
        filtered = []

        for sample in self._samples:
            label = getattr(sample.prefix_label, attr_name)
            if (dimension is None and label not in (None, Dimension.SAFE)) or (dimension is not None and label == dimension):
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

        assert self._samples is not None, "Samples should be loaded after load() call"

        total = len(self._samples)

        # Leakage type distribution - compute in single pass
        leakage_counts = Counter(sample.get_leakage_type() for sample in self._samples)
        provocative = leakage_counts["provocative"]
        passive = leakage_counts["passive"]
        neutral = leakage_counts["neutral"]

        # Per-harm statistics
        harm_stats = {}
        for short_name, attr_name in [
            ("H", "hate_violence"),
            ("IH", "ideological"),
            ("SE", "sexual"),
            ("IL", "illegal"),
            ("SI", "self_inflicted"),
        ]:
            toxic = len([s for s in self._samples
                        if getattr(s.prefix_label, attr_name) == Dimension.TOXIC])
            topical = len([s for s in self._samples
                          if getattr(s.prefix_label, attr_name) == Dimension.TOPICAL])
            safe = len([s for s in self._samples
                       if getattr(s.prefix_label, attr_name) == Dimension.SAFE])
            harm_stats[short_name] = {
                "toxic": toxic,
                "topical": topical,
                "safe": safe,
            }

        return {
            "total_samples": total,
            "leakage_types": {
                "provocative": provocative,
                "passive": passive,
                "neutral": neutral,
            },
            "per_harm_statistics": harm_stats,
        }

    def __len__(self) -> int:
        """Get number of samples."""
        if self._samples is None:
            self.load()
        assert self._samples is not None, "Samples should be loaded after load() call"
        return len(self._samples)

    def __getitem__(self, idx: Union[int, slice]) -> Union[HAVOCSample, List[HAVOCSample]]:
        """Get sample by index or slice."""
        if self._samples is None:
            self.load()
        assert self._samples is not None, "Samples should be loaded after load() call"
        if isinstance(idx, slice):
            return self._samples[idx]
        return self._samples[idx]
