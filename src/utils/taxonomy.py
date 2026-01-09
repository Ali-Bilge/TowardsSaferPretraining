"""
Taxonomy definitions for the three-dimensional safety classification framework.
"""

import logging
from enum import Enum
from typing import List, Dict
from dataclasses import dataclass

# Sentinel for default parameter
_MISSING = object()


class HarmCategory(Enum):
    """Five harm categories in the taxonomy."""
    HATE_VIOLENCE = "Hate & Violence"
    IDEOLOGICAL = "Ideological Harm"
    SEXUAL = "Sexual"
    ILLEGAL = "Illegal Activities"
    SELF_INFLICTED = "Self-Inflicted Harm"

    @classmethod
    def get_short_names(cls) -> Dict[str, str]:
        """Get mapping from full names to short names."""
        mapping = cls.get_short_name_mapping()
        return {category.value: short_name for category, short_name in mapping.items()}

    @classmethod
    def get_short_name_mapping(cls) -> Dict["HarmCategory", str]:
        """Get mapping from HarmCategory enum to short name."""
        return {
            cls.HATE_VIOLENCE: "H",
            cls.IDEOLOGICAL: "IH",
            cls.SEXUAL: "SE",
            cls.ILLEGAL: "IL",
            cls.SELF_INFLICTED: "SI",
        }

    @classmethod
    def from_short_name(cls, short_name: str) -> "HarmCategory":
        """Get HarmCategory from short name."""
        reverse_mapping = {v: k for k, v in cls.get_short_name_mapping().items()}
        if short_name not in reverse_mapping:
            raise ValueError(f"Unknown short name: '{short_name}'. Valid options: {list(reverse_mapping.keys())}")
        return reverse_mapping[short_name]


class Dimension(Enum):
    """Three dimensions in the taxonomy."""
    SAFE = "none"  # Also called "None" in the data
    TOPICAL = "topical"
    TOXIC = "intent"  # Also called "Intent" in the data (internally uses "intent")

    @classmethod
    def from_label(cls, label: str, default: "Dimension" = _MISSING) -> "Dimension":
        """
        Parse dimension from various label formats.

        Args:
            label: Label string to parse
            default: Default dimension to return if parsing fails (default: SAFE)

        Returns:
            Dimension enum value
        """
        if default is _MISSING:
            default = cls.SAFE

        label_lower = label.lower().strip()
        if label_lower in ["none", "safe", ""]:
            return cls.SAFE
        elif label_lower == "topical":
            return cls.TOPICAL
        elif label_lower in ["intent", "toxic"]:
            return cls.TOXIC
        else:
            # Handle unexpected values gracefully
            logging.warning(f"Unknown dimension label '{label}', defaulting to {default.value}")
            return default


@dataclass
class HarmLabel:
    """Represents harm labels across all categories."""
    hate_violence: Dimension = Dimension.SAFE
    ideological: Dimension = Dimension.SAFE
    sexual: Dimension = Dimension.SAFE
    illegal: Dimension = Dimension.SAFE
    self_inflicted: Dimension = Dimension.SAFE

    def _get_category_dimension_mapping(self) -> Dict[HarmCategory, Dimension]:
        """Get mapping from HarmCategory to corresponding dimension attributes."""
        return {
            HarmCategory.HATE_VIOLENCE: self.hate_violence,
            HarmCategory.IDEOLOGICAL: self.ideological,
            HarmCategory.SEXUAL: self.sexual,
            HarmCategory.ILLEGAL: self.illegal,
            HarmCategory.SELF_INFLICTED: self.self_inflicted,
        }

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        # Build mapping from short names to dimension values
        short_name_mapping = HarmCategory.get_short_name_mapping()
        attribute_mapping = self._get_category_dimension_mapping()

        return {short_name_mapping[category]: dimension.value
                for category, dimension in attribute_mapping.items()}

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "HarmLabel":
        """Create from dictionary with short names."""
        # Build reverse lookup from short names to attribute names
        short_name_mapping = HarmCategory.get_short_name_mapping()
        short_to_attr = {
            short_name: category.name.lower()
            for category, short_name in short_name_mapping.items()
        }

        # Create kwargs dict for cls() constructor
        kwargs = {}
        for short_name, attr_name in short_to_attr.items():
            label_value = data.get(short_name, "none")
            kwargs[attr_name] = Dimension.from_label(label_value)

        return cls(**kwargs)

    @classmethod
    def from_list(cls, labels: List[str]) -> "HarmLabel":
        """Create from list format: [H, IH, SE, IL, SI]."""
        if len(labels) != 5:
            raise ValueError(f"Expected 5 labels, got {len(labels)}")

        # Define expected order: index -> attribute mapping
        label_order = [
            ("hate_violence", 0),
            ("ideological", 1),
            ("sexual", 2),
            ("illegal", 3),
            ("self_inflicted", 4),
        ]

        return cls(**{
            attr_name: Dimension.from_label(labels[index])
            for attr_name, index in label_order
        })

    def is_toxic(self) -> bool:
        """Check if any harm category is toxic."""
        return any(attr == Dimension.TOXIC for attr in (self.hate_violence, self.ideological, self.sexual, self.illegal, self.self_inflicted))

    def is_topical(self) -> bool:
        """Check if any harm category is topical (but not toxic)."""
        return any(attr == Dimension.TOPICAL for attr in (self.hate_violence, self.ideological, self.sexual, self.illegal, self.self_inflicted)) and not self.is_toxic()

    def is_safe(self) -> bool:
        """Check if all harm categories are safe."""
        return all(dim == Dimension.SAFE for dim in (self.hate_violence, self.ideological, self.sexual, self.illegal, self.self_inflicted))

    def get_toxic_harms(self) -> List[HarmCategory]:
        """Get list of harm categories that are toxic."""
        # Build mapping from HarmCategory to dimension values
        attribute_mapping = self._get_category_dimension_mapping()

        return [category for category, dimension in attribute_mapping.items()
                if dimension == Dimension.TOXIC]
