"""
Perspective API wrapper for toxicity detection.

Requires API key from: https://developers.perspectiveapi.com/s/
"""

import logging
from typing import Optional
import time

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

try:
    from googleapiclient import discovery  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore
    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False


class PerspectiveAPI:
    """
    Wrapper for Google's Perspective API.

    Maps Perspective toxicity scores to four-dimensional taxonomy.
    Uses threshold-based classification.

    Example:
        api = PerspectiveAPI(api_key="...")
        label = api.predict("This is a test text")
    """

    # Perspective API attributes we use
    ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
        "SEXUALLY_EXPLICIT",
    ]

    # Thresholds for classification (from paper experimentation)
    TOXIC_THRESHOLD = 0.7    # High confidence toxic
    TOPICAL_THRESHOLD = 0.3  # Medium confidence (topical)

    def __init__(
        self,
        api_key: str,
        toxic_threshold: float = 0.7,
        topical_threshold: float = 0.3,
        languages: Optional[list] = None
    ):
        """
        Initialize Perspective API client.

        Args:
            api_key: Perspective API key
            toxic_threshold: Score threshold for toxic classification
            topical_threshold: Score threshold for topical classification
            languages: List of language codes to analyze (default: ['en'])
        """
        if not PERSPECTIVE_AVAILABLE:
            raise RuntimeError(
                "Perspective API requires: pip install google-api-python-client"
            )

        self.toxic_threshold = toxic_threshold
        self.topical_threshold = topical_threshold
        self.languages = languages if languages is not None else ['en']

        # Build API client
        try:
            self.client = discovery.build(
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
                static_discovery=False,
            )
            logger.info("Initialized Perspective API client")
        except Exception as e:
            logger.error(f"Failed to initialize Perspective API client: {e}")
            self.client = None
            raise RuntimeError(f"Perspective API initialization failed: {e}")

    def get_scores(self, text: str, max_retries: int = 3) -> Optional[dict]:
        """
        Get toxicity scores from Perspective API.

        Args:
            text: Text to analyze
            max_retries: Number of retries on failure

        Returns:
            Dictionary of attribute scores, or None on error
        """
        # Validate and clean text
        text = text.strip()
        if not text:
            logger.warning("Empty text provided to Perspective API")
            return None

        if len(text) > 20000:
            logger.warning(f"Text too long ({len(text)} chars), truncating to 20000")
            text = text[:20000]

        analyze_request = {
            'comment': {'text': text},
            'requestedAttributes': {attr: {} for attr in self.ATTRIBUTES},
            'languages': self.languages,
        }

        for attempt in range(max_retries):
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()

                # Extract scores
                scores = {}
                for attr in self.ATTRIBUTES:
                    if attr in response['attributeScores']:
                        score = response['attributeScores'][attr]['summaryScore']['value']
                        scores[attr] = score

                return scores

            except HttpError as e:
                if e.resp.status == 429 or (500 <= e.resp.status < 600):  # Rate limit or server error
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited or server error ({e.resp.status}), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Perspective API error: {e}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None

        logger.error(f"Failed after {max_retries} retries")
        return None

    def scores_to_label(self, scores: dict) -> 'HarmLabel':
        """
        Convert Perspective scores to HarmLabel.

        Maps Perspective attributes to our taxonomy:
        - TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY -> Hate/Violence (H)
        - IDENTITY_ATTACK -> Ideological Harm (IH)
        - SEXUALLY_EXPLICIT -> Sexual Content (SE)
        - THREAT -> Hate/Violence (H)

        Args:
            scores: Dictionary of attribute scores

        Returns:
            HarmLabel with classifications
        """
        label = HarmLabel()

        # Get max toxicity score across general toxicity attributes
        general_toxicity = max([
            scores.get("TOXICITY", 0.0),
            scores.get("SEVERE_TOXICITY", 0.0),
            scores.get("INSULT", 0.0),
            scores.get("PROFANITY", 0.0),
        ])

        # Classify Hate/Violence (H)
        if general_toxicity >= self.toxic_threshold:
            label.hate_violence = Dimension.TOXIC
        elif general_toxicity >= self.topical_threshold:
            label.hate_violence = Dimension.TOPICAL

        # Classify Ideological Harm (IH)
        identity_attack = scores.get("IDENTITY_ATTACK", 0.0)
        if identity_attack >= self.toxic_threshold:
            label.ideological = Dimension.TOXIC
        elif identity_attack >= self.topical_threshold:
            label.ideological = Dimension.TOPICAL

        # Classify Sexual Content (SE)
        sexually_explicit = scores.get("SEXUALLY_EXPLICIT", 0.0)
        if sexually_explicit >= self.toxic_threshold:
            label.sexual = Dimension.TOXIC
        elif sexually_explicit >= self.topical_threshold:
            label.sexual = Dimension.TOPICAL

        # Classify Hate/Violence (H) - THREAT treated as hate/violence
        threat = scores.get("THREAT", 0.0)
        if threat >= self.toxic_threshold:
            label.hate_violence = Dimension.TOXIC
        elif threat >= self.topical_threshold:
            label.hate_violence = Dimension.TOPICAL

        # Note: Perspective doesn't have good mapping for Self-Inflicted (SI)
        # We leave it as SAFE

        return label

    def predict(self, text: str) -> 'HarmLabel':
        """
        Predict harm label for text.

        Args:
            text: Text to classify

        Returns:
            HarmLabel with classifications
        """
        scores = self.get_scores(text)
        if scores is None:
            logger.warning("Failed to get scores, returning safe label")
            return HarmLabel()

        return self.scores_to_label(scores)

    def __call__(self, text: str) -> 'HarmLabel':
        """Make classifier callable."""
        return self.predict(text)
