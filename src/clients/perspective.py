"""
Perspective Comment Analyzer client adapter.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

from ..utils.taxonomy import HarmLabel, Dimension

logger = logging.getLogger(__name__)

try:
    from googleapiclient import discovery  # type: ignore
    from googleapiclient.errors import HttpError  # type: ignore

    PERSPECTIVE_AVAILABLE = True
except ImportError:
    PERSPECTIVE_AVAILABLE = False


class PerspectiveClient:
    """
    Adapter for Google's Perspective API.

    Maps Perspective toxicity scores to our taxonomy using threshold-based classification.
    """

    ATTRIBUTES = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT",
        "SEXUALLY_EXPLICIT",
    ]

    TOXIC_THRESHOLD = 0.7
    TOPICAL_THRESHOLD = 0.3

    def __init__(
        self,
        api_key: str,
        toxic_threshold: float = 0.7,
        topical_threshold: float = 0.3,
        languages: Optional[list] = None,
    ):
        if not PERSPECTIVE_AVAILABLE:
            raise RuntimeError("Perspective API requires: pip install google-api-python-client")

        self.toxic_threshold = toxic_threshold
        self.topical_threshold = topical_threshold
        self.languages = languages if languages is not None else ["en"]

        self.api_key = api_key
        self.client = None

        # Prefer googleapiclient when it works, but fall back to direct REST calls.
        # Some key configurations block access to the discovery document endpoint even
        # when the actual `comments:analyze` method works.
        try:
            self.client = discovery.build(  # type: ignore
                "commentanalyzer",
                "v1alpha1",
                developerKey=api_key,
                static_discovery=True,
            )
            logger.info("Initialized Perspective API client (googleapiclient)")
        except Exception as e:
            logger.warning(
                "Failed to initialize Perspective API client via googleapiclient; "
                "falling back to direct REST. Error: %s",
                e,
            )

    def _analyze_via_rest(self, analyze_request: dict) -> Optional[dict]:
        url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={self.api_key}"
        data = json.dumps(analyze_request).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body)
        except urllib.error.HTTPError as e:
            try:
                _ = e.read()
            except Exception:
                pass
            logger.error("Perspective REST HTTPError: %s", e.code)
            return None
        except Exception as e:
            logger.error("Perspective REST unexpected error: %s", e)
            return None

    def get_scores(self, text: str, max_retries: int = 3) -> Optional[dict]:
        text = text.strip()
        if not text:
            logger.warning("Empty text provided to Perspective API")
            return None
        if len(text) > 20000:
            logger.warning("Text too long (%s chars), truncating to 20000", len(text))
            text = text[:20000]

        analyze_request = {
            "comment": {"text": text},
            "requestedAttributes": {attr: {} for attr in self.ATTRIBUTES},
            "languages": self.languages,
        }

        for attempt in range(max_retries):
            try:
                if self.client is not None:
                    response = self.client.comments().analyze(body=analyze_request).execute()
                else:
                    response = self._analyze_via_rest(analyze_request)
                    if response is None:
                        return None

                scores = {}
                for attr in self.ATTRIBUTES:
                    if attr in response["attributeScores"]:
                        score = response["attributeScores"][attr]["summaryScore"]["value"]
                        scores[attr] = score
                return scores
            except HttpError as e:  # type: ignore
                if e.resp.status == 429 or (500 <= e.resp.status < 600):
                    wait_time = 2**attempt
                    logger.warning(
                        "Rate limited or server error (%s), waiting %ss...",
                        e.resp.status,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    continue
                logger.error("Perspective API error: %s", e)
                return None
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                return None

        logger.error("Failed after %s retries", max_retries)
        return None

    def scores_to_label(self, scores: dict) -> HarmLabel:
        label = HarmLabel()

        general_toxicity = max(
            [
                scores.get("TOXICITY", 0.0),
                scores.get("SEVERE_TOXICITY", 0.0),
                scores.get("INSULT", 0.0),
                scores.get("PROFANITY", 0.0),
            ]
        )

        if general_toxicity >= self.toxic_threshold:
            label.hate_violence = Dimension.TOXIC
        elif general_toxicity >= self.topical_threshold:
            label.hate_violence = Dimension.TOPICAL

        identity_attack = scores.get("IDENTITY_ATTACK", 0.0)
        if identity_attack >= self.toxic_threshold:
            label.ideological = Dimension.TOXIC
        elif identity_attack >= self.topical_threshold:
            label.ideological = Dimension.TOPICAL

        sexually_explicit = scores.get("SEXUALLY_EXPLICIT", 0.0)
        if sexually_explicit >= self.toxic_threshold:
            label.sexual = Dimension.TOXIC
        elif sexually_explicit >= self.topical_threshold:
            label.sexual = Dimension.TOPICAL

        threat = scores.get("THREAT", 0.0)
        threat_severity = Dimension.SAFE
        if threat >= self.toxic_threshold:
            threat_severity = Dimension.TOXIC
        elif threat >= self.topical_threshold:
            threat_severity = Dimension.TOPICAL

        severity_order = {Dimension.SAFE: 0, Dimension.TOPICAL: 1, Dimension.TOXIC: 2}
        if severity_order[threat_severity] > severity_order[label.hate_violence]:
            label.hate_violence = threat_severity

        return label

    def predict(self, text: str) -> HarmLabel:
        scores = self.get_scores(text)
        if scores is None:
            logger.warning("Failed to get scores, returning safe label")
            return HarmLabel()
        return self.scores_to_label(scores)

    def __call__(self, text: str) -> HarmLabel:
        return self.predict(text)


# Backwards compatibility
PerspectiveAPI = PerspectiveClient

