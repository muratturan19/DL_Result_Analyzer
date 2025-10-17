# -*- coding: utf-8 -*-
"""LLM based analyzer utilities."""

from __future__ import annotations

import os
from textwrap import dedent
from typing import Dict, Literal

from anthropic import Anthropic
from openai import OpenAI


class LLMAnalyzer:
    """Analyze YOLO training runs with an LLM backend."""

    def __init__(self, provider: Literal["claude", "openai"] = "claude") -> None:
        self.provider = provider

        if provider == "claude":
            self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        else:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def analyze(self, metrics: Dict, config: Dict) -> Dict:
        """Create the analysis prompt and dispatch it to the chosen provider."""

        prompt = self._build_prompt(metrics or {}, config or {})

        if self.provider == "claude":
            return self._analyze_with_claude(prompt)
        return self._analyze_with_gpt(prompt)

    def _build_prompt(self, metrics: Dict, config: Dict) -> str:
        """Compose a domain informed prompt for the LLM.

        The prompt encodes the FKT leather seat dent detection case study so that
        the model produces actionable, context aware guidance instead of vague
        advice.  Every requirement from the incident review is folded into the
        instructions below.
        """

        precision = metrics.get("precision", "N/A")
        recall = metrics.get("recall", "N/A")
        map50 = metrics.get("map50", "N/A")
        map50_95 = metrics.get("map50_95", "N/A")
        loss = metrics.get("loss", "N/A")

        epochs = config.get("epochs", config.get("epoch", "N/A"))
        batch = config.get("batch", config.get("batch_size", "N/A"))
        lr0 = config.get("lr0", config.get("learning_rate", "N/A"))
        iou_train = config.get("iou", config.get("iou_threshold", "N/A"))
        conf_train = config.get("conf", config.get("conf_threshold", "N/A"))

        return dedent(
            f"""
            You are an elite YOLO troubleshooting expert helping the FKT leather seat dent
            detection project. Use the case study knowledge below to craft a thorough, structured
            analysis. Always ground your reasoning in the provided metrics and config and offer
            precise, reproducible remediation steps.

            ### Case Study Knowledge to Reference
            - Low recall with healthy precision usually signals that detections exist but are
              rejected by the evaluation IoU threshold. Explain why relaxing validation IoU
              (e.g., testing 0.3/0.4/0.5/0.6) can recover recall and how to run the sweep:
              ```python
              for iou in [0.3, 0.4, 0.5, 0.6]:
                  metrics = model.val(data="data.yaml", iou=iou, conf=0.5)
                  print(f"IoU={{iou}} → Recall={{metrics.recall:.3f}}").
              ```
              Clarify that Non-Maximum Suppression (training-time IoU/conf) is separate from
              validation IoU thresholds used for metrics, and specify when to tune each.
            - Provide an annotation quality checklist: inspect ~50 samples, verify polygon
              consistency, ensure no missing dents, and confirm that Roboflow augmentations are
              disabled because they distort labels.
            - Recommend code-based augmentations with explicit parameters (e.g., `degrees=15`,
              `translate=0.1`, `fliplr=0.5`, `hsv_h=0.015`, `hsv_s=0.7`, `hsv_v=0.4`,
              `mosaic=1.0`). Explain the expected benefit in terms of synthetic variation.
            - Describe the dataset growth plan: target 600-800 images, capture both wide and close
              shots per seat (double-shot protocol), diversify lighting, and propose a consistent
              naming convention such as `YYMMDD_HHMM_ModelA_###_{"GENEL"|"YAKIN"}.jpg`.
            - Present action items ordered by urgency with timelines (`BUGÜN`, `YARIN`, `BU HAFTA`,
              `2-3 HAFTA`) and quantify expected improvements (e.g., recall 47% → 70-75%, F1
              51% → 65-70%).

            ### Metrics
            - Precision: {precision}
            - Recall: {recall}
            - mAP@0.5: {map50}
            - mAP@0.5:0.95: {map50_95}
            - Loss: {loss}

            ### Training Config
            - Epochs: {epochs}
            - Batch Size: {batch}
            - Learning Rate: {lr0}
            - Training IoU (NMS): {iou_train}
            - Training Confidence (NMS): {conf_train}

            ### Response Format (strict)
            1. **ÖZET** – 2-3 sentences covering overall health and the key blocker.
            2. **GÜÇLÜ YÖNLER** – bullet list.
            3. **ZAYIF YÖNLER** – bullet list with root causes (mention IoU vs confidence when relevant).
            4. **AKSİYON ÖNERİLERİ** – numbered list sorted by urgency. Each item must include:
               - urgency tag (BUGÜN / YARIN / BU HAFTA / 2-3 HAFTA),
               - concrete instructions (commands, parameter values, inspection counts),
               - expected metric impact when possible.
            5. **RİSK SEVİYESİ** – one of `low`, `medium`, `high` with justification.
            6. **NOTLAR** – clarify the difference between confidence and IoU thresholds and how to
               set production vs validation values after improvements.

            Do not suggest collecting vague "more data" without counts, and never recommend
            Roboflow-side augmentations. Base every recommendation on the evidence above.
            """
        ).strip()

    def _analyze_with_claude(self, prompt: str) -> Dict:
        raise NotImplementedError("Claude analysis is not implemented yet.")

    def _analyze_with_gpt(self, prompt: str) -> Dict:
        raise NotImplementedError("OpenAI analysis is not implemented yet.")


__all__ = ["LLMAnalyzer"]
