# -*- coding: utf-8 -*-
"""LLM based analyzer utilities."""

from __future__ import annotations

import base64
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from anthropic import Anthropic
from openai import OpenAI

from app.prompts.analysis_prompt import DL_ANALYSIS_PROMPT
from app.prompts.qa_prompt import QA_PROMPT

try:  # Anthropics specific error classes are optional in the runtime.
    from anthropic import APIConnectionError as AnthropicConnectionError
    from anthropic import APIError as AnthropicAPIError
except (ImportError, AttributeError):  # pragma: no cover - fallback for old SDKs
    AnthropicConnectionError = RuntimeError  # type: ignore[assignment]
    AnthropicAPIError = RuntimeError  # type: ignore[assignment]

try:  # OpenAI specific error classes are optional in the runtime.
    from openai import APIConnectionError as OpenAIConnectionError
    from openai import APIError as OpenAIAPIError
except (ImportError, AttributeError):  # pragma: no cover - fallback for old SDKs
    OpenAIConnectionError = RuntimeError  # type: ignore[assignment]
    OpenAIAPIError = RuntimeError  # type: ignore[assignment]
logger = logging.getLogger(__name__)

GPT_ANALYSIS_JSON_SCHEMA: Dict[str, Any] = {
    "name": "DLRunAnalysis",
    "schema": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "summary": {"type": "string"},
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "weaknesses": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "actions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": [
                        "module",
                        "problem",
                        "evidence",
                        "recommendation",
                        "expected_gain",
                        "validation_plan",
                    ],
                    "properties": {
                        "module": {"type": "string"},
                        "problem": {"type": "string"},
                        "evidence": {"type": "string"},
                        "recommendation": {"type": "string"},
                        "expected_gain": {"type": "string"},
                        "validation_plan": {"type": "string"},
                    },
                },
                "default": [],
            },
            "risk": {"type": "string"},
            "deploy_profile": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "release_decision": {"type": "string"},
                    "rollout_strategy": {"type": "string"},
                    "monitoring_plan": {"type": "string"},
                    "owner": {"type": "string"},
                    "notes": {"type": "string"},
                },
            },
            "notes": {"type": "string"},
            "calibration": {
                "type": "object",
                "additionalProperties": True,
            },
            # Legacy compatibility keys
            "action_items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                },
                "default": [],
            },
            "risk_level": {"type": "string"},
        },
        "required": [
            "summary",
            "strengths",
            "weaknesses",
            "actions",
            "risk",
            "deploy_profile",
        ],
    },
}

GPT_QA_JSON_SCHEMA: Dict[str, Any] = {
    "name": "DLFollowUpQA",
    "schema": {
        "type": "object",
        "additionalProperties": True,
        "properties": {
            "answer": {"type": "string"},
            "references": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "follow_up_questions": {
                "type": "array",
                "items": {"type": "string"},
                "default": [],
            },
            "notes": {"type": "string"},
        },
        "required": ["answer", "references", "follow_up_questions"],
    },
}


class LLMAnalyzer:
    """Analyze YOLO training runs with an LLM backend."""

    def __init__(self, provider: Literal["claude", "openai"] = "claude") -> None:
        self.provider = provider
        self._openai_high_reasoning = False

        self._openai_response_format_supported: Optional[bool] = None

        if provider == "claude":
            api_key = os.getenv("CLAUDE_API_KEY")
            api_key_present = bool(api_key)
            logger.info(
                "Anthropic istemcisi oluşturuluyor (api_key_var=%s)",
                api_key_present,
            )
            if api_key_present:
                self.client = Anthropic(api_key=api_key)
            else:
                logger.warning("CLAUDE_API_KEY bulunamadı, stub istemci kullanılacak.")

                def _raise_missing_key(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - stub
                    raise RuntimeError("Claude API key is not configured.")

                messages_stub = type(
                    "MessagesStub",
                    (),
                    {"create": staticmethod(_raise_missing_key)},
                )()

                self.client = type(
                    "AnthropicStub",
                    (),
                    {"messages": messages_stub},
                )()
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            api_key_present = bool(api_key)
            logger.info(
                "OpenAI istemcisi oluşturuluyor (api_key_var=%s)",
                api_key_present,
            )
            self._openai_high_reasoning = (
                os.getenv("OPENAI_HIGH_REASONING", "").strip().lower()
                in {"1", "true", "yes", "on"}
            )
            if api_key_present:
                self.client = OpenAI(api_key=api_key)
            else:
                logger.warning("OPENAI_API_KEY bulunamadı, stub istemci kullanılacak.")

                def _raise_missing_key(*_args: Any, **_kwargs: Any) -> Any:  # pragma: no cover - stub
                    raise RuntimeError("OpenAI API key is not configured.")

                completions_stub = type(
                    "CompletionsStub",
                    (),
                    {"create": staticmethod(_raise_missing_key)},
                )()

                chat_stub = type(
                    "ChatStub",
                    (),
                    {"completions": completions_stub},
                )()

                responses_stub = type(
                    "ResponsesStub",
                    (),
                    {"create": staticmethod(_raise_missing_key)},
                )()

                self.client = type(
                    "OpenAIStub",
                    (),
                    {"chat": chat_stub, "responses": responses_stub},
                )()

    def _openai_supports_response_format(self) -> bool:
        """Return True when the OpenAI client supports response_format."""

        if self.provider != "openai":
            return False

        if self._openai_response_format_supported is not None:
            return self._openai_response_format_supported

        try:
            responses_client = getattr(self.client, "responses", None)
            create_fn = getattr(responses_client, "create", None)
            if create_fn is None:
                self._openai_response_format_supported = False
            else:
                signature = inspect.signature(create_fn)
                parameters = signature.parameters
                self._openai_response_format_supported = (
                    "response_format" in parameters
                    or any(
                        parameter.kind
                        == inspect.Parameter.VAR_KEYWORD
                        for parameter in parameters.values()
                    )
                )
        except (TypeError, ValueError):  # pragma: no cover - defensive against exotic clients
            self._openai_response_format_supported = False

        if not self._openai_response_format_supported:
            logger.debug(
                "OpenAI istemcisi response_format parametresini desteklemiyor; serbest metin moduna geçiliyor."
            )

        return self._openai_response_format_supported

    @staticmethod
    def _openai_supports_temperature(model_name: str) -> bool:
        """Return True when the given OpenAI model supports temperature."""

        unsupported_prefixes = ("gpt-5",)
        if any(model_name.startswith(prefix) for prefix in unsupported_prefixes):
            logger.debug(
                "OpenAI modeli %s temperature parametresini desteklemiyor; parametre atlanacak.",
                model_name,
            )
            return False
        return True

    @staticmethod
    def _try_fix_truncated_json(json_str: str) -> Dict[str, Any] | None:
        """Attempt to fix common JSON truncation issues.

        This handles cases where the response was cut off mid-string or mid-object.
        Returns the parsed dict if successful, None otherwise.
        """
        if not json_str or not json_str.strip():
            return None

        json_str = json_str.strip()

        # Check if it starts with { and doesn't end with }
        if not json_str.startswith('{'):
            return None

        # Try to find where the JSON got cut off
        try:
            # Count braces to see if we need to close
            open_braces = json_str.count('{')
            close_braces = json_str.count('}')

            if open_braces > close_braces:
                # Check if we're in the middle of a string (unterminated string)
                # Count quotes but be careful about escaped quotes
                quote_count = 0
                escaped = False
                for char in json_str:
                    if escaped:
                        escaped = False
                        continue
                    if char == '\\':
                        escaped = True
                        continue
                    if char == '"':
                        quote_count += 1

                # If odd number of quotes, we have an unterminated string
                fixed = json_str
                if quote_count % 2 == 1:
                    # Close the string
                    fixed += '"'
                    logger.debug("Added missing closing quote")

                # Close any open arrays
                open_brackets = fixed.count('[')
                close_brackets = fixed.count(']')
                if open_brackets > close_brackets:
                    fixed += ']' * (open_brackets - close_brackets)
                    logger.debug("Added %d missing closing bracket(s)", open_brackets - close_brackets)

                # Close any open objects
                open_braces = fixed.count('{')
                close_braces = fixed.count('}')
                if open_braces > close_braces:
                    fixed += '}' * (open_braces - close_braces)
                    logger.debug("Added %d missing closing brace(s)", open_braces - close_braces)

                # Try to parse
                try:
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    # If still failing, try removing incomplete last item
                    # Find the last complete field and truncate there
                    logger.debug("Still failed after fixing braces/brackets, trying to remove incomplete field")

                    # Look for common patterns like: "field": "incomplete value
                    # We'll try to find the last comma before the incomplete part
                    last_good_comma = -1
                    depth = 0
                    in_string = False
                    escaped = False

                    for i, char in enumerate(fixed):
                        if escaped:
                            escaped = False
                            continue
                        if char == '\\':
                            escaped = True
                            continue
                        if char == '"':
                            in_string = not in_string
                            continue
                        if not in_string:
                            if char in '{[':
                                depth += 1
                            elif char in '}]':
                                depth -= 1
                            elif char == ',' and depth == 1:
                                last_good_comma = i

                    if last_good_comma > 0:
                        # Truncate after last good comma and close
                        truncated = fixed[:last_good_comma]
                        open_braces = truncated.count('{')
                        close_braces = truncated.count('}')
                        truncated += '}' * (open_braces - close_braces)

                        try:
                            return json.loads(truncated)
                        except json.JSONDecodeError:
                            pass

        except Exception as e:
            logger.debug("Error while trying to fix truncated JSON: %s", str(e))

        return None

    @staticmethod
    def _normalise_list(value: Any) -> List[str]:
        """Ensure list-like fields are always returned as a list of strings."""

        if value is None:
            return []

        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]

        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                return []
            # Split on newlines or semicolons if present to handle compact formats.
            if "\n" in cleaned:
                return [segment.strip() for segment in cleaned.splitlines() if segment.strip()]
            if ";" in cleaned:
                return [segment.strip() for segment in cleaned.split(";") if segment.strip()]
            return [cleaned]

        return [str(value).strip()]

    @staticmethod
    def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
        """Extract a JSON object from an LLM response."""
        if not raw_text:
            raise ValueError("LLM response payload is empty.")

        raw_text = raw_text.strip()

        # Strategy 1: Try to extract from markdown code blocks (even though we asked not to use them)
        code_block_match = re.search(
            r"```(?:json)?\s*(.+?)\s*```",
            raw_text,
            flags=re.DOTALL
        )
        if code_block_match:
            json_candidate = code_block_match.group(1).strip()
            try:
                payload = json.loads(json_candidate)
                logger.debug("Successfully parsed JSON from markdown code block")
            except json.JSONDecodeError as e:
                logger.warning(
                    "Found markdown code block but JSON parsing failed: %s",
                    str(e)
                )
                # Try to fix truncated JSON
                payload = LLMAnalyzer._try_fix_truncated_json(json_candidate)
        else:
            payload = None

        # Strategy 2: Try parsing the entire response as JSON
        if payload is None:
            try:
                payload = json.loads(raw_text)
                logger.debug("Successfully parsed entire response as JSON")
            except json.JSONDecodeError as e:
                logger.debug("Failed to parse entire response as JSON: %s", str(e))
                payload = None

        # Strategy 3: Extract JSON by counting braces
        if payload is None:
            first_brace = raw_text.find('{')
            if first_brace == -1:
                logger.error("No JSON object found in response: %s", raw_text[:200])
                raise ValueError("Unable to locate JSON object in LLM response.")

            brace_count = 0
            in_string = False
            escape_next = False

            for i in range(first_brace, len(raw_text)):
                char = raw_text[i]

                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"':
                    in_string = not in_string
                    continue

                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_candidate = raw_text[first_brace:i+1]
                            try:
                                payload = json.loads(json_candidate)
                                logger.debug("Successfully extracted and parsed JSON by brace counting")
                                break
                            except json.JSONDecodeError as e:
                                logger.warning(
                                    "Extracted JSON by brace counting but parsing failed: %s. Trying to fix...",
                                    str(e)
                                )
                                payload = LLMAnalyzer._try_fix_truncated_json(json_candidate)
                                if payload:
                                    logger.info("Successfully fixed truncated JSON")
                                    break

        if payload is None:
            logger.error("All parsing strategies failed. Response preview: %s", raw_text[:500])
            raise ValueError("Unable to parse JSON from LLM response after trying all strategies.")

        return payload

    @staticmethod
    def _parse_structured_output(raw_text: str) -> Dict[str, Any]:
        """Parse the LLM response into the standardized dictionary format."""

        payload = LLMAnalyzer._extract_json_payload(raw_text)
        structured: Dict[str, Any] = {
            "summary": str(payload.get("summary", "")).strip(),
            "strengths": LLMAnalyzer._normalise_list(payload.get("strengths")),
            "weaknesses": LLMAnalyzer._normalise_list(payload.get("weaknesses")),
        }

        raw_actions: Any = payload.get("actions")
        legacy_actions: Any = None
        if raw_actions is None and "action_items" in payload:
            legacy_actions = payload.get("action_items")
            raw_actions = legacy_actions

        if raw_actions is None:
            normalised_actions: List[Dict[str, Any]] = []
        elif isinstance(raw_actions, list):
            normalised_actions = []
            for item in raw_actions:
                if isinstance(item, dict):
                    normalised_actions.append(item)
                elif isinstance(item, str):
                    normalised_actions.append({"recommendation": item.strip()})
                else:
                    normalised_actions.append({"value": item})
        elif isinstance(raw_actions, dict):
            normalised_actions = [raw_actions]
        else:
            normalised_actions = [{"value": raw_actions}]

        structured["actions"] = normalised_actions
        if legacy_actions is not None and "action_items" not in structured:
            structured["action_items"] = normalised_actions

        risk_value = payload.get("risk") or payload.get("risk_level") or ""
        structured["risk"] = str(risk_value).strip()
        if "risk_level" in payload and "risk_level" not in structured:
            structured["risk_level"] = str(payload.get("risk_level", "")).strip()

        notes_value = payload.get("notes")
        structured["notes"] = str(notes_value).strip() if notes_value is not None else ""

        deploy_profile = payload.get("deploy_profile")
        if isinstance(deploy_profile, dict):
            structured["deploy_profile"] = deploy_profile
        else:
            structured["deploy_profile"] = {}

        if "calibration" in payload:
            structured["calibration"] = payload.get("calibration")

        extra_fields = {
            key: value
            for key, value in payload.items()
            if key not in structured
        }
        structured.update(extra_fields)

        return structured

    @staticmethod
    def _parse_qa_output(raw_text: str) -> Dict[str, Any]:
        """Parse an LLM question-answer response into a normalized structure."""

        payload = LLMAnalyzer._extract_json_payload(raw_text)

        answer = str(payload.get("answer", "")).strip()
        references = LLMAnalyzer._normalise_list(payload.get("references"))
        follow_up = LLMAnalyzer._normalise_list(payload.get("follow_up_questions"))

        notes_value = payload.get("notes")
        notes = str(notes_value).strip() if notes_value is not None else ""

        response: Dict[str, Any] = {
            "answer": answer,
            "references": references,
            "follow_up_questions": follow_up,
        }

        if notes:
            response["notes"] = notes

        known_keys = {"answer", "references", "follow_up_questions", "notes"}
        extras = {
            key: value
            for key, value in payload.items()
            if key not in known_keys
        }
        if extras:
            response["extra"] = extras

        return response

    def analyze(
        self,
        metrics: Dict,
        config: Dict,
        project_context: Optional[Dict] = None,
        training_code: Optional[Dict[str, str]] = None,
        history: Optional[Dict] = None,
        artefacts: Optional[Dict] = None,
        graph_images: Optional[List[Path]] = None,
    ) -> Dict:
        """Create the analysis prompt and dispatch it to the chosen provider."""

        prompt = self._build_prompt(
            metrics or {},
            config or {},
            project_context or {},
            training_code or {},
            history or {},
            artefacts or {},
        )
        logger.debug(
            "LLM prompt hazırlandı (provider=%s, uzunluk=%s karakter)",
            self.provider,
            len(prompt),
        )

        if self.provider == "claude":
            return self._analyze_with_claude(prompt, graph_images=graph_images or [])
        return self._analyze_with_gpt(prompt, graph_images=graph_images or [])

    @staticmethod
    def _format_percentage(value: Any) -> str:
        """Convert ratio or percentage like values into a percentage string."""

        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return str(value)

        if numeric <= 1:
            numeric *= 100

        return f"{numeric:.2f}"

    def _build_prompt(
        self,
        metrics: Dict,
        config: Dict,
        project_context: Optional[Dict] = None,
        training_code: Optional[Dict[str, str]] = None,
        history: Optional[Dict] = None,
        artefacts: Optional[Dict] = None,
    ) -> str:
        """Compose a domain informed prompt for the LLM.

        The prompt encodes the FKT leather seat dent detection case study so that
        the model produces actionable, context aware guidance instead of vague
        advice.  Every requirement from the incident review is folded into the
        instructions below.
        """
        metrics_json = json.dumps(metrics or {}, indent=2, ensure_ascii=False)
        config_json = json.dumps(config or {}, indent=2, ensure_ascii=False)

        dataset_section = "Belirtilmedi"
        if isinstance(config, dict):
            dataset_info = config.get("dataset")
            if dataset_info:
                try:
                    dataset_section = json.dumps(dataset_info, indent=2, ensure_ascii=False)
                except TypeError:
                    dataset_section = str(dataset_info)

        recall_percent = self._format_percentage(metrics.get("recall")) if metrics else "N/A"
        precision_percent = (
            self._format_percentage(metrics.get("precision")) if metrics else "N/A"
        )

        map50_percent = self._format_percentage(metrics.get("map50")) if metrics else "N/A"

        f1_score: Optional[float] = None
        try:
            precision_value = float(metrics.get("precision", 0))
            recall_value = float(metrics.get("recall", 0))
            if precision_value + recall_value > 0:
                f1_score = (2 * precision_value * recall_value) / (precision_value + recall_value)
        except (TypeError, ValueError, ZeroDivisionError):
            f1_score = None

        f1_percent = self._format_percentage(f1_score) if f1_score is not None else "N/A"

        project_json = json.dumps(project_context or {}, indent=2, ensure_ascii=False) if project_context else "Belirtilmedi"

        training_code_filename = training_code.get("filename") if training_code else None
        training_code_excerpt = training_code.get("excerpt") if training_code else None
        if training_code_filename or training_code_excerpt:
            training_code_text = (
                f"Dosya: {training_code_filename or 'Belirtilmedi'}\n\n{training_code_excerpt or ''}"
            ).strip()
            if not training_code_text:
                training_code_text = f"Dosya: {training_code_filename or 'Belirtilmedi'}"
        else:
            training_code_text = "Kod paylaşılmadı."

        history_json = json.dumps(history or {}, indent=2, ensure_ascii=False)
        artefacts_json = json.dumps(artefacts or {}, indent=2, ensure_ascii=False)

        return DL_ANALYSIS_PROMPT.format(
            metrics=metrics_json,
            config=config_json,
            recall=recall_percent,
            precision=precision_percent,
            map50=map50_percent,
            f1=f1_percent,
            project_context=project_json,
            training_code=training_code_text,
            history=history_json,
            artefacts=artefacts_json,
            dataset=dataset_section,
        ).strip()

    def _build_qa_prompt(self, question: str, context: Dict[str, Any]) -> str:
        metrics_json = json.dumps(context.get("metrics") or {}, indent=2, ensure_ascii=False)
        config_json = json.dumps(context.get("config") or {}, indent=2, ensure_ascii=False)
        history_json = json.dumps(context.get("history") or {}, indent=2, ensure_ascii=False)
        artefacts_json = json.dumps(context.get("artefacts") or {}, indent=2, ensure_ascii=False)

        config_section = context.get("config") or {}
        dataset_section = "Belirtilmedi"
        if isinstance(config_section, dict):
            dataset_info = config_section.get("dataset")
            if dataset_info:
                try:
                    dataset_section = json.dumps(dataset_info, indent=2, ensure_ascii=False)
                except TypeError:
                    dataset_section = str(dataset_info)

        training_code = context.get("training_code") or {}
        training_code_text = "Kod paylaşılmadı."
        if isinstance(training_code, dict) and (training_code.get("filename") or training_code.get("excerpt")):
            filename = training_code.get("filename") or "Belirtilmedi"
            excerpt = training_code.get("excerpt", "").strip()
            training_code_text = f"Dosya: {filename}"
            if excerpt:
                training_code_text = f"{training_code_text}\n\n{excerpt}"

        analysis = context.get("analysis") or {}
        summary_text = str(analysis.get("summary", "Belirtilmedi"))
        strengths_text = json.dumps(analysis.get("strengths") or [], indent=2, ensure_ascii=False)
        weaknesses_text = json.dumps(analysis.get("weaknesses") or [], indent=2, ensure_ascii=False)
        risk_text = json.dumps(analysis.get("risk") or "Belirtilmedi", ensure_ascii=False)
        deploy_profile_text = json.dumps(analysis.get("deploy_profile") or {}, indent=2, ensure_ascii=False)
        actions_text = json.dumps(analysis.get("actions") or [], indent=2, ensure_ascii=False)

        question_text = question.strip() or "Soru belirtilmedi."

        return QA_PROMPT.format(
            summary=summary_text,
            strengths=strengths_text,
            weaknesses=weaknesses_text,
            risk=risk_text,
            deploy_profile=deploy_profile_text,
            actions=actions_text,
            metrics=metrics_json,
            config=config_json,
            dataset=dataset_section,
            history=history_json,
            training_code=training_code_text,
            artefacts=artefacts_json,
            question=question_text,
        ).strip()

    def answer_question(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_qa_prompt(question, context or {})
        logger.debug(
            "LLM QA prompt hazırlandı (provider=%s, uzunluk=%s karakter)",
            self.provider,
            len(prompt),
        )

        try:
            if self.provider == "claude":
                return self._qa_with_claude(prompt)
            return self._qa_with_gpt(prompt)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning("LLM QA isteği başarısız oldu: %s", exc)
            return self._fallback_answer(question, context, error=str(exc))

    @staticmethod
    def _encode_image_to_base64(image_path: Path) -> tuple[str, str]:
        """Encode an image file to base64 and detect its media type."""
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Detect media type based on file extension
        extension = image_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        media_type = media_type_map.get(extension, "image/png")

        # Read and encode the image
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()

        encoded = base64.standard_b64encode(image_data).decode("utf-8")
        return encoded, media_type

    def _analyze_with_claude(self, prompt: str, graph_images: Optional[List[Path]] = None) -> Dict:
        system_instruction = (
            "You are an assistant that strictly replies with ONLY a valid JSON object. "
            "CRITICAL: Return PURE JSON ONLY - NO markdown, NO code blocks, NO ```json, NO explanatory text. "
            "Start your response with { and end with }. "
            "The JSON must contain these keys: "
            '"summary", "strengths", "weaknesses", "actions", "risk", "deploy_profile", and "notes". '
            'Include "calibration" only when you have calibration artefacts to share. '
            'The "actions" value must be an array of objects with the keys '
            '"module", "problem", "evidence", "recommendation", "expected_gain", and "validation_plan". '
            "Ensure all strings are properly escaped, especially quotes and newlines. "
            "All textual content must be written entirely in fluent Turkish; avoid English wording "
            "except for metric names or hyperparameters."
        )

        # Build the content with text and images
        content_blocks: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Add images if provided
        if graph_images:
            for image_path in graph_images:
                try:
                    encoded_data, media_type = self._encode_image_to_base64(image_path)
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded_data,
                        },
                    })
                    logger.debug("Grafik eklendi: %s", image_path.name)
                except Exception as exc:
                    logger.warning("Grafik eklenemedi (%s): %s", image_path.name, exc)

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=8192,  # Increased for detailed graph analysis
                temperature=0,
                system=system_instruction,
                messages=[
                    {
                        "role": "user",
                        "content": content_blocks,
                    }
                ],
            )
        except (TimeoutError, AnthropicConnectionError, AnthropicAPIError) as exc:  # pragma: no cover - network calls
            logger.exception("Claude isteği başarısız oldu")
            raise RuntimeError(f"Claude request failed: {exc}") from exc

        text_chunks = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                text_chunks.append(getattr(block, "text", ""))

        raw_text = "".join(text_chunks).strip()
        if not raw_text and hasattr(response, "model_response"):  # pragma: no cover - compatibility path
            raw_text = str(getattr(response, "model_response", "")).strip()

        # Extract token usage information
        usage_info = {}
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                usage_info["input_tokens"] = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                usage_info["output_tokens"] = usage.output_tokens
            if "input_tokens" in usage_info and "output_tokens" in usage_info:
                usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]

        logger.debug(
            "Claude yanıtı alındı (uzunluk=%s karakter, tokens=%s)",
            len(raw_text),
            usage_info.get("total_tokens", "N/A"),
        )

        result = self._parse_structured_output(raw_text)
        if usage_info:
            result["token_usage"] = usage_info
        return result

    def _analyze_with_gpt(self, prompt: str, graph_images: Optional[List[Path]] = None) -> Dict:
        system_instruction = (
            "You are an assistant that returns thorough analyses as JSON. "
            'Respond with an object containing "summary", "strengths", "weaknesses", '
            '"actions", "risk", "deploy_profile", and "notes". Provide "calibration" only when '
            "you have artefacts to share. Ensure \"actions\" is an array of objects containing the "
            "keys module, problem, evidence, recommendation, expected_gain, and validation_plan. "
            "All textual content must be produced entirely in natural Turkish; avoid English "
            "sentences except for metric names or hyperparameters."
        )

        model_name = "gpt-5-thinking" if self._openai_high_reasoning else "gpt-5"
        reasoning_effort = "high" if self._openai_high_reasoning else "medium"

        # Build user content with text and images
        user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]

        # Add images if provided (OpenAI uses data URLs)
        if graph_images:
            for image_path in graph_images:
                try:
                    encoded_data, media_type = self._encode_image_to_base64(image_path)
                    data_url = f"data:{media_type};base64,{encoded_data}"
                    user_content.append({
                        "type": "input_image",
                        "image_url": data_url,
                    })
                    logger.debug("Grafik eklendi (GPT): %s", image_path.name)
                except Exception as exc:
                    logger.warning("Grafik eklenemedi (%s): %s", image_path.name, exc)

        try:
            request_kwargs: Dict[str, Any] = {
                "model": model_name,
                "reasoning": {"effort": reasoning_effort},
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": system_instruction},
                        ],
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
            }

            if self._openai_supports_temperature(model_name):
                request_kwargs["temperature"] = 0

            if self._openai_supports_response_format():
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": GPT_ANALYSIS_JSON_SCHEMA,
                }

            response = self.client.responses.create(  # type: ignore[attr-defined]
                **request_kwargs
            )
        except (TimeoutError, OpenAIConnectionError, OpenAIAPIError) as exc:  # pragma: no cover - network calls
            logger.exception("OpenAI isteği başarısız oldu")
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        raw_text = ""
        output_blocks = getattr(response, "output", None) or []
        if output_blocks:
            collected_chunks: List[str] = []
            for output_block in output_blocks:
                content_blocks = getattr(output_block, "content", None) or []
                for content in content_blocks:
                    content_type = getattr(content, "type", "")
                    if content_type in {"output_text", "text", ""}:
                        text_value = getattr(content, "text", "") or ""
                        if text_value:
                            collected_chunks.append(text_value)
            raw_text = "".join(collected_chunks).strip()

        if not raw_text and hasattr(response, "output_text"):
            raw_text = getattr(response, "output_text", "") or ""

        # Extract token usage information
        usage_info = {}
        if hasattr(response, "usage"):
            usage = response.usage
            # OpenAI typically uses prompt_tokens, completion_tokens, total_tokens
            if hasattr(usage, "prompt_tokens"):
                usage_info["input_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                usage_info["output_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                usage_info["total_tokens"] = usage.total_tokens
            elif "input_tokens" in usage_info and "output_tokens" in usage_info:
                usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]

        logger.debug(
            "OpenAI yanıtı alındı (uzunluk=%s karakter, tokens=%s)",
            len(raw_text),
            usage_info.get("total_tokens", "N/A"),
        )

        result = self._parse_structured_output(raw_text)
        if usage_info:
            result["token_usage"] = usage_info
        return result

    def _qa_with_claude(self, prompt: str) -> Dict[str, Any]:
        system_instruction = (
            "You are an assistant that strictly replies with ONLY a valid JSON object. "
            "Return keys answer (string), references (array of strings), follow_up_questions (array of strings), "
            "and optional notes. All text must be Turkish."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                temperature=0,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}],
            )
        except (TimeoutError, AnthropicConnectionError, AnthropicAPIError) as exc:  # pragma: no cover - network calls
            logger.exception("Claude QA isteği başarısız oldu")
            raise RuntimeError(f"Claude QA request failed: {exc}") from exc

        text_chunks = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", "") == "text":
                text_chunks.append(getattr(block, "text", ""))

        raw_text = "".join(text_chunks).strip()
        if not raw_text and hasattr(response, "model_response"):
            raw_text = str(getattr(response, "model_response", "")).strip()

        # Extract token usage information
        usage_info = {}
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "input_tokens"):
                usage_info["input_tokens"] = usage.input_tokens
            if hasattr(usage, "output_tokens"):
                usage_info["output_tokens"] = usage.output_tokens
            if "input_tokens" in usage_info and "output_tokens" in usage_info:
                usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]

        logger.debug(
            "Claude QA yanıtı alındı (uzunluk=%s karakter, tokens=%s)",
            len(raw_text),
            usage_info.get("total_tokens", "N/A"),
        )

        result = self._parse_qa_output(raw_text)
        if usage_info:
            result["token_usage"] = usage_info
        return result

    def _qa_with_gpt(self, prompt: str) -> Dict[str, Any]:
        system_instruction = (
            "You are an assistant that answers follow-up questions using JSON only. "
            "The JSON must contain answer, references, follow_up_questions, and optional notes. "
            "Every sentence must be in Turkish."
        )

        model_name = "gpt-5-thinking" if self._openai_high_reasoning else "gpt-5"
        reasoning_effort = "high" if self._openai_high_reasoning else "medium"

        try:
            request_kwargs: Dict[str, Any] = {
                "model": model_name,
                "reasoning": {"effort": reasoning_effort},
                "input": [
                    {
                        "role": "system",
                        "content": [
                            {"type": "input_text", "text": system_instruction},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                        ],
                    },
                ],
            }

            if self._openai_supports_temperature(model_name):
                request_kwargs["temperature"] = 0

            if self._openai_supports_response_format():
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": GPT_QA_JSON_SCHEMA,
                }

            response = self.client.responses.create(  # type: ignore[attr-defined]
                **request_kwargs
            )
        except (TimeoutError, OpenAIConnectionError, OpenAIAPIError) as exc:  # pragma: no cover - network calls
            logger.exception("OpenAI QA isteği başarısız oldu")
            raise RuntimeError(f"OpenAI QA request failed: {exc}") from exc

        raw_text = ""
        output_blocks = getattr(response, "output", None) or []
        if output_blocks:
            collected_chunks: List[str] = []
            for output_block in output_blocks:
                content_blocks = getattr(output_block, "content", None) or []
                for content in content_blocks:
                    content_type = getattr(content, "type", "")
                    if content_type in {"output_text", "text", ""}:
                        text_value = getattr(content, "text", "") or ""
                        if text_value:
                            collected_chunks.append(text_value)
            raw_text = "".join(collected_chunks).strip()

        if not raw_text and hasattr(response, "output_text"):
            raw_text = getattr(response, "output_text", "") or ""

        # Extract token usage information
        usage_info = {}
        if hasattr(response, "usage"):
            usage = response.usage
            if hasattr(usage, "prompt_tokens"):
                usage_info["input_tokens"] = usage.prompt_tokens
            if hasattr(usage, "completion_tokens"):
                usage_info["output_tokens"] = usage.completion_tokens
            if hasattr(usage, "total_tokens"):
                usage_info["total_tokens"] = usage.total_tokens
            elif "input_tokens" in usage_info and "output_tokens" in usage_info:
                usage_info["total_tokens"] = usage_info["input_tokens"] + usage_info["output_tokens"]

        logger.debug(
            "OpenAI QA yanıtı alındı (uzunluk=%s karakter, tokens=%s)",
            len(raw_text),
            usage_info.get("total_tokens", "N/A"),
        )

        result = self._parse_qa_output(raw_text)
        if usage_info:
            result["token_usage"] = usage_info
        return result

    def _fallback_answer(
        self,
        question: str,
        context: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        metrics = context.get("metrics") or {}
        config = context.get("config") or {}
        analysis = context.get("analysis") or {}
        dataset = {}
        if isinstance(config, dict):
            dataset = config.get("dataset") or {}

        lines: List[str] = []
        summary = analysis.get("summary") if isinstance(analysis, dict) else None
        if summary:
            lines.append(f"Önceki özet: {summary}")

        metric_pairs = [
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("map50", "mAP@0.5"),
            ("map50_95", "mAP@0.5:0.95"),
        ]
        metric_highlights: List[str] = []
        for key, label in metric_pairs:
            value = metrics.get(key)
            try:
                if value is None:
                    continue
                metric_highlights.append(f"{label}: {self._format_percentage(value)}%")
            except Exception:  # pragma: no cover - defensive
                continue
        if metric_highlights:
            lines.append("Metrikler → " + ", ".join(metric_highlights))

        dataset_counts: List[str] = []
        for key, label in [
            ("train_images", "Eğitim"),
            ("val_images", "Doğrulama"),
            ("test_images", "Test"),
            ("total_images", "Toplam"),
        ]:
            value = dataset.get(key)
            numeric_value: Optional[int] = None
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                try:
                    numeric_value = int(value)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    numeric_value = None
            elif isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    try:
                        numeric_value = int(float(stripped))
                    except ValueError:
                        numeric_value = None
            if numeric_value is not None:
                dataset_counts.append(f"{label}: {numeric_value} görsel")
        if dataset_counts:
            lines.append("Veri seti → " + ", ".join(dataset_counts))

        class_count = dataset.get("class_count") if isinstance(dataset, dict) else None
        if class_count:
            lines.append(f"Sınıf sayısı: {class_count}")

        if not lines:
            lines.append("LLM yanıtı üretilemedi; temel özet bilgileri sınırlı.")

        question_text = question.strip()
        if question_text:
            lines.append(f"Soru özetiniz: {question_text}")

        if error:
            lines.append(f"Not: LLM entegrasyonu hata verdi ({error}).")

        references: List[str] = []
        if metrics:
            references.append("results.csv → metrik özeti")
        if dataset_counts:
            references.append("args.yaml → veri seti görsel adetleri")

        follow_up: List[str] = []
        if error:
            follow_up.append("LLM API anahtarını tanımlayıp tekrar deneyin.")
        if not dataset_counts:
            follow_up.append("Veri seti görsel sayılarını detaylandırın.")

        notes = "Bu yanıt LLM erişimi olmadığı için kural tabanlı olarak oluşturuldu."

        return {
            "answer": "\n".join(lines),
            "references": references or ["results.csv → metrik özeti"],
            "follow_up_questions": follow_up or ["LLM erişimi sağlandığında soruyu yeniden iletin."],
            "notes": notes,
        }


__all__ = ["LLMAnalyzer"]
