# -*- coding: utf-8 -*-
"""LLM based analyzer utilities."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Literal

from anthropic import Anthropic
from openai import OpenAI

from app.prompts.analysis_prompt import DL_ANALYSIS_PROMPT

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


class LLMAnalyzer:
    """Analyze YOLO training runs with an LLM backend."""

    def __init__(self, provider: Literal["claude", "openai"] = "claude") -> None:
        self.provider = provider

        if provider == "claude":
            api_key_present = bool(os.getenv("CLAUDE_API_KEY"))
            logger.info(
                "Anthropic istemcisi oluşturuluyor (api_key_var=%s)",
                api_key_present,
            )
            self.client = Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            api_key_present = bool(api_key)
            logger.info(
                "OpenAI istemcisi oluşturuluyor (api_key_var=%s)",
                api_key_present,
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

                self.client = type(
                    "OpenAIStub",
                    (),
                    {"chat": chat_stub},
                )()

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
    def _parse_structured_output(raw_text: str) -> Dict[str, Any]:
        """Parse the LLM response into the standardized dictionary format."""

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

        structured: Dict[str, Any] = {
            "summary": str(payload.get("summary", "")).strip(),
            "strengths": LLMAnalyzer._normalise_list(payload.get("strengths")),
            "weaknesses": LLMAnalyzer._normalise_list(payload.get("weaknesses")),
            "action_items": payload.get("action_items") or [],
            "risk_level": str(payload.get("risk_level", "")).strip(),
            "notes": str(payload.get("notes", "")).strip(),
        }

        action_items = structured["action_items"]
        if not isinstance(action_items, list):
            action_items = [action_items]

        normalised_action_items: List[Dict[str, Any]] = []
        for item in action_items:
            if isinstance(item, dict):
                normalised_action_items.append(item)
                continue

            if isinstance(item, str):
                normalised_action_items.append({"description": item.strip()})
                continue

            normalised_action_items.append({"value": item})

        structured["action_items"] = normalised_action_items

        return structured

    def analyze(self, metrics: Dict, config: Dict) -> Dict:
        """Create the analysis prompt and dispatch it to the chosen provider."""

        prompt = self._build_prompt(metrics or {}, config or {})
        logger.debug(
            "LLM prompt hazırlandı (provider=%s, uzunluk=%s karakter)",
            self.provider,
            len(prompt),
        )

        if self.provider == "claude":
            return self._analyze_with_claude(prompt)
        return self._analyze_with_gpt(prompt)

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

    def _build_prompt(self, metrics: Dict, config: Dict) -> str:
        """Compose a domain informed prompt for the LLM.

        The prompt encodes the FKT leather seat dent detection case study so that
        the model produces actionable, context aware guidance instead of vague
        advice.  Every requirement from the incident review is folded into the
        instructions below.
        """
        metrics_json = json.dumps(metrics or {}, indent=2, ensure_ascii=False)
        config_json = json.dumps(config or {}, indent=2, ensure_ascii=False)

        recall_percent = self._format_percentage(metrics.get("recall")) if metrics else "N/A"
        precision_percent = (
            self._format_percentage(metrics.get("precision")) if metrics else "N/A"
        )

        return DL_ANALYSIS_PROMPT.format(
            metrics=metrics_json,
            config=config_json,
            recall=recall_percent,
            precision=precision_percent,
        ).strip()

    def _analyze_with_claude(self, prompt: str) -> Dict:
        system_instruction = (
            "You are an assistant that strictly replies with ONLY a valid JSON object. "
            "CRITICAL: Return PURE JSON ONLY - NO markdown, NO code blocks, NO ```json, NO explanatory text. "
            "Start your response with { and end with }. "
            "The JSON must contain exactly these keys: "
            '"summary", "strengths", "weaknesses", "action_items", "risk_level", and "notes". '
            'The "action_items" value must be a JSON array of objects. '
            "Ensure all strings are properly escaped, especially quotes and newlines."
        )

        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=4096,  # Increased for longer Turkish responses
                temperature=0,
                system=system_instruction,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
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

        logger.debug(
            "Claude yanıtı alındı (uzunluk=%s karakter)",
            len(raw_text),
        )
        return self._parse_structured_output(raw_text)

    def _analyze_with_gpt(self, prompt: str) -> Dict:
        system_instruction = (
            "You are an assistant that returns thorough analyses as JSON. "
            'Respond with an object containing "summary", "strengths", "weaknesses", '
            '"action_items", "risk_level", and "notes". Ensure "action_items" is an array of '
            "objects with actionable guidance."
        )

        try:
            response = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model="gpt-4o",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
            )
        except (TimeoutError, OpenAIConnectionError, OpenAIAPIError) as exc:  # pragma: no cover - network calls
            logger.exception("OpenAI isteği başarısız oldu")
            raise RuntimeError(f"OpenAI request failed: {exc}") from exc

        raw_text = ""
        if getattr(response, "choices", None):
            raw_text = response.choices[0].message.content or ""

        logger.debug(
            "OpenAI yanıtı alındı (uzunluk=%s karakter)",
            len(raw_text),
        )
        return self._parse_structured_output(raw_text)


__all__ = ["LLMAnalyzer"]
