"""Tests for LLM analyzer."""
import json

import pytest

from app.analyzers.llm_analyzer import LLMAnalyzer


class TestLLMAnalyzer:
    """Test suite for LLMAnalyzer."""

    def test_analyzer_initialization_claude(self):
        """Test LLM analyzer initialization with Claude provider."""
        analyzer = LLMAnalyzer(provider="claude")
        assert analyzer.provider == "claude"
        assert analyzer.client is not None

    def test_analyzer_initialization_openai(self):
        """Test LLM analyzer initialization with OpenAI provider."""
        analyzer = LLMAnalyzer(provider="openai")
        assert analyzer.provider == "openai"
        assert analyzer.client is not None

    def test_build_prompt_with_valid_data(self):
        """Test prompt building with valid metrics and config."""
        analyzer = LLMAnalyzer(provider="claude")

        metrics = {
            "precision": 0.79,
            "recall": 0.82,
            "map50": 0.86,
            "map50_95": 0.40,
            "loss": 0.73,
        }

        config = {
            "epochs": 100,
            "batch": 16,
            "lr0": 0.01,
            "iou": 0.7,
            "conf": 0.25,
        }

        prompt = analyzer._build_prompt(metrics, config)

        # Check prompt contains key information
        assert "MODULE CHECKLIST" in prompt
        assert "SCHEMA REMINDERS" in prompt
        assert "GPT-5 USAGE NOTES" in prompt
        assert "Precision 79" in prompt or "0.79" in prompt or "79" in prompt
        assert "Recall" in prompt and ("0.82" in prompt or "82" in prompt)
        assert "epochs" in prompt.lower() or "100" in prompt  # Config is surfaced
        assert 'veri seti boyutu' in prompt.lower()
        assert 'train/val/test' in prompt.lower()
        assert '"dataset_review"' in prompt
        assert '"architecture_alignment"' in prompt
        assert 'yolo large (l)' in prompt.lower()
        assert len(prompt) > 1000  # Prompt should be substantial

    def test_build_prompt_with_missing_metrics(self):
        """Test prompt building with missing metrics."""
        analyzer = LLMAnalyzer(provider="claude")

        metrics = {}  # Empty metrics
        config = {}  # Empty config

        prompt = analyzer._build_prompt(metrics, config)

        # Should still build prompt with N/A values
        assert "N/A" in prompt
        assert "FKT" in prompt

    def test_normalise_list_with_list_input(self):
        """Test _normalise_list with list input."""
        result = LLMAnalyzer._normalise_list(["item1", "item2", "item3"])
        assert result == ["item1", "item2", "item3"]

    def test_normalise_list_with_string_input(self):
        """Test _normalise_list with string input."""
        result = LLMAnalyzer._normalise_list("single item")
        assert result == ["single item"]

    def test_normalise_list_with_none_input(self):
        """Test _normalise_list with None input."""
        result = LLMAnalyzer._normalise_list(None)
        assert result == []

    def test_normalise_list_with_newline_separated_string(self):
        """Test _normalise_list with newline-separated string."""
        result = LLMAnalyzer._normalise_list("item1\nitem2\nitem3")
        assert result == ["item1", "item2", "item3"]

    def test_parse_structured_output_valid_json(self):
        """Test parsing valid JSON output."""
        json_response = """
        {
            "summary": "Test summary",
            "strengths": ["strength1", "strength2"],
            "weaknesses": ["weakness1"],
            "actions": [
                {
                    "module": "veri",
                    "problem": "Eksik örnek",
                    "evidence": "Class A recall %65",
                    "recommendation": "Sınıf başına 30 yeni örnek topla",
                    "expected_gain": "Recall +%5",
                    "validation_plan": "Yeni veri ile hold-out değerlendirmesi"
                }
            ],
            "risk": "medium",
            "deploy_profile": {
                "release_decision": "delay",
                "rollout_strategy": "Ek veri tamamlanana kadar staging",
                "monitoring_plan": "Her gün recall ölçümü"
            },
            "notes": "Test notes"
        }
        """

        result = LLMAnalyzer._parse_structured_output(json_response)

        assert result["summary"] == "Test summary"
        assert result["strengths"] == ["strength1", "strength2"]
        assert result["weaknesses"] == ["weakness1"]
        assert result["risk"] == "medium"
        assert result["notes"] == "Test notes"
        assert result["actions"][0]["module"] == "veri"
        assert result["deploy_profile"]["release_decision"] == "delay"

    def test_parse_structured_output_json_in_text(self):
        """Test parsing JSON embedded in text."""
        text_response = """
        Here is the analysis:
        {
            "summary": "Embedded JSON",
            "strengths": ["good"],
            "weaknesses": ["bad"],
            "actions": [],
            "risk": "low",
            "deploy_profile": {},
            "notes": ""
        }
        Additional text here.
        """

        result = LLMAnalyzer._parse_structured_output(text_response)

        assert result["summary"] == "Embedded JSON"
        assert result["strengths"] == ["good"]
        assert result["weaknesses"] == ["bad"]
        assert result["risk"] == "low"

    def test_parse_structured_output_preserves_calibration(self):
        """Calibration artefacts should pass through untouched."""

        response = {
            "summary": "Calibrated",
            "strengths": [],
            "weaknesses": [],
            "actions": [],
            "risk": "low",
            "deploy_profile": {"release_decision": "go"},
            "calibration": {
                "temperature": 1.2,
                "plots": ["calibration_plot.png"],
            },
        }

        result = LLMAnalyzer._parse_structured_output(json.dumps(response))

        assert "calibration" in result
        assert result["calibration"]["temperature"] == 1.2

    def test_parse_structured_output_empty_raises_error(self):
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError, match="LLM response payload is empty"):
            LLMAnalyzer._parse_structured_output("")

    def test_parse_structured_output_no_json_raises_error(self):
        """Test that response without JSON raises ValueError."""
        with pytest.raises(ValueError, match="Unable to locate JSON object"):
            LLMAnalyzer._parse_structured_output("No JSON here at all")

    def test_openai_responses_payload_and_schema(self, monkeypatch):
        """Ensure OpenAI responses API payload respects schema and preserves keys."""

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        response_payload = {
            "summary": "Detaylı özet",
            "strengths": ["yüksek recall"],
            "weaknesses": ["düşük precision"],
            "actions": [
                {
                    "module": "eşik",
                    "problem": "Confidence yüksek",
                    "evidence": "PR eğrisi 0.25 üstünde düşüyor",
                    "recommendation": "Confidence 0.25 → 0.18",
                    "expected_gain": "Precision +%3",
                    "validation_plan": "Yeni eşikle A/B testi",
                }
            ],
            "risk": "medium",
            "notes": "Notlar",
            "deploy_profile": {
                "release_decision": "delay",
                "rollout_strategy": "Staging'de kal",
            },
            "calibration": {"chart": "calibration.png"},
        }
        response_text = json.dumps(response_payload)

        class DummyResponses:
            def __init__(self) -> None:
                self.last_kwargs = None

            def create(self, **kwargs):
                self.last_kwargs = kwargs

                class _Response:
                    output = [
                        type(
                            "Out",
                            (),
                            {
                                "content": [
                                    type(
                                        "Content",
                                        (),
                                        {"text": response_text},
                                    )()
                                ]
                            },
                        )()
                    ]

                return _Response()

        dummy_responses = DummyResponses()

        class DummyClient:
            def __init__(self) -> None:
                self.responses = dummy_responses

        monkeypatch.setattr(
            "app.analyzers.llm_analyzer.OpenAI",
            lambda api_key: DummyClient(),
        )

        analyzer = LLMAnalyzer(provider="openai")
        result = analyzer._analyze_with_gpt("prompt")

        kwargs = dummy_responses.last_kwargs
        assert kwargs["model"] == "gpt-5"
        assert kwargs["reasoning"]["effort"] == "medium"
        assert kwargs["response_format"]["type"] == "json_schema"
        assert "temperature" not in kwargs
        schema_properties = kwargs["response_format"]["json_schema"]["schema"]["properties"]
        assert "actions" in schema_properties
        assert "deploy_profile" in schema_properties
        assert "risk" in schema_properties

        assert result["actions"] == response_payload["actions"]
        assert result["deploy_profile"] == response_payload["deploy_profile"]
        assert result["calibration"] == response_payload["calibration"]
        assert result["risk"] == response_payload["risk"]

    def test_openai_high_reasoning_switches_model(self, monkeypatch):
        """High reasoning requests should use the thinking model and effort."""

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_HIGH_REASONING", "true")

        response_payload = {
            "summary": "Özet",
            "strengths": [],
            "weaknesses": [],
            "actions": [],
            "risk": "low",
            "notes": "",
            "deploy_profile": {},
        }
        response_text = json.dumps(response_payload)

        class DummyResponses:
            def __init__(self) -> None:
                self.last_kwargs = None

            def create(self, **kwargs):
                self.last_kwargs = kwargs

                class _Response:
                    output = [
                        type(
                            "Out",
                            (),
                            {
                                "content": [
                                    type(
                                        "Content",
                                        (),
                                        {"text": response_text},
                                    )()
                                ]
                            },
                        )()
                    ]

                return _Response()

        dummy_responses = DummyResponses()

        class DummyClient:
            def __init__(self) -> None:
                self.responses = dummy_responses

        monkeypatch.setattr(
            "app.analyzers.llm_analyzer.OpenAI",
            lambda api_key: DummyClient(),
        )

        analyzer = LLMAnalyzer(provider="openai")
        analyzer._analyze_with_gpt("prompt")

        kwargs = dummy_responses.last_kwargs
        assert kwargs["model"] == "gpt-5-thinking"
        assert "temperature" not in kwargs
        assert kwargs["reasoning"]["effort"] == "high"
