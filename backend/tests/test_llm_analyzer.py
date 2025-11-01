"""Tests for LLM analyzer."""
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
        assert "FKT" in prompt
        assert "leather seat dent" in prompt.lower()
        assert "0.79" in prompt or "79" in prompt  # Precision value
        assert "0.82" in prompt or "82" in prompt  # Recall value
        assert "100" in prompt  # Epochs
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
            "action_items": [{"description": "action1"}],
            "risk_level": "medium",
            "notes": "Test notes"
        }
        """

        result = LLMAnalyzer._parse_structured_output(json_response)

        assert result["summary"] == "Test summary"
        assert result["strengths"] == ["strength1", "strength2"]
        assert result["weaknesses"] == ["weakness1"]
        assert result["risk_level"] == "medium"
        assert result["notes"] == "Test notes"

    def test_parse_structured_output_json_in_text(self):
        """Test parsing JSON embedded in text."""
        text_response = """
        Here is the analysis:
        {
            "summary": "Embedded JSON",
            "strengths": ["good"],
            "weaknesses": ["bad"],
            "action_items": [],
            "risk_level": "low",
            "notes": ""
        }
        Additional text here.
        """

        result = LLMAnalyzer._parse_structured_output(text_response)

        assert result["summary"] == "Embedded JSON"
        assert result["strengths"] == ["good"]
        assert result["weaknesses"] == ["bad"]

    def test_parse_structured_output_empty_raises_error(self):
        """Test that empty response raises ValueError."""
        with pytest.raises(ValueError, match="LLM response payload is empty"):
            LLMAnalyzer._parse_structured_output("")

    def test_parse_structured_output_no_json_raises_error(self):
        """Test that response without JSON raises ValueError."""
        with pytest.raises(ValueError, match="Unable to locate JSON object"):
            LLMAnalyzer._parse_structured_output("No JSON here at all")
