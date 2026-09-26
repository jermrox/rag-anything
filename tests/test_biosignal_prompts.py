"""Tests for the biosignal prompt templates and their isolation."""

import pytest

from raganything.biosignal.prompts import BIOSIGNAL_PROMPTS, SYSTEM_PROMPT_FIELDS


class TestAnswerTemplate:
    def test_formats_with_exactly_the_legal_fields(self):
        rendered = BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"].format(
            response_type="Multiple Paragraphs",
            user_prompt="be brief",
            context_data="SENTINEL",
        )
        assert "SENTINEL" in rendered
        assert "be brief" in rendered

    def test_context_placeholder_is_present(self):
        # Omitting it silently discards every retrieved chunk, which is why the
        # module asserts this at import time as well.
        assert "{context_data}" in BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"]

    def test_import_time_validator_rejects_a_missing_context_slot(self):
        from raganything.biosignal.prompts import _assert_format_fields

        with pytest.raises(RuntimeError, match="drops"):
            _assert_format_fields(
                "bad", "no context here: {response_type}{user_prompt}"
            )

    def test_import_time_validator_rejects_a_stray_field(self):
        from raganything.biosignal.prompts import _assert_format_fields

        with pytest.raises(RuntimeError, match="may only reference"):
            _assert_format_fields("bad", "{context_data} {something_else}")

    def test_no_template_has_unbalanced_braces(self):
        for name, template in BIOSIGNAL_PROMPTS.items():
            assert template.count("{") == template.count("}"), name

    def test_router_template_formats(self):
        rendered = BIOSIGNAL_PROMPTS["ROUTER_CLASSIFY"].format(
            question="q", metrics="hrv_rmssd, mean_hr"
        )
        assert '"route"' in rendered
        assert "hrv_rmssd" in rendered

    def test_scope_template_formats(self):
        rendered = BIOSIGNAL_PROMPTS["SCOPE_USER_PROMPT"].format(
            start="2026-07-01", end="2026-08-01"
        )
        assert "2026-07-01" in rendered

    def test_system_prompt_fields_are_the_documented_three(self):
        assert SYSTEM_PROMPT_FIELDS == ("response_type", "user_prompt", "context_data")


class TestGlobalRegistryIsolation:
    def test_importing_the_query_layer_does_not_pollute_prompts(self):
        """Regression guard for a trap in the surrounding framework.

        Keys added to the shared registry after import are erased by the next
        language switch, and any key without a Chinese counterpart breaks
        tests/test_prompt_language.py. So the biosignal prompts must stay in
        their own dictionary.
        """
        from raganything.prompt import PROMPTS

        before = set(PROMPTS.keys())
        import raganything.biosignal.query  # noqa: F401 - imported for effect
        import raganything.biosignal.router  # noqa: F401
        import raganything.biosignal.verify  # noqa: F401

        assert set(PROMPTS.keys()) == before

    def test_biosignal_keys_are_absent_from_the_shared_registry(self):
        from raganything.prompt import PROMPTS

        for key in BIOSIGNAL_PROMPTS:
            assert key not in PROMPTS

    def test_resetting_prompts_leaves_biosignal_templates_intact(self):
        from raganything.prompt_manager import reset_prompts

        before = dict(BIOSIGNAL_PROMPTS)
        reset_prompts()
        assert BIOSIGNAL_PROMPTS == before

    def test_switching_language_leaves_biosignal_templates_intact(self):
        from raganything.prompt_manager import reset_prompts, set_prompt_language

        before = dict(BIOSIGNAL_PROMPTS)
        try:
            set_prompt_language("zh")
            assert BIOSIGNAL_PROMPTS == before
        finally:
            reset_prompts()
