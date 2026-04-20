"""Tests for LLM config: optional temperature/top_p and api_base routing."""

from dataclasses import fields
from unittest.mock import AsyncMock, patch

import pytest

from skydiscover.config import Config, LLMConfig, LLMModelConfig, apply_overrides

_OPENAI_DEFAULT_API_BASE: str = next(
    f.default for f in fields(LLMConfig) if f.name == "api_base"
)


class TestLLMConfigDefaults:
    def test_default_temperature(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.temperature == 0.7

    def test_default_top_p_is_none(self):
        cfg = LLMConfig(name="test-model")
        assert cfg.top_p is None

    def test_explicit_none_temperature(self):
        cfg = LLMConfig(name="test-model", temperature=None)
        assert cfg.temperature is None

    def test_explicit_none_top_p(self):
        cfg = LLMConfig(name="test-model", top_p=None)
        assert cfg.top_p is None

    def test_both_none(self):
        cfg = LLMConfig(name="test-model", temperature=None, top_p=None)
        assert cfg.temperature is None
        assert cfg.top_p is None


class TestApiBaseRouting:
    def test_unknown_model_preserves_local_api_base(self):
        local = "http://localhost:11434/v1"
        cfg = LLMConfig(
            name="my-custom-local-model",
            api_base=local,
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == local

    def test_unknown_model_gets_openai_default(self):
        cfg = LLMConfig(
            name="my-custom-local-model",
            models=[LLMModelConfig(name="my-custom-local-model")],
        )
        assert cfg.models[0].api_base == _OPENAI_DEFAULT_API_BASE

    def test_mixed_providers_with_local_api_base(self):
        cfg = LLMConfig(
            api_base="http://localhost:11434/v1",
            models=[
                LLMModelConfig(name="anthropic/claude-3-sonnet"),
                LLMModelConfig(name="my-local-model"),
            ],
        )
        assert cfg.models[0].api_base == "https://api.anthropic.com/v1/"
        assert cfg.models[1].api_base == "http://localhost:11434/v1"

    def test_openrouter_model_uses_openrouter_api_base(self):
        cfg = LLMConfig(
            models=[LLMModelConfig(name="openrouter/deepseek/deepseek-r1")],
        )
        assert cfg.models[0].api_base == "https://openrouter.ai/api/v1"

    def test_openrouter_prefers_openrouter_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("OPENAI_API_KEY", "oa-key")
        cfg = LLMConfig(
            models=[LLMModelConfig(name="openrouter/deepseek/deepseek-r1")],
        )
        assert cfg.models[0].api_key == "or-key"

    def test_openrouter_api_base_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://my-cn-proxy.example/v1")
        cfg = LLMConfig(
            models=[LLMModelConfig(name="openrouter/deepseek/deepseek-r1")],
        )
        assert cfg.models[0].api_base == "https://my-cn-proxy.example/v1"

    def test_apply_overrides_openrouter_uses_env_api_base(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_BASE", "https://my-cn-proxy.example/v1")
        cfg = LLMConfig(models=[LLMModelConfig(name="gpt-5")])
        outer = type("Cfg", (), {"llm": cfg, "agentic": type("A", (), {"enabled": False})()})()
        apply_overrides(outer, model="openrouter/openai/gpt-5")
        assert outer.llm.models[0].api_base == "https://my-cn-proxy.example/v1"

    def test_openai_model_bridges_to_openrouter_when_only_openrouter_key_exists(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_BASE", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        cfg = LLMConfig(models=[LLMModelConfig(name="gpt-5")])
        assert cfg.models[0].api_key == "or-key"
        assert cfg.models[0].api_base == "https://openrouter.ai/api/v1"
        assert cfg.models[0].name == "openai/gpt-5"

    def test_openrouter_bare_claude_model_gets_upstream_prefix(self):
        cfg = LLMConfig(models=[LLMModelConfig(name="openrouter/claude-3-5-haiku")])
        assert cfg.models[0].name == "anthropic/claude-3-5-haiku"

    def test_api_key_can_be_hardcoded_in_single_code_location(self, monkeypatch):
        import skydiscover.config as config_module

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("SKYDISCOVER_API_KEY", raising=False)
        monkeypatch.setattr(config_module, "_HARDCODED_FALLBACK_API_KEY", "sk-or-hardcoded")

        cfg = LLMConfig(models=[LLMModelConfig(name="openrouter/deepseek/deepseek-r1")])
        assert cfg.models[0].api_key == "sk-or-hardcoded"

    def test_apply_overrides_bridged_openai_model_gets_upstream_prefix(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        cfg = LLMConfig(models=[LLMModelConfig(name="gpt-5")])
        outer = type("Cfg", (), {"llm": cfg, "agentic": type("A", (), {"enabled": False})()})()
        apply_overrides(outer, model="gpt-5-mini")
        assert outer.llm.models[0].name == "openai/gpt-5-mini"
        assert outer.llm.models[0].api_base == "https://openrouter.ai/api/v1"

    def test_apply_overrides_evox_model_override_enables_share_llm(self):
        cfg = Config()
        cfg.search.type = "evox"
        cfg.search.share_llm = False

        apply_overrides(cfg, model="openrouter/deepseek/deepseek-r1")

        assert cfg.search.share_llm is True

    def test_apply_overrides_model_syncs_default_monitor_summary_model(self):
        cfg = Config()
        cfg.monitor.summary_model = "gpt-5-mini"

        apply_overrides(cfg, model="openrouter/deepseek/deepseek-r1")

        assert cfg.monitor.summary_model == "deepseek/deepseek-r1"
        assert cfg.monitor.summary_api_base == cfg.llm.api_base

    def test_apply_overrides_keeps_custom_monitor_summary_model(self):
        cfg = Config()
        cfg.monitor.summary_model = "openai/gpt-4o-mini"

        apply_overrides(cfg, model="openrouter/deepseek/deepseek-r1")

        assert cfg.monitor.summary_model == "openai/gpt-4o-mini"


class TestOpenAILLMParams:
    def _make_llm(self, temperature=0.7, top_p=0.95):
        from skydiscover.llm.openai import OpenAILLM

        cfg = LLMModelConfig(
            name="test-model",
            temperature=temperature,
            top_p=top_p,
            api_base="http://localhost:1234/v1",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI"):
            llm = OpenAILLM(cfg)
        return llm

    def test_cloudflare_gateway_uses_cf_aig_auth_header(self, monkeypatch):
        from skydiscover.llm.openai import OpenAILLM

        monkeypatch.setenv("CF_AIG_AUTH_TOKEN", "cf-token")
        cfg = LLMModelConfig(
            name="openrouter/deepseek/deepseek-chat",
            temperature=0.7,
            top_p=0.95,
            api_base="https://gateway.ai.cloudflare.com/v1/acc/gw/openrouter",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI") as mock_openai:
            OpenAILLM(cfg)
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["default_headers"]["cf-aig-authorization"] == "Bearer cf-token"

    def test_default_headers_json_env_is_passed_to_client(self, monkeypatch):
        from skydiscover.llm.openai import OpenAILLM

        monkeypatch.setenv("OPENAI_DEFAULT_HEADERS_JSON", '{"x-test-header": "abc"}')
        cfg = LLMModelConfig(
            name="test-model",
            temperature=0.7,
            top_p=0.95,
            api_base="http://localhost:1234/v1",
            api_key="fake",
            timeout=10,
            retries=0,
            retry_delay=0,
        )
        with patch("skydiscover.llm.openai.openai.OpenAI") as mock_openai:
            OpenAILLM(cfg)
        kwargs = mock_openai.call_args.kwargs
        assert kwargs["default_headers"]["x-test-header"] == "abc"

    @pytest.mark.asyncio
    async def test_params_include_temperature_and_top_p(self):
        llm = self._make_llm(temperature=0.5, top_p=0.9)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(
            system_message="sys",
            messages=[{"role": "user", "content": "user"}],
            temperature=0.5,
            top_p=0.9,
        )
        params = llm._call_api.call_args[0][0]
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    @pytest.mark.asyncio
    async def test_params_exclude_none_top_p(self):
        llm = self._make_llm(top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "top_p" not in params
        assert "temperature" in params

    @pytest.mark.asyncio
    async def test_params_exclude_none_temperature(self):
        llm = self._make_llm(temperature=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" in params

    @pytest.mark.asyncio
    async def test_params_exclude_both_none(self):
        llm = self._make_llm(temperature=None, top_p=None)
        llm._call_api = AsyncMock(return_value="response")
        await llm.generate(system_message="sys", messages=[{"role": "user", "content": "user"}])
        params = llm._call_api.call_args[0][0]
        assert "temperature" not in params
        assert "top_p" not in params

    def test_extract_chat_text_handles_none_message_content(self):
        from skydiscover.llm.openai import OpenAILLM

        class _Msg:
            content = None

        class _Choice:
            message = _Msg()
            text = "fallback-text"

        class _Resp:
            choices = [_Choice()]

        assert OpenAILLM._extract_chat_text(_Resp()) == "fallback-text"

    def test_extract_chat_text_handles_dict_content_parts(self):
        from skydiscover.llm.openai import OpenAILLM

        resp = {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "output_text", "text": "hello"}, {"text": " world"}]
                    }
                }
            ]
        }
        assert OpenAILLM._extract_chat_text(resp) == "hello world"

    def test_convert_messages_to_responses_input_ignores_invalid_items(self):
        from skydiscover.llm.responses_utils import convert_messages_to_responses_input

        items = convert_messages_to_responses_input(
            [None, {"role": "user", "content": [{"type": "text", "text": "hi"}, None]}]
        )
        assert len(items) == 1
        assert items[0]["content"][0]["text"] == "hi"

    def test_extract_responses_output_handles_empty_content(self):
        from skydiscover.llm.responses_utils import extract_responses_output

        class _Item:
            type = "message"
            content = None

        class _Resp:
            output = [_Item()]

        text, image_b64, tool_calls = extract_responses_output(_Resp())
        assert text == ""
        assert image_b64 is None
        assert tool_calls == []

    @pytest.mark.asyncio
    async def test_call_api_via_responses_uses_module_helpers(self):
        llm = self._make_llm()

        class _TextPart:
            text = "ok"

        class _Item:
            type = "message"
            content = [_TextPart()]

        class _Resp:
            output = [_Item()]

        llm.client.responses.create = lambda **kwargs: _Resp()
        text = await llm._call_api_via_responses(
            {
                "model": "openrouter/deepseek/deepseek-r1",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
            }
        )
        assert text == "ok"

    @pytest.mark.asyncio
    async def test_call_api_falls_back_when_chat_text_empty(self):
        llm = self._make_llm()

        class _ChatResp:
            choices = None

        class _TextPart:
            text = "resp-ok"

        class _Item:
            type = "message"
            content = [_TextPart()]

        class _Resp:
            output = [_Item()]

        llm.client.chat.completions.create = lambda **kwargs: _ChatResp()
        llm.client.responses.create = lambda **kwargs: _Resp()
        text = await llm._call_api(
            {"model": "openrouter/deepseek/deepseek-r1", "messages": [{"role": "user", "content": "x"}]}
        )
        assert text == "resp-ok"
