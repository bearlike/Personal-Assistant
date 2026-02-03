import asyncio
import sys
import types

import pytest

from core.common import get_mock_speaker
from tools.core.talk_to_user import TalkToUser
from tools.integration.homeassistant import HomeAssistant
from tools.integration.mcp import MCPToolRunner, _load_mcp_config


def test_talk_to_user_set_state(monkeypatch):
    monkeypatch.setattr(TalkToUser, "__init__", lambda self: None)
    tool = TalkToUser()
    step = types.SimpleNamespace(action_argument="hello")
    result = tool.set_state(step)
    assert result.content == "hello"


def test_talk_to_user_requires_step(monkeypatch):
    monkeypatch.setattr(TalkToUser, "__init__", lambda self: None)
    tool = TalkToUser()
    with pytest.raises(ValueError):
        tool.set_state(None)


def test_mcp_config_requires_env(monkeypatch):
    monkeypatch.delenv("MESEEKS_MCP_CONFIG", raising=False)
    with pytest.raises(ValueError):
        _load_mcp_config()


def test_mcp_tool_runner_uses_async(monkeypatch):
    runner = MCPToolRunner(server_name="srv", tool_name="tool")
    async def _fake_invoke(_):
        return "ok"

    monkeypatch.setattr(runner, "_invoke_async", _fake_invoke)
    step = types.SimpleNamespace(action_argument="ping")
    result = runner.run(step)
    assert result.content == "ok"


def test_mcp_invoke_async_success(monkeypatch, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"servers": {"srv": {"transport": "stdio"}}}', encoding="utf-8")
    monkeypatch.setenv("MESEEKS_MCP_CONFIG", str(config_path))

    class DummyTool:
        name = "tool"

        async def ainvoke(self, input_text):
            return f"out:{input_text}"

    class DummyClient:
        def __init__(self, servers):
            self.servers = servers

        async def get_tools(self, server_name):
            return [DummyTool()]

    module = types.ModuleType("langchain_mcp_adapters.client")
    module.MultiServerMCPClient = DummyClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", module)

    runner = MCPToolRunner(server_name="srv", tool_name="tool")
    result = asyncio.run(runner._invoke_async("hi"))
    assert result == "out:hi"


def test_mcp_invoke_async_missing_tool(monkeypatch, tmp_path):
    config_path = tmp_path / "mcp.json"
    config_path.write_text('{"servers": {"srv": {"transport": "stdio"}}}', encoding="utf-8")
    monkeypatch.setenv("MESEEKS_MCP_CONFIG", str(config_path))

    class DummyClient:
        def __init__(self, servers):
            self.servers = servers

        async def get_tools(self, server_name):
            return []

    module = types.ModuleType("langchain_mcp_adapters.client")
    module.MultiServerMCPClient = DummyClient
    monkeypatch.setitem(sys.modules, "langchain_mcp_adapters.client", module)

    runner = MCPToolRunner(server_name="srv", tool_name="tool")
    with pytest.raises(ValueError):
        asyncio.run(runner._invoke_async("hi"))


def _make_homeassistant():
    ha = HomeAssistant.__new__(HomeAssistant)
    ha.base_url = "http://test"
    ha._api_token = "token"
    ha.api_headers = {"Authorization": "Bearer token", "Content-Type": "application/json"}
    ha.cache = {
        "entity_ids": [],
        "sensor_ids": [],
        "entities": [],
        "services": [],
        "sensors": [],
        "allowed_domains": ["scene", "switch"],
    }
    ha.model_name = "dummy"
    ha.model = object()
    ha._save_json = lambda *_args, **_kwargs: None
    ha._load_rag_documents = lambda *_args, **_kwargs: []
    ha.update_cache = lambda: None
    return ha


def test_homeassistant_clean_answer():
    answer = HomeAssistant._clean_answer('RealFeel 10km/h "test"')
    assert "Real Feel" in answer
    assert "kilometer per hour" in answer
    assert '"' not in answer


def test_homeassistant_update_services(monkeypatch):
    ha = _make_homeassistant()

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"domain": "scene"}]

    monkeypatch.setattr(
        "tools.integration.homeassistant.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )
    assert ha.update_services() is True
    assert ha.cache["services"]


def test_homeassistant_update_entities(monkeypatch):
    ha = _make_homeassistant()

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [
                {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
                {"entity_id": "sensor.temp", "state": "20", "attributes": {}},
            ]

    monkeypatch.setattr(
        "tools.integration.homeassistant.requests.get",
        lambda *args, **kwargs: DummyResponse(),
    )
    assert ha.update_entities() is True
    assert any(entity["entity_id"] == "light.kitchen" for entity in ha.cache["entities"])
    assert any(sensor["entity_id"] == "sensor.temp" for sensor in ha.cache["sensors"])


def test_homeassistant_update_entity_ids(monkeypatch):
    ha = _make_homeassistant()

    def fake_update_entities():
        ha.cache["entities"] = [
            {"entity_id": "light.kitchen", "state": "on", "attributes": {}},
            {"entity_id": "switch.fan", "state": "off", "attributes": {}},
        ]
        return True

    monkeypatch.setattr(ha, "update_entities", fake_update_entities)
    assert ha.update_entity_ids() is True
    assert "light.kitchen" in ha.cache["entity_ids"]


def test_homeassistant_prompt_builders():
    class DummyParser:
        def get_format_instructions(self):
            return "format"

    set_prompt = HomeAssistant._create_set_prompt("system", DummyParser())
    get_prompt = HomeAssistant._create_get_prompt("system")
    assert set_prompt is not None
    assert get_prompt is not None


def test_homeassistant_call_service_success(monkeypatch):
    ha = _make_homeassistant()

    class DummyResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return [{"ok": True}]

        @property
        def text(self):
            return "ok"

    monkeypatch.setattr(
        "tools.integration.homeassistant.requests.post",
        lambda *args, **kwargs: DummyResponse(),
    )
    ok, payload = ha.call_service("scene", "turn_on", "scene.lamp")
    assert ok is True
    assert payload == [{"ok": True}]


def test_homeassistant_call_service_invalid_domain():
    ha = _make_homeassistant()
    with pytest.raises(ValueError):
        ha.call_service("invalid", "turn_on", "scene.lamp")


def test_homeassistant_invoke_service(monkeypatch):
    ha = _make_homeassistant()

    class DummyCall:
        domain = "scene"
        service = "turn_on"
        entity_id = "scene.lamp"

    class DummyChain:
        def invoke(self, *args, **kwargs):
            return DummyCall()

    monkeypatch.setattr(ha, "call_service", lambda **kwargs: (True, {"ok": True}))
    step = types.SimpleNamespace(action_argument="turn on lamp")
    result = ha._invoke_service_and_set_state(DummyChain(), [], step)
    assert "Successfully called service" in result.content


def test_homeassistant_set_state(monkeypatch):
    ha = _make_homeassistant()

    class DummyChain:
        def __or__(self, _other):
            return self

    class DummyParser:
        def __init__(self, *args, **kwargs):
            pass

        def get_format_instructions(self):
            return "format"

    monkeypatch.setattr(
        "tools.integration.homeassistant.PydanticOutputParser",
        DummyParser,
    )
    monkeypatch.setattr(HomeAssistant, "_create_set_prompt", lambda *_a, **_k: DummyChain())
    monkeypatch.setattr(
        "tools.integration.homeassistant.ha_render_system_prompt",
        lambda *args, **kwargs: "prompt",
    )
    monkeypatch.setattr(
        HomeAssistant,
        "_invoke_service_and_set_state",
        lambda *a, **k: get_mock_speaker()(content="ok"),
    )
    step = types.SimpleNamespace(action_argument="turn on")
    result = ha.set_state(step)
    assert result.content == "ok"


def test_homeassistant_get_state(monkeypatch):
    ha = _make_homeassistant()

    class DummyChain:
        def __or__(self, _other):
            return self

        def invoke(self, *args, **kwargs):
            return types.SimpleNamespace(content="answer")

    monkeypatch.setattr(HomeAssistant, "_create_get_prompt", lambda *_a, **_k: DummyChain())
    monkeypatch.setattr(
        "tools.integration.homeassistant.ha_render_system_prompt",
        lambda *args, **kwargs: "prompt",
    )
    step = types.SimpleNamespace(action_argument="status")
    result = ha.get_state(step)
    assert result.content == "answer"
