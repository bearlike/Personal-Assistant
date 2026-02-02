import json

from core.tool_registry import load_registry


def test_default_registry(monkeypatch):
    monkeypatch.delenv("MESEEKS_TOOL_MANIFEST", raising=False)
    registry = load_registry()
    tool_ids = {spec.tool_id for spec in registry.list_specs()}
    assert "home_assistant_tool" in tool_ids
    assert "talk_to_user_tool" in tool_ids


def test_manifest_local_tool(tmp_path, monkeypatch):
    module_path = tmp_path / "dummy_tool.py"
    module_path.write_text(
        "class DummyTool:\n"
        "    def run(self, action_step):\n"
        "        return None\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "tools": [
                    {
                        "tool_id": "dummy_tool",
                        "name": "Dummy",
                        "description": "Test tool",
                        "module": "dummy_tool",
                        "class": "DummyTool",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    registry = load_registry(str(manifest_path))
    tool = registry.get("dummy_tool")

    assert tool is not None
    assert tool.__class__.__name__ == "DummyTool"


def test_manifest_empty_falls_back(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"tools": []}), encoding="utf-8")

    registry = load_registry(str(manifest_path))
    tool_ids = {spec.tool_id for spec in registry.list_specs()}
    assert "home_assistant_tool" in tool_ids
    assert "talk_to_user_tool" in tool_ids
