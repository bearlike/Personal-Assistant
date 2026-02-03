from core.classes import ActionStep
from core.permissions import (
    PermissionDecision,
    PermissionPolicy,
    PermissionRule,
    load_permission_policy,
)


def test_default_policy_allows_get():
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                tool_id="talk_to_user_tool",
                action_type="*",
                decision=PermissionDecision.ALLOW,
            )
        ],
        default_by_action={"get": PermissionDecision.ALLOW, "set": PermissionDecision.ASK},
        default_decision=PermissionDecision.ASK,
    )
    step = ActionStep(
        action_consumer="home_assistant_tool",
        action_type="get",
        action_argument="lights",
    )
    assert policy.decide(step) == PermissionDecision.ALLOW


def test_rule_override_denies():
    policy = PermissionPolicy(
        rules=[
            PermissionRule(
                tool_id="home_assistant_tool",
                action_type="set",
                decision=PermissionDecision.DENY,
            )
        ],
        default_by_action={"get": PermissionDecision.ALLOW, "set": PermissionDecision.ASK},
        default_decision=PermissionDecision.ASK,
    )
    step = ActionStep(
        action_consumer="home_assistant_tool",
        action_type="set",
        action_argument="heater",
    )
    assert policy.decide(step) == PermissionDecision.DENY


def test_load_policy_from_json(tmp_path, monkeypatch):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        """
        {
          "rules": [
            {"tool_id": "home_assistant_tool", "action_type": "set", "decision": "deny"}
          ],
          "default_by_action": {"get": "allow", "set": "ask"},
          "default_decision": "ask"
        }
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("MESEEKS_PERMISSION_POLICY", str(policy_path))
    policy = load_permission_policy()
    step = ActionStep(
        action_consumer="home_assistant_tool",
        action_type="set",
        action_argument="lights",
    )
    assert policy.decide(step) == PermissionDecision.DENY
