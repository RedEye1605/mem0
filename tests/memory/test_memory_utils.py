import hashlib

import pytest

from mem0.memory.utils import (
    ensure_json_instruction,
    extract_json,
    normalize_facts,
    parse_messages,
    process_telemetry_filters,
    remove_code_blocks,
    remove_spaces_from_entities,
    sanitize_relationship_for_cypher,
)


# ---------------------------------------------------------------------------
# normalize_facts
# ---------------------------------------------------------------------------


class TestNormalizeFacts:
    def test_plain_strings(self):
        assert normalize_facts(["fact1", "fact2"]) == ["fact1", "fact2"]

    def test_empty_list(self):
        assert normalize_facts([]) == []

    def test_none_input(self):
        assert normalize_facts(None) == []

    def test_dict_with_fact_key(self):
        assert normalize_facts([{"fact": "User likes pizza"}]) == ["User likes pizza"]

    def test_dict_with_text_key(self):
        assert normalize_facts([{"text": "Lives in NYC"}]) == ["Lives in NYC"]

    def test_dict_with_text_key_preferred_over_empty_fact(self):
        """'text' key is used when 'fact' key is absent."""
        assert normalize_facts([{"text": "works at Acme"}]) == ["works at Acme"]

    def test_dict_without_fact_or_text_skipped(self, caplog):
        result = normalize_facts([{"something": "else"}])
        assert result == []
        assert "Unexpected fact shape" in caplog.text

    def test_empty_string_fact_skipped(self):
        assert normalize_facts([""]) == []

    def test_non_string_non_dict_converted_to_str(self):
        assert normalize_facts([123]) == ["123"]

    def test_mixed_types(self):
        facts = [
            "plain fact",
            {"fact": "dict fact"},
            {"text": "text fact"},
            42,
            "",
            {"other": "skipped"},
        ]
        assert normalize_facts(facts) == ["plain fact", "dict fact", "text fact", "42"]

    def test_unicode_content(self):
        facts = ["用户喜欢披萨", {"fact": "住在雅加达"}]
        result = normalize_facts(facts)
        assert result == ["用户喜欢披萨", "住在雅加达"]


# ---------------------------------------------------------------------------
# parse_messages
# ---------------------------------------------------------------------------


class TestParseMessages:
    def test_basic_user_message(self):
        msgs = [{"role": "user", "content": "hello"}]
        assert parse_messages(msgs) == "user: hello\n"

    def test_system_and_user(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        result = parse_messages(msgs)
        assert "system: You are helpful.\n" in result
        assert "user: Hi\n" in result

    def test_all_roles(self):
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": "asst"},
        ]
        result = parse_messages(msgs)
        assert result == "system: sys\nuser: usr\nassistant: asst\n"

    def test_unicode_content(self):
        msgs = [{"role": "user", "content": "Halo, apa kabar?"}]
        result = parse_messages(msgs)
        assert "Halo, apa kabar?" in result


# ---------------------------------------------------------------------------
# ensure_json_instruction
# ---------------------------------------------------------------------------


class TestEnsureJsonInstruction:
    def test_json_already_present(self):
        sys_p, user_p = ensure_json_instruction("Use JSON format.", "input data")
        assert sys_p == "Use JSON format."

    def test_json_not_present_appends_instruction(self):
        sys_p, user_p = ensure_json_instruction("Extract facts.", "here is the data")
        assert "json" in sys_p.lower() or "json" in user_p.lower()
        assert "facts" in sys_p

    def test_json_in_user_prompt(self):
        sys_p, user_p = ensure_json_instruction("Extract facts.", "Return valid json")
        # Should NOT modify since json appears in user prompt
        assert sys_p == "Extract facts."


# ---------------------------------------------------------------------------
# extract_json
# ---------------------------------------------------------------------------


class TestExtractJson:
    def test_plain_json(self):
        text = '{"facts": ["a", "b"]}'
        assert extract_json(text) == '{"facts": ["a", "b"]}'

    def test_json_in_code_block(self):
        text = '```json\n{"facts": ["a"]}\n```'
        result = extract_json(text)
        assert '"facts"' in result

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"facts": ["x"]}\nEnd.'
        result = extract_json(text)
        assert result == '{"facts": ["x"]}'

    def test_no_json_returns_text(self):
        text = "just plain text"
        assert extract_json(text) == "just plain text"

    def test_empty_string(self):
        assert extract_json("") == ""


# ---------------------------------------------------------------------------
# remove_code_blocks
# ---------------------------------------------------------------------------


class TestRemoveCodeBlocks:
    def test_no_code_block(self):
        assert remove_code_blocks("plain text") == "plain text"

    def test_code_block_with_language(self):
        assert remove_code_blocks("```json\n{\"key\": \"val\"}\n```") == '{"key": "val"}'

    def test_code_block_without_language(self):
        assert remove_code_blocks("```\nsome code\n```") == "some code"

    def test_strips_whitespace(self):
        assert remove_code_blocks("  ```python\nx = 1\n```  ") == "x = 1"


# ---------------------------------------------------------------------------
# process_telemetry_filters
# ---------------------------------------------------------------------------


class TestProcessTelemetryFilters:
    def test_none_returns_empty_dict(self):
        assert process_telemetry_filters(None) == {}

    def test_user_id_encoded(self):
        keys, encoded = process_telemetry_filters({"user_id": "alice"})
        assert keys == ["user_id"]
        expected = hashlib.md5("alice".encode("utf-8")).hexdigest()
        assert encoded["user_id"] == expected

    def test_all_ids_encoded(self):
        filters = {"user_id": "u", "agent_id": "a", "run_id": "r"}
        keys, encoded = process_telemetry_filters(filters)
        assert sorted(keys) == sorted(["user_id", "agent_id", "run_id"])
        for key in ["user_id", "agent_id", "run_id"]:
            expected = hashlib.md5(filters[key].encode("utf-8")).hexdigest()
            assert encoded[key] == expected

    def test_non_ascii_id_encoded(self):
        """Non-ASCII user IDs should not crash the encoding."""
        keys, encoded = process_telemetry_filters({"user_id": "用户123"})
        assert keys == ["user_id"]
        expected = hashlib.md5("用户123".encode("utf-8")).hexdigest()
        assert encoded["user_id"] == expected

    def test_unicode_run_id(self):
        """Indonesian/Japanese Unicode in run_id."""
        keys, encoded = process_telemetry_filters({"run_id": "session-雅加达"})
        assert keys == ["run_id"]
        expected = hashlib.md5("session-雅加达".encode("utf-8")).hexdigest()
        assert encoded["run_id"] == expected


# ---------------------------------------------------------------------------
# sanitize_relationship_for_cypher
# ---------------------------------------------------------------------------


class TestRemoveSpacesFromEntities:
    """
    Covers behavior used by Neo4j, Memgraph (sanitize_relationship=True),
    Kuzu, and Neptune (sanitize_relationship=False). All backends delegate here.
    """

    @pytest.mark.parametrize(
        "sanitize",
        [True, False],
        ids=["cypher_sanitized", "plain"],
    )
    def test_filters_empty_and_incomplete_dicts(self, sanitize):
        mixed = [
            {},
            {"source": "a"},
            {"source": "a", "relationship": "r"},
            {"source": "x", "relationship": "rel", "destination": "y"},
        ]
        out = remove_spaces_from_entities(mixed, sanitize_relationship=sanitize)
        assert len(out) == 1
        assert out[0]["source"] == "x"
        assert out[0]["destination"] == "y"

    @pytest.mark.parametrize("sanitize", [True, False])
    def test_all_empty_returns_empty(self, sanitize):
        assert remove_spaces_from_entities([{}, {}, {}], sanitize_relationship=sanitize) == []

    def test_skips_non_dict_entries(self):
        assert remove_spaces_from_entities([None, "not-a-dict", 123, {"source": "a", "relationship": "r", "destination": "b"}]) == [
            {"source": "a", "relationship": "r", "destination": "b"}
        ]

    def test_sanitize_true_relationship_uses_sanitizer(self):
        """Neo4j / Memgraph path: special characters mapped via sanitize_relationship_for_cypher."""
        entities = [{"source": "A", "relationship": "x/y", "destination": "B"}]
        out = remove_spaces_from_entities(entities, sanitize_relationship=True)
        assert out[0]["relationship"] == sanitize_relationship_for_cypher("x/y".lower().replace(" ", "_"))

    def test_sanitize_false_relationship_plain_only(self):
        """Kuzu / Neptune path: only lowercase and spaces to underscores."""
        entities = [{"source": "A", "relationship": "Works At", "destination": "B Co"}]
        out = remove_spaces_from_entities(entities, sanitize_relationship=False)
        assert out[0]["relationship"] == "works_at"
        assert out[0]["source"] == "a"
        assert out[0]["destination"] == "b_co"

    def test_sanitize_true_vs_false_slash_in_relationship(self):
        """Slash is rewritten when sanitizing (Cypher path); kept as-is for plain path."""
        base = {"source": "s", "relationship": "a/b", "destination": "d"}
        t = remove_spaces_from_entities([dict(base)], sanitize_relationship=True)[0]["relationship"]
        f = remove_spaces_from_entities([dict(base)], sanitize_relationship=False)[0]["relationship"]
        assert t == sanitize_relationship_for_cypher("a/b")
        assert f == "a/b"


# ---------------------------------------------------------------------------
# sanitize_relationship_for_cypher
# ---------------------------------------------------------------------------


class TestSanitizeRelationshipForCypher:
    def test_simple_string_unchanged(self):
        assert sanitize_relationship_for_cypher("works_at") == "works_at"

    def test_slash_replaced(self):
        assert sanitize_relationship_for_cypher("a/b") == "a_slash_b"

    def test_parentheses_replaced(self):
        assert sanitize_relationship_for_cypher("test(x)") == "test_lparen_x_rparen"

    def test_spaces_not_replaced(self):
        """Spaces are not in the char_map — they're handled by remove_spaces_from_entities instead."""
        assert sanitize_relationship_for_cypher("works at") == "works at"

    def test_consecutive_underscores_collapsed(self):
        assert sanitize_relationship_for_cypher("a__b") == "a_b"

    def test_leading_trailing_underscores_stripped(self):
        assert sanitize_relationship_for_cypher("_hello_") == "hello"

    def test_unicode_characters(self):
        """CJK punctuation should be replaced."""
        result = sanitize_relationship_for_cypher("关系，测试")
        assert "comma_" in result
        assert "," not in result

    def test_empty_string(self):
        assert sanitize_relationship_for_cypher("") == ""
