from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_MOD_PATH = Path(__file__).resolve().parents[2] / "skydiscover" / "llm" / "responses_utils.py"
_spec = spec_from_file_location("responses_utils_mod", _MOD_PATH)
_mod = module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)

extract_responses_output = _mod.extract_responses_output


def test_extract_responses_output_handles_dict_shape():
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "hello"},
                    {"type": "output_text", "output_text": "world"},
                ],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_fn",
                "arguments": '{"x":1}',
            },
        ]
    }
    text, image_b64, tool_calls = extract_responses_output(response)
    assert text == "hello\nworld"
    assert image_b64 is None
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "tool_fn"
