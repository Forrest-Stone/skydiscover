"""Shared utilities for the OpenAI Responses API.

Provides message conversion and output extraction helpers used by both
the non-agentic path (openai.py) and the agentic path (agentic_generator.py).
"""

from typing import Any, Dict, List, Optional, Tuple


def convert_messages_to_responses_input(messages: List[Dict[str, Any]]) -> list:
    """Convert Chat Completions-style messages to Responses API input format.

    Handles:
    - user / assistant text messages (plain string or multipart content)
    - assistant messages with tool_calls -> function_call items
    - tool role messages -> function_call_output items
    """
    items: list = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.get("tool_call_id", ""),
                    "output": content if isinstance(content, str) else "",
                }
            )
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    fn = tc.get("function", {})
                    if not isinstance(fn, dict):
                        fn = {}
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": tc.get("id", ""),
                            "name": fn.get("name", ""),
                            "arguments": fn.get("arguments", "{}"),
                        }
                    )
                # If assistant had both text and tool_calls, skip the text
                # (Responses API treats function_call items as the assistant turn)
                if not content:
                    continue

        # Text-only message (user, assistant without tool_calls, or system)
        if isinstance(content, str):
            items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                }
            )
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")
                if ptype == "text":
                    parts.append({"type": "input_text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    url = part.get("image_url", {}).get("url", "")
                    parts.append({"type": "input_image", "image_url": url, "detail": "auto"})
            items.append({"type": "message", "role": role, "content": parts})

    return items


def extract_responses_output(
    response,
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    """Extract text, image, and tool calls from a Responses API response.

    Returns:
        (text, image_b64, tool_calls) where tool_calls is a list of
        Chat-Completions-compatible tool call dicts (may be empty).
    """
    text_parts: List[str] = []
    image_b64: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = []

    output_items = getattr(response, "output", None)
    if output_items is None and isinstance(response, dict):
        output_items = response.get("output")
    output_items = output_items or []
    for item in output_items:
        if isinstance(item, dict):
            item_type = item.get("type")
        else:
            item_type = getattr(item, "type", None)
        if item_type == "message":
            if isinstance(item, dict):
                content_parts = item.get("content") or []
            else:
                content_parts = getattr(item, "content", None) or []
            for part in content_parts:
                if isinstance(part, dict):
                    text = part.get("text")
                    if text is None:
                        text = part.get("output_text")
                    if text:
                        text_parts.append(str(text))
                elif hasattr(part, "text") and part.text is not None:
                    text_parts.append(part.text)
        elif item_type == "image_generation_call":
            result = item.get("result") if isinstance(item, dict) else getattr(item, "result", None)
            if result:
                image_b64 = result
        elif item_type == "function_call":
            if isinstance(item, dict):
                call_id = item.get("call_id", "")
                name = item.get("name", "")
                arguments = item.get("arguments", "{}")
            else:
                call_id = getattr(item, "call_id", "")
                name = getattr(item, "name", "")
                arguments = getattr(item, "arguments", "{}")
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                }
            )

    return "\n".join(text_parts), image_b64, tool_calls
