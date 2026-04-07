"""Base LLM interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class LLMResponse:
    """Response from an LLM generation call.

    text: generated text content.
    image_path: path to generated image file, or None for text-only.
    """

    text: str = ""
    image_path: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    raw_usage: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return max(0, int(self.input_tokens)) + max(0, int(self.output_tokens))


class LLMInterface(ABC):
    """Abstract base for LLM backends.

    Subclass this and implement generate() to add a new LLM provider.
    """

    @abstractmethod
    async def generate(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            system_message: system prompt string.
            messages: conversation history as list of {role, content} dicts.
            **kwargs: backend-specific options (e.g. image_output=True for
                image generation, output_dir, program_id, temperature).

        Returns:
            LLMResponse with text and optional image_path.
        """
        pass

    async def generate_with_usage(
        self, system_message: str, messages: List[Dict[str, Any]], **kwargs
    ) -> LLMResponse:
        """Generate a response with usage metadata if available."""
        return await self.generate(system_message, messages, **kwargs)
