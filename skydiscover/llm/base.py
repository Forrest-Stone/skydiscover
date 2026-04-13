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
    model_name: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    estimated_cost: Optional[float] = None
    usage_raw: Optional[Dict[str, Any]] = None

    @property
    def total_tokens(self) -> int:
        return max(0, int(self.prompt_tokens or 0)) + max(0, int(self.completion_tokens or 0))

    @property
    def input_tokens(self) -> int:
        """Backward-compatible alias for prompt_tokens."""
        return max(0, int(self.prompt_tokens or 0))

    @property
    def output_tokens(self) -> int:
        """Backward-compatible alias for completion_tokens."""
        return max(0, int(self.completion_tokens or 0))

    @property
    def raw_usage(self) -> Optional[Dict[str, Any]]:
        """Backward-compatible alias for usage_raw."""
        return self.usage_raw


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
