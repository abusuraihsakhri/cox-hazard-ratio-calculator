"""Deterministic local reasoning adapter used by the optional supervisor demo."""

from .base import PHIGuard


class MockLLM:
    def __init__(self, system_name: str = "Cox Hazard Ratio Calculator"):
        self.system_name = system_name

    def invoke(self, prompt: str) -> str:
        PHIGuard.assert_no_phi(prompt)
        return (
            f"[{self.system_name} deterministic adapter] "
            f"Received query: {prompt[:120]}"
        )


class LLMFactory:
    """Create supported local adapters.

    The repository currently ships only the deterministic mock adapter. External
    providers are intentionally not accepted because no authenticated provider
    integration is implemented in this project.
    """

    @staticmethod
    def create(provider: str = "mock", system_name: str = "Cox Hazard Ratio Calculator"):
        normalized = str(provider).strip().lower()
        if normalized in {"mock", "deterministic", "test"}:
            return MockLLM(system_name)
        raise ValueError(
            f"Unsupported MODEL_PROVIDER '{provider}'. "
            "This repository currently supports only 'mock'."
        )
