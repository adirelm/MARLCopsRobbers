"""SDK package — the single business-logic entry point (CLAUDE.md §3).

``from src.sdk import MarlSDK`` is the sanctioned public import (V3 §14 ``__all__``).
"""

from src.sdk.sdk import MarlSDK

__all__ = ["MarlSDK"]
