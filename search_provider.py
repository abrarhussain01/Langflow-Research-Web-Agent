"""Resolve search once per research question, preserving configured credentials."""

import os


def resolve_search_provider():
    provider = os.getenv("SEARCH_PROVIDER", "auto").strip().lower()
    if provider == "auto":
        return "brightdata" if os.getenv("BRIGHTDATA_API_KEY", "").strip() else "parallel"
    if provider not in {"brightdata", "parallel", "none"}:
        raise ValueError("SEARCH_PROVIDER must be auto, brightdata, parallel, or none")
    if provider == "brightdata" and not os.getenv("BRIGHTDATA_API_KEY", "").strip():
        raise ValueError("SEARCH_PROVIDER=brightdata requires BRIGHTDATA_API_KEY")
    return provider
