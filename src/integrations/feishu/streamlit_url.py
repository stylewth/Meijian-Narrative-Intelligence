"""Resolve the Streamlit URL used by newly-created Feishu cards."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


AUTO_STREAMLIT_URL = "auto"
DEFAULT_STREAMLIT_PORT = 8501
MAX_AUTO_DETECT_PORT = 8599


def resolve_streamlit_public_url(
    configured_url: str | None,
    *,
    environ: Mapping[str, Any] | None = None,
    probe: Callable[[str], bool] | None = None,
) -> str:
    """Return an explicit URL or discover the live local Streamlit instance.

    ``auto`` (and an empty value) searches the default Streamlit port range.
    ``STREAMLIT_SERVER_PORT`` narrows that search to one port.  The probe is
    injectable so offline tests never need to make network requests.
    """

    if configured_url is not None and not isinstance(configured_url, str):
        raise ValueError("STREAMLIT_PUBLIC_URL must be text or auto")
    value = "" if configured_url is None else configured_url.strip()
    if value and value.lower() != AUTO_STREAMLIT_URL:
        return value

    environment = os.environ if environ is None else environ
    ports = _candidate_ports(environment)
    probe_fn = _probe_streamlit_health if probe is None else probe
    for port in ports:
        base_url = f"http://127.0.0.1:{port}"
        if probe_fn(f"{base_url}/_stcore/health"):
            return base_url

    if len(ports) == 1:
        searched = str(ports[0])
    else:
        searched = f"{ports[0]}-{ports[-1]}"
    raise ValueError(
        f"STREAMLIT_PUBLIC_URL=auto found no live local Streamlit on port {searched}"
    )


def _candidate_ports(environ: Mapping[str, Any]) -> tuple[int, ...]:
    raw_port = environ.get("STREAMLIT_SERVER_PORT")
    if raw_port is None or (isinstance(raw_port, str) and not raw_port.strip()):
        return tuple(range(DEFAULT_STREAMLIT_PORT, MAX_AUTO_DETECT_PORT + 1))

    try:
        port = int(str(raw_port).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError("STREAMLIT_SERVER_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("STREAMLIT_SERVER_PORT must be between 1 and 65535")
    return (port,)


def _probe_streamlit_health(health_url: str) -> bool:
    request = Request(health_url, headers={"Accept": "text/plain"})
    try:
        with urlopen(request, timeout=0.75) as response:
            body = response.read().decode("utf-8", errors="replace").strip().lower()
            return response.status == 200 and body == "ok"
    except (HTTPError, URLError, TimeoutError, OSError):
        return False


__all__ = [
    "AUTO_STREAMLIT_URL",
    "DEFAULT_STREAMLIT_PORT",
    "MAX_AUTO_DETECT_PORT",
    "resolve_streamlit_public_url",
]
