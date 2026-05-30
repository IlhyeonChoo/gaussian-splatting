"""Network-boundary access controls for the web UI."""

from __future__ import annotations

import ipaddress
import subprocess
from ipaddress import IPv4Address, IPv6Address

from .config import WebUIConfig


TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")
LOOPBACK_NETWORKS = (
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
)


def detect_tailscale_ipv4(timeout_seconds: float = 2.0) -> str | None:
    """Return the first IPv4 address reported by `tailscale ip -4`."""
    try:
        completed = subprocess.run(
            ["tailscale", "ip", "-4"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None

    if completed.returncode != 0:
        return None

    for token in completed.stdout.split():
        try:
            address = ipaddress.ip_address(token)
        except ValueError:
            continue
        if isinstance(address, IPv4Address):
            return str(address)
    return None


def _parse_ip(value: str) -> IPv4Address | IPv6Address | None:
    try:
        return ipaddress.ip_address(value)
    except ValueError:
        return None


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    address = _parse_ip(host)
    return bool(address and address.is_loopback)


def _is_tailscale_host(host: str) -> bool:
    address = _parse_ip(host)
    return bool(address and isinstance(address, IPv4Address) and address in TAILSCALE_CGNAT)


def validate_bind_host(host: str, unsafe_allow_all: bool) -> None:
    """Reject bind hosts that would expose the app outside loopback/Tailscale."""
    if unsafe_allow_all:
        return

    if host in {"0.0.0.0", "::"}:
        raise ValueError(
            "Refusing to bind to all interfaces. Set WEBUI_UNSAFE_ALLOW_ALL=1 "
            "only if you intentionally want external exposure."
        )
    if _is_loopback_host(host) or _is_tailscale_host(host):
        return

    raise ValueError(
        f"Refusing to bind to non-loopback, non-Tailscale host '{host}'. "
        "Use a Tailscale IP, localhost, or set WEBUI_UNSAFE_ALLOW_ALL=1."
    )


def resolve_bind_host(config: WebUIConfig) -> str:
    """Resolve the host uvicorn should bind to."""
    if config.host_override:
        validate_bind_host(config.host_override, config.unsafe_allow_all)
        return config.host_override

    if config.bind_mode == "localhost":
        return "127.0.0.1"

    tailscale_ip = detect_tailscale_ipv4()
    if config.bind_mode == "tailscale":
        if not tailscale_ip:
            raise ValueError("WEBUI_BIND_MODE=tailscale requested, but no Tailscale IPv4 was detected.")
        return tailscale_ip

    return tailscale_ip or "127.0.0.1"


def build_allowed_networks(
    config: WebUIConfig,
    tailscale_ip: str | None = None,
) -> tuple[ipaddress._BaseNetwork, ...]:
    """Build request client networks accepted by middleware."""
    networks: list[ipaddress._BaseNetwork] = list(LOOPBACK_NETWORKS)
    networks.append(TAILSCALE_CGNAT)

    if tailscale_ip:
        address = ipaddress.ip_address(tailscale_ip)
        networks.append(ipaddress.ip_network(f"{address}/32", strict=False))

    for cidr in config.allowed_cidrs:
        network = ipaddress.ip_network(cidr, strict=False)
        if not config.unsafe_allow_all and network.prefixlen == 0:
            raise ValueError(
                "Refusing WEBUI_ALLOWED_CIDRS with a catch-all network unless "
                "WEBUI_UNSAFE_ALLOW_ALL=1 is set."
            )
        networks.append(network)

    return tuple(networks)


def is_client_allowed(
    client_host: str,
    config: WebUIConfig,
    tailscale_ip: str | None = None,
) -> bool:
    """Return whether a request from client_host should be accepted."""
    if config.unsafe_allow_all:
        return True

    address = _parse_ip(client_host)
    if address is None:
        return False

    return any(address in network for network in build_allowed_networks(config, tailscale_ip))

