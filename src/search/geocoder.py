"""Geocodificacao de enderecos via Nominatim (OpenStreetMap) — 100% gratis.

Rate limit: 1 req/s conforme politica do Nominatim.
"""
from __future__ import annotations

import sys
import time

import requests


def _log(*args, **kwargs):
    print(*args, **kwargs, file=sys.stderr, flush=True)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "LastroProject/1.0 (imoveis-br-demo)"

_last_call: float = 0.0


def geocode(texto: str) -> tuple[float, float] | None:
    """Converte texto de endereco em (lat, lon). Retorna None se nao encontrar."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    params = {
        "q": texto,
        "format": "json",
        "limit": 1,
        "countrycodes": "br",
        "addressdetails": 0,
    }
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt"}

    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=8)
        _last_call = time.time()
        data = r.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception:
        pass
    return None


def geocode_com_fallback(texto: str, cidade_fallback: str = "Guarulhos, São Paulo") -> tuple[float, float]:
    """Tenta geocodificar; se falhar, usa cidade_fallback."""
    result = geocode(f"{texto}, Brasil")
    if result:
        return result
    result = geocode(f"{cidade_fallback}, Brasil")
    return result or (-23.4628, -46.5333)


def geocode_detalhado(texto: str) -> dict:
    """Geocodifica com addressdetails=1; retorna {lat, lon, cidade, estado, bairro}.

    Retorna dict vazio se falhar.
    """
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)

    params = {
        "q": f"{texto}, Brasil",
        "format": "json",
        "limit": 1,
        "countrycodes": "br",
        "addressdetails": 1,
    }
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt"}

    _log(f"[GEOCODER] Requisição Nominatim: q={params['q']!r}")
    try:
        r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=8)
        _last_call = time.time()
        _log(f"[GEOCODER] Status HTTP: {r.status_code}")
        data = r.json()
        if not data:
            _log("[GEOCODER] Nominatim retornou lista vazia — nenhum resultado.")
            return {}
        hit = data[0]
        addr = hit.get("address", {})
        _log(f"[GEOCODER] Endereço bruto retornado: {addr}")
        cidade = (
            addr.get("city") or
            addr.get("town") or
            addr.get("municipality") or
            addr.get("county") or ""
        )
        estado = addr.get("state") or ""
        bairro = (
            addr.get("suburb") or
            addr.get("city_district") or
            addr.get("neighbourhood") or
            addr.get("quarter") or ""
        )
        result = {
            "lat":    float(hit["lat"]),
            "lon":    float(hit["lon"]),
            "cidade": cidade,
            "estado": estado,
            "bairro": bairro,
        }
        _log(f"[GEOCODER] Resultado: cidade={cidade!r} estado={estado!r} bairro={bairro!r} lat={result['lat']} lon={result['lon']}")
        return result
    except Exception as e:
        _log(f"[GEOCODER] Exceção: {e}")
        return {}
