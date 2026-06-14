"""
Cliente Mercado Livre para busca de imóveis reais.

Fluxo de ativação (credenciais gratuitas):
  1. Acesse https://developers.mercadolibre.com.br
  2. Crie um app (categoria Imóveis) → obtenha CLIENT_ID e CLIENT_SECRET
  3. Copie para o arquivo .env (veja .env.example)
  4. Na primeira execução do app, clique em "Conectar Mercado Livre"
  5. Faça login com sua conta ML e autorize o app
  6. O access_token é salvo em .ml_token.json e renovado automaticamente

Sem credenciais → app funciona normalmente com o dataset local.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

_BASE = "https://api.mercadolibre.com"
_AUTH = "https://auth.mercadolibre.com.br/authorization"
_TOKEN = f"{_BASE}/oauth/token"
_SEARCH = f"{_BASE}/sites/MLB/search"
_CATEGORY = "MLB1459"  # Imóveis Brasil
_TOKEN_FILE = Path(".ml_token.json")

_ATTR_MAP = {
    "quartos":       ["BEDROOMS"],
    "banheiros":     ["BATHROOMS"],
    "area":          ["TOTAL_AREA", "COVERED_AREA", "USEFUL_AREA"],
    "vagas_garagem": ["PARKING_LOTS"],
}

_TIPO_HINTS = {
    "apartment": ["apartamento", "apto", "flat", "cobertura", "studio"],
    "house":     ["casa", "sobrado", "residencia", "chacara", "sitio"],
    "land":      ["terreno", "lote"],
    "commercial":["comercial", "sala", "galpao", "loja", "escritorio"],
}


# ─── Auth helpers ─────────────────────────────────────────────────────────

def get_credentials() -> tuple[str, str] | None:
    """Retorna (client_id, client_secret) do .env ou variáveis de ambiente."""
    cid = os.getenv("ML_CLIENT_ID", "").strip()
    sec = os.getenv("ML_CLIENT_SECRET", "").strip()
    return (cid, sec) if cid and sec else None


def auth_url() -> str:
    """URL para o usuário autorizar o app no Mercado Livre."""
    creds = get_credentials()
    if not creds:
        return ""
    cid, _ = creds
    redirect = os.getenv("ML_REDIRECT_URI", "http://localhost:8501/callback")
    return (
        f"{_AUTH}?response_type=code&client_id={cid}"
        f"&redirect_uri={redirect}&scope=read"
    )


def trocar_codigo(code: str) -> bool:
    """Troca o authorization code por access+refresh tokens. Salva em .ml_token.json."""
    creds = get_credentials()
    if not creds:
        return False
    cid, sec = creds
    redirect = os.getenv("ML_REDIRECT_URI", "http://localhost:8501/callback")
    try:
        r = requests.post(_TOKEN, data={
            "grant_type": "authorization_code",
            "client_id": cid,
            "client_secret": sec,
            "code": code,
            "redirect_uri": redirect,
        }, timeout=10)
        data = r.json()
        if "access_token" in data:
            data["expires_at"] = time.time() + data.get("expires_in", 21600) - 60
            _TOKEN_FILE.write_text(json.dumps(data))
            return True
    except Exception:
        pass
    return False


def _renovar_token(refresh_token: str) -> dict | None:
    creds = get_credentials()
    if not creds:
        return None
    cid, sec = creds
    try:
        r = requests.post(_TOKEN, data={
            "grant_type": "refresh_token",
            "client_id": cid,
            "client_secret": sec,
            "refresh_token": refresh_token,
        }, timeout=10)
        data = r.json()
        if "access_token" in data:
            data["expires_at"] = time.time() + data.get("expires_in", 21600) - 60
            _TOKEN_FILE.write_text(json.dumps(data))
            return data
    except Exception:
        pass
    return None


def _access_token() -> str | None:
    if not _TOKEN_FILE.exists():
        return None
    try:
        data = json.loads(_TOKEN_FILE.read_text())
        if time.time() > data.get("expires_at", 0):
            renewed = _renovar_token(data.get("refresh_token", ""))
            return renewed.get("access_token") if renewed else None
        return data.get("access_token")
    except Exception:
        return None


def esta_autenticado() -> bool:
    return bool(_access_token())


# ─── Data helpers ─────────────────────────────────────────────────────────

def _inferir_tipo(titulo: str) -> str:
    t = titulo.lower()
    for tipo, kws in _TIPO_HINTS.items():
        if any(kw in t for kw in kws):
            return tipo
    return "apartment"


def _get_attr(attrs: list[dict], ids: list[str]) -> float:
    import re
    for aid in ids:
        for a in attrs:
            if a.get("id") == aid and a.get("value_name"):
                try:
                    return float(re.sub(r"[^\d.]", "", str(a["value_name"])))
                except (ValueError, TypeError):
                    pass
    return 0.0


def _normalizar(raw: dict) -> dict:
    attrs = raw.get("attributes", [])
    loc = raw.get("location") or {}
    addr = raw.get("address") or {}
    sl = addr.get("search_location") or {}

    cidade = (
        loc.get("city", {}).get("name")
        or sl.get("city", {}).get("name")
        or addr.get("city_name", "")
    ).lower().strip()

    estado = (
        loc.get("state", {}).get("name")
        or sl.get("state", {}).get("name")
        or addr.get("state_name", "")
    ).strip()

    bairro = (
        loc.get("neighborhood", {}).get("name")
        or sl.get("neighborhood", {}).get("name")
        or ""
    ).strip()

    thumb = raw.get("thumbnail") or ""
    thumb = thumb.replace("-I.jpg", "-O.jpg").replace("_I.jpg", "_O.jpg")

    return {
        "id":            raw.get("id", ""),
        "titulo":        raw.get("title", ""),
        "preco":         float(raw.get("price") or 0),
        "thumbnail":     thumb,
        "link":          raw.get("permalink", ""),
        "cidade":        cidade,
        "estado":        estado,
        "bairro":        bairro,
        "tipo":          _inferir_tipo(raw.get("title", "")),
        "quartos":       int(_get_attr(attrs, _ATTR_MAP["quartos"])),
        "banheiros":     int(_get_attr(attrs, _ATTR_MAP["banheiros"])),
        "area":          _get_attr(attrs, _ATTR_MAP["area"]),
        "vagas_garagem": int(_get_attr(attrs, _ATTR_MAP["vagas_garagem"])),
        "condominio":    0.0,
        "iptu":          0.0,
        "latitude":      float(loc["latitude"]) if loc.get("latitude") else None,
        "longitude":     float(loc["longitude"]) if loc.get("longitude") else None,
        "fonte":         "mercadolivre",
    }


# ─── Public API ───────────────────────────────────────────────────────────

def buscar_publico(query: str, limit: int = 24) -> list[dict]:
    """
    Busca imóveis no Mercado Livre sem autenticação (endpoint público).
    Retorna [] em caso de erro ou bloqueio.
    """
    try:
        r = requests.get(
            _SEARCH,
            params={"q": query, "category": _CATEGORY,
                    "limit": min(limit, 48), "sort": "relevance"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"},
            timeout=12,
        )
        if r.status_code in (401, 403, 429):
            return []
        r.raise_for_status()
        return [_normalizar(item) for item in r.json().get("results", [])]
    except Exception:
        return []


def buscar(query: str, limit: int = 24, offset: int = 0) -> list[dict]:
    """
    Busca imóveis no Mercado Livre.
    Requer autenticação — veja auth_url() / trocar_codigo().
    Retorna [] se não autenticado ou em caso de erro.
    """
    token = _access_token()
    if not token:
        return []
    try:
        r = requests.get(
            _SEARCH,
            params={"q": query, "category": _CATEGORY,
                    "limit": min(limit, 48), "offset": offset},
            headers={"Authorization": f"Bearer {token}",
                     "User-Agent": "LastroProject/1.0"},
            timeout=12,
        )
        if r.status_code == 401:
            _TOKEN_FILE.unlink(missing_ok=True)
            return []
        r.raise_for_status()
        return [_normalizar(item) for item in r.json().get("results", [])]
    except Exception:
        return []
