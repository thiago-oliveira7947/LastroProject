"""Busca de pontos de interesse (POIs) via Overpass API (OpenStreetMap) — gratis.

Retorna localizacoes (lat, lon) de cada POI para que o app possa calcular
distancias localmente e filtrar imoveis por proximidade.
"""
from __future__ import annotations

import requests

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Mapeia categoria do usuario -> tags Overpass amenity
AMENITY_MAP: dict[str, list[str]] = {
    "supermercado": ["supermarket", "convenience"],
    "escola":       ["school", "college", "university", "kindergarten"],
    "restaurante":  ["restaurant", "fast_food", "cafe", "food_court"],
    "hospital":     ["hospital", "clinic", "pharmacy"],
    "parque":       [],   # usa leisure=park (tratamento especial)
    "metro":        ["subway_entrance"],
}

# Tags de transporte publ. adicionais (railway)
RAILWAY_MAP: dict[str, list[str]] = {
    "metro": ["station", "halt", "subway_entrance"],
}


def buscar_pois_localizacoes(
    lat: float,
    lon: float,
    raio_m: int,
    categorias: list[str],
) -> dict[str, list[tuple[float, float]]]:
    """Retorna {categoria: [(lat, lon), ...]} para todos os POIs no raio."""
    results: dict[str, list[tuple[float, float]]] = {}

    for cat in categorias:
        amenities = AMENITY_MAP.get(cat, [])
        partes = []

        if amenities:
            filtro = "|".join(amenities)
            partes.append(
                f'node["amenity"~"{filtro}"](around:{raio_m},{lat},{lon});'
            )

        if cat == "parque":
            partes.append(
                f'node["leisure"="park"](around:{raio_m},{lat},{lon});'
                f'way["leisure"="park"](around:{raio_m},{lat},{lon});'
            )

        if cat in RAILWAY_MAP:
            filtro_rw = "|".join(RAILWAY_MAP[cat])
            partes.append(
                f'node["railway"~"{filtro_rw}"](around:{raio_m},{lat},{lon});'
            )

        if not partes:
            results[cat] = []
            continue

        query = f"""
[out:json][timeout:15];
(
  {"".join(partes)}
);
out body;
"""
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, timeout=20)
            data = r.json()
            results[cat] = [
                (float(el["lat"]), float(el["lon"]))
                for el in data.get("elements", [])
                if "lat" in el and "lon" in el
            ]
        except Exception:
            results[cat] = []

    return results
