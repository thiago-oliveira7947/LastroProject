"""Parser de linguagem natural para extrair filtros de busca de imoveis.

Exemplo:
    parse("apartamento 3 quartos Jardim Maia Guarulhos perto de escola")
    -> QueryParsed(tipo_hint=["apartment"], quartos_hint=3,
                   poi_hints=["escola"], location_text="Jardim Maia Guarulhos")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TIPO_KEYWORDS: dict[str, list[str]] = {
    "apartment": ["apartamento", "apto", "flat", "studio", "cobertura"],
    "house":     ["casa", "sobrado", "residencia", "chácara", "chacara", "sitio"],
    "commercial":["comercial", "sala comercial", "loja", "galpao", "escritório", "escritorio"],
    "land":      ["terreno", "lote", "área", "area"],
}

POI_KEYWORDS: dict[str, list[str]] = {
    "supermercado": ["supermercado", "mercado", "mercadao", "mercearia"],
    "escola":       ["escola", "colegio", "faculdade", "universidade", "creche", "educacao"],
    "restaurante":  ["restaurante", "bar", "lanchonete", "padaria", "comida"],
    "hospital":     ["hospital", "clinica", "ubs", "posto de saude", "farmacia"],
    "parque":       ["parque", "praca", "bosque", "area verde", "jardim"],
    "metro":        ["metro", "trem", "estacao", "cptm", "ferroviaria"],
}

STOPWORDS = {
    "perto", "proximo", "próximo", "próxima", "proxima", "de", "a", "ao", "da",
    "do", "e", "com", "em", "para", "por", "na", "no", "nas", "nos",
    "um", "uma", "uns", "umas", "o", "a", "os", "as",
}


@dataclass
class QueryParsed:
    location_text: str = ""
    tipo_hint: list[str] = field(default_factory=list)
    quartos_hint: int | None = None
    banheiros_hint: int | None = None
    vagas_hint: int | None = None
    poi_hints: list[str] = field(default_factory=list)


def parse(texto: str) -> QueryParsed:
    t = texto.lower().strip()
    result = QueryParsed()

    # Tipo
    for tipo, kws in TIPO_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                if tipo not in result.tipo_hint:
                    result.tipo_hint.append(tipo)
                t = t.replace(kw, " ")

    # Quartos / suites / dormitorios
    m = re.search(r"(\d+)\s*(?:quarto|dorm|suite|suíte|dorm\.)", t)
    if m:
        result.quartos_hint = int(m.group(1))
        t = t[: m.start()] + t[m.end() :]

    # Banheiros
    m = re.search(r"(\d+)\s*(?:banheiro|wc|lavabo)", t)
    if m:
        result.banheiros_hint = int(m.group(1))
        t = t[: m.start()] + t[m.end() :]

    # Vagas
    m = re.search(r"(\d+)\s*(?:vaga|garagem)", t)
    if m:
        result.vagas_hint = int(m.group(1))
        t = t[: m.start()] + t[m.end() :]

    # POIs
    for poi, kws in POI_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                if poi not in result.poi_hints:
                    result.poi_hints.append(poi)
                t = t.replace(kw, " ")

    # Remove stopwords da parte de localizacao
    tokens = t.split()
    loc_tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]
    result.location_text = " ".join(loc_tokens).strip()

    return result
