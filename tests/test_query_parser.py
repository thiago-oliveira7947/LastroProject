"""Testes para src/search/query_parser.py — sem dependências externas."""
from __future__ import annotations

import pytest

from src.search.query_parser import parse, QueryParsed, TIPO_KEYWORDS, POI_KEYWORDS, STOPWORDS


# ─── QueryParsed defaults ─────────────────────────────────────────────────────

class TestQueryParsedDefaults:
    def test_default_location_empty(self):
        q = QueryParsed()
        assert q.location_text == ""

    def test_default_tipo_hint_empty(self):
        q = QueryParsed()
        assert q.tipo_hint == []

    def test_default_quartos_none(self):
        q = QueryParsed()
        assert q.quartos_hint is None

    def test_default_banheiros_none(self):
        q = QueryParsed()
        assert q.banheiros_hint is None

    def test_default_vagas_none(self):
        q = QueryParsed()
        assert q.vagas_hint is None

    def test_default_poi_hints_empty(self):
        q = QueryParsed()
        assert q.poi_hints == []


# ─── Tipos ────────────────────────────────────────────────────────────────────

class TestTipos:
    def test_apartamento_detected(self):
        result = parse("apartamento 2 quartos")
        assert "apartment" in result.tipo_hint

    def test_apto_shorthand(self):
        result = parse("apto centro guarulhos")
        assert "apartment" in result.tipo_hint

    def test_flat_detected(self):
        result = parse("flat perto do metro")
        assert "apartment" in result.tipo_hint

    def test_studio_detected(self):
        result = parse("studio moderno")
        assert "apartment" in result.tipo_hint

    def test_cobertura_detected(self):
        result = parse("cobertura duplex")
        assert "apartment" in result.tipo_hint

    def test_casa_detected(self):
        result = parse("casa 3 quartos jardim")
        assert "house" in result.tipo_hint

    def test_sobrado_detected(self):
        result = parse("sobrado 200m2")
        assert "house" in result.tipo_hint

    def test_residencia_detected(self):
        result = parse("residencia familiar")
        assert "house" in result.tipo_hint

    def test_chacara_detected(self):
        result = parse("chacara sitio grande")
        assert "house" in result.tipo_hint

    def test_sitio_detected(self):
        result = parse("sitio com piscina")
        assert "house" in result.tipo_hint

    def test_comercial_detected(self):
        result = parse("sala comercial centro")
        assert "commercial" in result.tipo_hint

    def test_loja_detected(self):
        result = parse("loja ponto comercial")
        assert "commercial" in result.tipo_hint

    def test_galpao_detected(self):
        result = parse("galpao industrial")
        assert "commercial" in result.tipo_hint

    def test_escritorio_detected(self):
        result = parse("escritorio centro")
        assert "commercial" in result.tipo_hint

    def test_terreno_detected(self):
        result = parse("terreno 500m2")
        assert "land" in result.tipo_hint

    def test_lote_detected(self):
        result = parse("lote residencial")
        assert "land" in result.tipo_hint

    def test_no_tipo_defaults_empty(self):
        result = parse("3 quartos perto do metro")
        assert result.tipo_hint == []

    def test_no_duplicate_tipo(self):
        # "apartamento" and "apto" both → apartment, but only once
        result = parse("apartamento apto grande")
        assert result.tipo_hint.count("apartment") == 1

    def test_multiple_tipos(self):
        result = parse("apartamento ou casa")
        assert "apartment" in result.tipo_hint
        assert "house" in result.tipo_hint


# ─── Quartos / Banheiros / Vagas ──────────────────────────────────────────────

class TestNumericos:
    def test_quartos_extracted(self):
        result = parse("apartamento 3 quartos")
        assert result.quartos_hint == 3

    def test_quarto_singular(self):
        result = parse("1 quarto kitnet")
        assert result.quartos_hint == 1

    def test_dorm_keyword(self):
        result = parse("2 dorm centro")
        assert result.quartos_hint == 2

    def test_suite_keyword(self):
        result = parse("casa 4 suites")
        assert result.quartos_hint == 4

    def test_banheiros_extracted(self):
        result = parse("apto 2 banheiros")
        assert result.banheiros_hint == 2

    def test_wc_keyword(self):
        result = parse("casa 3 wc")
        assert result.banheiros_hint == 3

    def test_vagas_extracted(self):
        result = parse("apartamento 2 vagas")
        assert result.vagas_hint == 2

    def test_garagem_keyword(self):
        result = parse("sobrado 3 garagem")
        assert result.vagas_hint == 3

    def test_no_quartos_returns_none(self):
        result = parse("apartamento centro")
        assert result.quartos_hint is None

    def test_no_banheiros_returns_none(self):
        result = parse("casa bonita")
        assert result.banheiros_hint is None

    def test_no_vagas_returns_none(self):
        result = parse("terreno grande")
        assert result.vagas_hint is None

    def test_all_numericos_combined(self):
        result = parse("casa 4 quartos 3 banheiros 2 vagas")
        assert result.quartos_hint == 4
        assert result.banheiros_hint == 3
        assert result.vagas_hint == 2


# ─── POIs ─────────────────────────────────────────────────────────────────────

class TestPOIs:
    def test_supermercado_detected(self):
        result = parse("apartamento perto de supermercado")
        assert "supermercado" in result.poi_hints

    def test_mercado_detected(self):
        result = parse("casa proximo mercado")
        assert "supermercado" in result.poi_hints

    def test_escola_detected(self):
        result = parse("perto de escola")
        assert "escola" in result.poi_hints

    def test_colegio_detected(self):
        result = parse("colegio proximo")
        assert "escola" in result.poi_hints

    def test_faculdade_detected(self):
        result = parse("perto faculdade")
        assert "escola" in result.poi_hints

    def test_universidade_detected(self):
        result = parse("universidade 5 minutos")
        assert "escola" in result.poi_hints

    def test_restaurante_detected(self):
        result = parse("proximo a restaurante")
        assert "restaurante" in result.poi_hints

    def test_hospital_detected(self):
        result = parse("perto de hospital")
        assert "hospital" in result.poi_hints

    def test_clinica_detected(self):
        result = parse("clinica medica")
        assert "hospital" in result.poi_hints

    def test_farmacia_detected(self):
        result = parse("proximo farmacia")
        assert "hospital" in result.poi_hints

    def test_parque_detected(self):
        result = parse("perto do parque")
        assert "parque" in result.poi_hints

    def test_praca_detected(self):
        result = parse("praca perto")
        assert "parque" in result.poi_hints

    def test_metro_detected(self):
        result = parse("proximo ao metro")
        assert "metro" in result.poi_hints

    def test_cptm_detected(self):
        result = parse("linha cptm")
        assert "metro" in result.poi_hints

    def test_estacao_detected(self):
        result = parse("estacao de trem")
        assert "metro" in result.poi_hints

    def test_no_poi_hints(self):
        result = parse("apartamento 3 quartos Guarulhos")
        assert result.poi_hints == []

    def test_no_duplicate_poi(self):
        result = parse("mercado supermercado")
        assert result.poi_hints.count("supermercado") == 1

    def test_multiple_pois(self):
        result = parse("perto de escola e hospital e metro")
        assert "escola" in result.poi_hints
        assert "hospital" in result.poi_hints
        assert "metro" in result.poi_hints


# ─── location_text ────────────────────────────────────────────────────────────

class TestLocationText:
    def test_city_remains_in_location(self):
        result = parse("apartamento Guarulhos")
        assert "guarulhos" in result.location_text

    def test_stopwords_removed(self):
        result = parse("apartamento em Guarulhos")
        tokens = result.location_text.split()
        assert "em" not in tokens

    def test_short_tokens_removed(self):
        result = parse("a apartamento")
        tokens = result.location_text.split()
        assert "a" not in tokens

    def test_empty_query(self):
        result = parse("")
        assert result.location_text == ""
        assert result.tipo_hint == []
        assert result.quartos_hint is None
        assert result.poi_hints == []

    def test_only_stopwords(self):
        result = parse("de a em para no na")
        assert result.location_text == ""

    def test_complex_query(self):
        result = parse("apartamento 3 quartos Jardim Maia Guarulhos perto de escola")
        assert "apartment" in result.tipo_hint
        assert result.quartos_hint == 3
        assert "escola" in result.poi_hints
        assert "guarulhos" in result.location_text or "jardim" in result.location_text
