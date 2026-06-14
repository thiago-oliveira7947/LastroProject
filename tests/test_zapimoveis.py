"""Testes para src/external/zapimoveis.py."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.external.zapimoveis import (
    _slug,
    _parse_int_from_name,
    _bairro_from_name,
    _bairro_from_url,
    _clean_text,
    _fetch_html,
    _extrair_items_ld,
    _normalizar_ld,
    _url_zap,
    _url_viva,
    buscar,
    buscar_zap,
    buscar_vivareal,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

VALID_ITEM = {
    "item": {
        "@type": "Apartment",
        "@id": "zap_123",
        "name": "Apartamento 2 vagas em Vila Prudente, São Paulo",
        "description": "Excelente apartamento perto do metrô",
        "url": "https://www.zapimoveis.com.br/venda/imovel/zap_123/",
        "offers": {"price": "450000"},
        "address": {
            "addressLocality": "São Paulo",
            "addressRegion": "sp",
            "streetAddress": "Rua das Flores, 100",
        },
        "floorSize": {"value": "75"},
        "numberOfBedrooms": "2",
        "numberOfBathroomsTotal": "1",
        "image": ["https://cdn.example.com/img.jpg"],
    }
}

VALID_JSON_LD = {"@type": "ItemList", "itemListElement": [VALID_ITEM]}


def _make_html(data: dict) -> str:
    return f'<html><script type="application/ld+json">{json.dumps(data)}</script></html>'


def _make_multi_item_html(n: int, preco: str = "450000", quartos: str = "2", area: str = "70") -> str:
    items = []
    for i in range(n):
        items.append({
            "item": {
                "@type": "Apartment",
                "@id": f"id_{i}",
                "name": f"Apto {i}",
                "description": "",
                "url": "https://zapimoveis.com.br/venda/apto/",
                "offers": {"price": preco},
                "address": {
                    "addressLocality": "Guarulhos",
                    "addressRegion": "sp",
                    "streetAddress": "",
                },
                "floorSize": {"value": area},
                "numberOfBedrooms": quartos,
                "numberOfBathroomsTotal": "1",
                "image": [],
            }
        })
    ld = {"@type": "ItemList", "itemListElement": items}
    return f'<html><script type="application/ld+json">{json.dumps(ld)}</script></html>'


# ─── _slug ────────────────────────────────────────────────────────────────────

class TestSlug:
    def test_remove_accents(self):
        assert _slug("São Paulo") == "sao-paulo"

    def test_remove_special_chars(self):
        assert _slug("Minas Gerais!") == "minas-gerais"

    def test_lowercase(self):
        assert _slug("GUARULHOS") == "guarulhos"

    def test_multiple_spaces(self):
        assert _slug("vila   nova") == "vila-nova"

    def test_numbers_preserved(self):
        assert _slug("zona3") == "zona3"

    def test_empty_string(self):
        assert _slug("") == ""

    def test_cedilla(self):
        assert _slug("cobertura coração") == "cobertura-coracao"

    def test_already_ascii(self):
        assert _slug("guarulhos") == "guarulhos"


# ─── _parse_int_from_name ─────────────────────────────────────────────────────

class TestParseIntFromName:
    def test_vagas_match(self):
        assert _parse_int_from_name("Apto 2 vagas garagem", r"(\d+)\s*vaga") == 2

    def test_no_match_returns_zero(self):
        assert _parse_int_from_name("Apto sem vagas", r"(\d+)\s*vaga") == 0

    def test_case_insensitive(self):
        assert _parse_int_from_name("3 VAGAS cobertura", r"(\d+)\s*vaga") == 3

    def test_quartos_pattern(self):
        assert _parse_int_from_name("4 quartos", r"(\d+)\s*quarto") == 4

    def test_large_number(self):
        assert _parse_int_from_name("10 vagas", r"(\d+)\s*vaga") == 10


# ─── _bairro_from_url ────────────────────────────────────────────────────────

class TestBairroFromUrl:
    """Testa extração de bairro a partir da URL do anúncio ZAP/Viva Real."""

    def test_vila_nova_conceicao_zona_sul(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-1-quarto-mobiliado-vila-nova-conceicao-zona-sul-sao-paulo-sp-78m2-id-2763319033/"
        assert _bairro_from_url(url, "São Paulo") == "Vila Nova Conceicao"

    def test_vila_madalena_zona_oeste(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-3-quartos-com-churrasqueira-vila-madalena-zona-oeste-sao-paulo-sp-136m2-id-2888782275/"
        assert _bairro_from_url(url, "São Paulo") == "Vila Madalena"

    def test_single_word_bairro(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-2-quartos-tatuape-zona-leste-sao-paulo-sp-65m2-id-123/"
        assert _bairro_from_url(url, "São Paulo") == "Tatuape"

    def test_bairro_without_zona(self):
        # Cidades menores sem zona-direção
        url = "https://www.zapimoveis.com.br/imovel/venda-casa-3-quartos-jardim-maia-guarulhos-sp-90m2-id-456/"
        assert _bairro_from_url(url, "Guarulhos") == "Jardim Maia"

    def test_returns_empty_for_non_imovel_url(self):
        assert _bairro_from_url("https://www.zapimoveis.com.br/busca/", "São Paulo") == ""

    def test_returns_empty_for_empty_url(self):
        assert _bairro_from_url("", "São Paulo") == ""

    def test_returns_empty_without_anchor(self):
        # URL sem zona nem cidade conhecida e sem cidade_slug
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-id-999/"
        assert _bairro_from_url(url) == ""

    def test_zone_norte(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-casa-4-quartos-santana-zona-norte-sao-paulo-sp-120m2-id-789/"
        assert _bairro_from_url(url, "São Paulo") == "Santana"

    def test_three_word_bairro(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-2-quartos-jardim-sao-luis-zona-sul-sao-paulo-sp-55m2-id-321/"
        assert _bairro_from_url(url, "São Paulo") == "Jardim Sao Luis"

    def test_title_case_result(self):
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-1-quarto-mooca-zona-leste-sao-paulo-sp-45m2-id-111/"
        result = _bairro_from_url(url, "São Paulo")
        assert result == result.title()

    def test_url_takes_precedence_over_wrong_title_bairro(self):
        """URL deve sobrescrever bairro errado do título (caso Vila Madalena vs Sumarezinho)."""
        url = "https://www.zapimoveis.com.br/imovel/venda-apartamento-3-quartos-com-churrasqueira-vila-madalena-zona-oeste-sao-paulo-sp-136m2-id-2888782275/"
        # Título diz "Sumarezinho" mas URL diz "vila-madalena"
        titulo = "Apartamento em Sumarezinho, São Paulo"
        from src.external.zapimoveis import _bairro_from_name
        bairro_titulo = _bairro_from_name(titulo, "São Paulo")
        bairro_url = _bairro_from_url(url, "São Paulo")
        # A URL deve ganhar
        resultado_final = bairro_url or bairro_titulo
        assert resultado_final == "Vila Madalena"


# ─── _bairro_from_name ────────────────────────────────────────────────────────

class TestBairroFromName:
    def test_extracts_bairro(self):
        result = _bairro_from_name("Apartamento em Vila Prudente, São Paulo", "São Paulo")
        assert result == "Vila Prudente"

    def test_no_match_returns_empty(self):
        result = _bairro_from_name("Apartamento moderno", "São Paulo")
        assert result == ""

    def test_case_insensitive(self):
        result = _bairro_from_name("Apto EM Centro, Guarulhos", "Guarulhos")
        assert result == "Centro"

    def test_multi_word_bairro(self):
        result = _bairro_from_name("Casa em Jardim Maia, Guarulhos", "Guarulhos")
        assert result == "Jardim Maia"

    def test_wrong_city_no_match(self):
        result = _bairro_from_name("Apto em Centro, São Paulo", "Guarulhos")
        assert result == ""


# ─── _clean_text ──────────────────────────────────────────────────────────────

class TestCleanText:
    def test_strips_whitespace(self):
        assert _clean_text("  hello  ") == "hello"

    def test_normal_text_unchanged(self):
        text = "Apartamento 3 quartos"
        assert _clean_text(text) == text

    def test_m2_space_comma_restored(self):
        result = _clean_text("75m ,")
        assert "75m²" in result

    def test_m2_space_restored(self):
        result = _clean_text("80m ")
        assert "80m²" in result

    def test_empty_string(self):
        assert _clean_text("") == ""


# ─── _fetch_html ──────────────────────────────────────────────────────────────

class TestFetchHtml:
    def test_returns_html_on_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"<html>content</html>"
        with patch("src.external.zapimoveis.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result == "<html>content</html>"

    def test_returns_none_on_403(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        with patch("src.external.zapimoveis.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_404(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("src.external.zapimoveis.requests.get", return_value=mock_resp):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_connection_error(self):
        with patch("src.external.zapimoveis.requests.get", side_effect=ConnectionError()):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_returns_none_on_timeout(self):
        import requests as req
        with patch("src.external.zapimoveis.requests.get", side_effect=req.Timeout()):
            result = _fetch_html("https://example.com")
        assert result is None

    def test_uses_user_agent_header(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"ok"
        with patch("src.external.zapimoveis.requests.get", return_value=mock_resp) as mock_get:
            _fetch_html("https://example.com")
        headers = mock_get.call_args[1]["headers"]
        assert "User-Agent" in headers


# ─── _extrair_items_ld ────────────────────────────────────────────────────────

class TestExtrairItemsLd:
    def test_extracts_item_list(self):
        html = _make_html(VALID_JSON_LD)
        items = _extrair_items_ld(html)
        assert len(items) == 1
        assert items[0] == VALID_ITEM

    def test_returns_empty_for_wrong_type(self):
        data = {"@type": "Product", "itemListElement": [VALID_ITEM]}
        html = _make_html(data)
        assert _extrair_items_ld(html) == []

    def test_returns_empty_for_no_script(self):
        assert _extrair_items_ld("<html><body>no script</body></html>") == []

    def test_skips_malformed_json(self):
        html = '<html><script type="application/ld+json">INVALID_JSON{}</script></html>'
        assert _extrair_items_ld(html) == []

    def test_multiple_scripts_picks_item_list(self):
        other = {"@type": "BreadcrumbList", "itemListElement": []}
        html = (
            f'<script type="application/ld+json">{json.dumps(other)}</script>'
            f'<script type="application/ld+json">{json.dumps(VALID_JSON_LD)}</script>'
        )
        items = _extrair_items_ld(html)
        assert len(items) == 1

    def test_returns_empty_if_no_item_list_element_key(self):
        data = {"@type": "ItemList"}
        html = _make_html(data)
        assert _extrair_items_ld(html) == []

    def test_handles_multiple_items(self):
        ld = {"@type": "ItemList", "itemListElement": [VALID_ITEM, VALID_ITEM]}
        html = _make_html(ld)
        items = _extrair_items_ld(html)
        assert len(items) == 2


# ─── _normalizar_ld ───────────────────────────────────────────────────────────

class TestNormalizarLd:
    def test_full_entry_fields(self):
        result = _normalizar_ld(VALID_ITEM, "zap")
        assert result is not None
        assert result["id"] == "zap_123"
        assert result["preco"] == 450000.0
        assert result["area"] == 75.0
        assert result["tipo"] == "apartment"
        assert result["quartos"] == 2
        assert result["banheiros"] == 1
        assert result["cidade"] == "são paulo"
        assert result["estado"] == "São Paulo"
        assert result["bairro"] == "Vila Prudente"
        assert result["vagas_garagem"] == 2
        assert result["thumbnail"] == "https://cdn.example.com/img.jpg"
        assert result["fonte"] == "zap"
        assert result["latitude"] is None
        assert result["longitude"] is None

    def test_none_item_returns_none(self):
        assert _normalizar_ld({}, "zap") is None

    def test_area_floored_at_18_when_small(self):
        entry = {"item": {**VALID_ITEM["item"], "floorSize": {"value": "10"}}}
        result = _normalizar_ld(entry, "zap")
        assert result["area"] == 18.0

    def test_area_floored_at_18_when_zero(self):
        entry = {"item": {**VALID_ITEM["item"], "floorSize": None}}
        result = _normalizar_ld(entry, "zap")
        assert result["area"] == 18.0

    def test_area_invalid_string_defaults_18(self):
        entry = {"item": {**VALID_ITEM["item"], "floorSize": {"value": "consulte"}}}
        result = _normalizar_ld(entry, "zap")
        assert result["area"] == 18.0

    def test_missing_price_defaults_zero(self):
        item = {**VALID_ITEM["item"], "offers": {}}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["preco"] == 0.0

    def test_invalid_price_string_defaults_zero(self):
        item = {**VALID_ITEM["item"], "offers": {"price": "consulte"}}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["preco"] == 0.0

    def test_house_type(self):
        item = {**VALID_ITEM["item"], "@type": "House"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["tipo"] == "house"

    def test_land_type(self):
        item = {**VALID_ITEM["item"], "@type": "LandParcel"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["tipo"] == "land"

    def test_commercial_type(self):
        item = {**VALID_ITEM["item"], "@type": "Place"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["tipo"] == "commercial"

    def test_unknown_type_defaults_apartment(self):
        item = {**VALID_ITEM["item"], "@type": "UnknownType"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["tipo"] == "apartment"

    def test_rental_detected_from_url(self):
        item = {**VALID_ITEM["item"], "url": "https://zapimoveis.com.br/aluguel/apartamentos/sp/"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["negocio"] == "RENTAL"

    def test_sale_detected_from_url(self):
        result = _normalizar_ld(VALID_ITEM, "zap")
        assert result["negocio"] == "SALE"

    def test_no_image_empty_thumbnail(self):
        item = {**VALID_ITEM["item"], "image": []}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["thumbnail"] == ""

    def test_flat_entry_without_item_key(self):
        flat_item = {**VALID_ITEM["item"]}
        result = _normalizar_ld(flat_item, "zap")
        assert result is not None

    def test_condominium_fee_parsed(self):
        offers = {
            "price": "300000",
            "additionalProperty": {"name": "Condominium Fee", "value": "500"}
        }
        item = {**VALID_ITEM["item"], "offers": offers}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["condominio"] == 500.0

    def test_condominium_fee_not_parsed_when_wrong_name(self):
        offers = {
            "price": "300000",
            "additionalProperty": {"name": "Other Fee", "value": "500"}
        }
        item = {**VALID_ITEM["item"], "offers": offers}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["condominio"] == 0.0

    def test_description_truncated_at_500(self):
        long_desc = "x" * 700
        item = {**VALID_ITEM["item"], "description": long_desc}
        result = _normalizar_ld({"item": item}, "zap")
        assert len(result["descricao"]) <= 500

    def test_estado_resolved_from_uf_map(self):
        result = _normalizar_ld(VALID_ITEM, "zap")
        assert result["estado"] == "São Paulo"

    def test_vagas_parsed_from_name(self):
        item = {**VALID_ITEM["item"], "name": "Apto com 3 vagas em Centro, SP"}
        result = _normalizar_ld({"item": item}, "zap")
        assert result["vagas_garagem"] == 3

    def test_iptu_defaults_zero(self):
        result = _normalizar_ld(VALID_ITEM, "zap")
        assert result["iptu"] == 0.0

    def test_suites_defaults_zero(self):
        result = _normalizar_ld(VALID_ITEM, "zap")
        assert result["suites"] == 0


# ─── URL builders ─────────────────────────────────────────────────────────────

class TestUrlZap:
    def test_default_url_contains_venda(self):
        url = _url_zap("São Paulo", "São Paulo")
        assert "venda" in url

    def test_default_url_contains_apartamentos(self):
        url = _url_zap("São Paulo", "São Paulo")
        assert "apartamentos" in url

    def test_url_contains_cidade_slug(self):
        url = _url_zap("São Paulo", "São Paulo")
        assert "sao-paulo" in url

    def test_url_contains_uf(self):
        url = _url_zap("São Paulo", "São Paulo")
        assert "sp+" in url

    def test_pagina_2_adds_param(self):
        url = _url_zap("Guarulhos", "São Paulo", pagina=2)
        assert "pagina=2" in url

    def test_pagina_1_no_query_string(self):
        url = _url_zap("Guarulhos", "São Paulo", pagina=1)
        assert "pagina" not in url

    def test_rental_uses_aluguel(self):
        url = _url_zap("Guarulhos", "São Paulo", negocio="RENTAL")
        assert "aluguel" in url

    def test_house_type(self):
        url = _url_zap("Guarulhos", "São Paulo", tipo="house")
        assert "casas" in url

    def test_land_type(self):
        url = _url_zap("Guarulhos", "São Paulo", tipo="land")
        assert "terrenos" in url

    def test_commercial_type(self):
        url = _url_zap("Guarulhos", "São Paulo", tipo="commercial")
        assert "comercial" in url

    def test_unknown_estado_defaults_sp(self):
        url = _url_zap("Cidade", "Estado Desconhecido")
        assert "/sp+" in url

    def test_minas_gerais_uses_mg(self):
        url = _url_zap("Belo Horizonte", "Minas Gerais")
        assert "/mg+" in url


class TestUrlViva:
    def test_default_url_contains_vivareal(self):
        url = _url_viva("São Paulo", "São Paulo")
        assert "vivareal.com.br" in url

    def test_default_uses_apartamento_residencial(self):
        url = _url_viva("São Paulo", "São Paulo")
        assert "apartamento_residencial" in url

    def test_pagina_2_adds_param(self):
        url = _url_viva("Guarulhos", "São Paulo", pagina=2)
        assert "pagina=2" in url

    def test_pagina_1_no_query(self):
        url = _url_viva("Guarulhos", "São Paulo", pagina=1)
        assert "pagina" not in url

    def test_house_uses_casa_residencial(self):
        url = _url_viva("Guarulhos", "São Paulo", tipo="house")
        assert "casa_residencial" in url

    def test_land_uses_terreno(self):
        url = _url_viva("Guarulhos", "São Paulo", tipo="land")
        assert "terreno_condominio" in url

    def test_commercial_uses_sala_comercial(self):
        url = _url_viva("Guarulhos", "São Paulo", tipo="commercial")
        assert "sala_comercial" in url

    def test_rental_uses_aluguel(self):
        url = _url_viva("Guarulhos", "São Paulo", negocio="RENTAL")
        assert "aluguel" in url


# ─── buscar ───────────────────────────────────────────────────────────────────

class TestBuscar:
    def test_returns_results_from_html(self):
        html = _make_multi_item_html(3)
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", limit=10, tipos=["apartment"])
        assert len(results) == 3

    def test_respects_limit(self):
        html = _make_multi_item_html(10)
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", limit=3, tipos=["apartment"])
        assert len(results) <= 3

    def test_returns_empty_on_no_html(self):
        with patch("src.external.zapimoveis._fetch_html", return_value=None), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo")
        assert results == []

    def test_quartos_min_filter_excludes(self):
        html = _make_multi_item_html(5, quartos="2")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", quartos_min=3, tipos=["apartment"])
        assert results == []

    def test_quartos_zero_not_filtered(self):
        # quartos=0 (unknown) should NOT be filtered even if quartos_min > 0
        html = _make_multi_item_html(2, quartos="0")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", quartos_min=2, tipos=["apartment"])
        assert len(results) == 2

    def test_preco_max_filter_excludes(self):
        html = _make_multi_item_html(3, preco="800000")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", preco_max=500000, tipos=["apartment"])
        assert results == []

    def test_preco_min_filter_excludes(self):
        html = _make_multi_item_html(3, preco="100000")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", preco_min=300000, tipos=["apartment"])
        assert results == []

    def test_area_min_filter_excludes(self):
        html = _make_multi_item_html(3, area="40")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", area_min=100, tipos=["apartment"])
        assert results == []

    def test_preco_zero_not_filtered_by_preco_max(self):
        # preco=0 (unknown) should NOT be filtered by preco_max
        html = _make_multi_item_html(2, preco="0")
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", preco_max=500000, tipos=["apartment"])
        assert len(results) == 2

    def test_vivareal_portal_calls_viva_url(self):
        html = _make_multi_item_html(2)
        with patch("src.external.zapimoveis._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.zapimoveis.time.sleep"):
            buscar("Guarulhos", "São Paulo", portal="vivareal", tipos=["apartment"])
        url_called = mock_fetch.call_args[0][0]
        assert "vivareal.com.br" in url_called

    def test_zap_portal_calls_zap_url(self):
        html = _make_multi_item_html(2)
        with patch("src.external.zapimoveis._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.zapimoveis.time.sleep"):
            buscar("Guarulhos", "São Paulo", portal="zap", tipos=["apartment"])
        url_called = mock_fetch.call_args[0][0]
        assert "zapimoveis.com.br" in url_called

    def test_default_tipos_fetches_apartment_and_house(self):
        with patch("src.external.zapimoveis._fetch_html", return_value=None) as mock_fetch, \
             patch("src.external.zapimoveis.time.sleep"):
            buscar("Guarulhos", "São Paulo")
        # Should fetch for both apartment and house (2 URL calls)
        assert mock_fetch.call_count == 2

    def test_sleep_called_between_tipos(self):
        html = _make_multi_item_html(1)
        with patch("src.external.zapimoveis._fetch_html", return_value=html), \
             patch("src.external.zapimoveis.time.sleep") as mock_sleep:
            buscar("Guarulhos", "São Paulo", tipos=["apartment", "house"])
        assert mock_sleep.called

    def test_buscar_zap_passes_portal_zap(self):
        with patch("src.external.zapimoveis.buscar", return_value=[]) as mock_buscar:
            buscar_zap("Guarulhos", "São Paulo")
        _, kwargs = mock_buscar.call_args
        assert kwargs.get("portal") == "zap"

    def test_buscar_vivareal_passes_portal_vivareal(self):
        with patch("src.external.zapimoveis.buscar", return_value=[]) as mock_buscar:
            buscar_vivareal("Guarulhos", "São Paulo")
        _, kwargs = mock_buscar.call_args
        assert kwargs.get("portal") == "vivareal"

    def test_stops_fetching_when_limit_reached(self):
        html = _make_multi_item_html(20)
        with patch("src.external.zapimoveis._fetch_html", return_value=html) as mock_fetch, \
             patch("src.external.zapimoveis.time.sleep"):
            results = buscar("Guarulhos", "São Paulo", limit=5, tipos=["apartment", "house"])
        # Should stop before fetching house type since apartment alone fills limit
        assert mock_fetch.call_count == 1
