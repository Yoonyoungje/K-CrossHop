"""
utils/sparql_factory.py
Schema + Pattern → SPARQL query 생성기

사용:
  from utils.sparql_factory import build_query
  from utils.schema_registry import SCHEMAS

  sparql_str = build_query(SCHEMAS["film_director_country"], "d", limit=2000)
"""

KO_WIKI = "https://ko.wikipedia.org/"
EN_WIKI  = "https://en.wikipedia.org/"

# 패턴별 (subject, bridge, answer)의 요구 Wikipedia 언어
# "ko" = schema:isPartOf <ko.wikipedia.org>
# "en" = schema:isPartOf <en.wikipedia.org>
PATTERN_LANGS: dict[str, tuple[str, str, str]] = {
    "a": ("ko", "ko", "ko"),
    "b": ("ko", "en", "ko"),
    "c": ("ko", "ko", "en"),
    "d": ("ko", "en", "en"),
}

_PREFIXES = """\
PREFIX wd:     <http://www.wikidata.org/entity/>
PREFIX wdt:    <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs:   <http://www.w3.org/2000/01/rdf-schema#>"""

_LABELS = """\
  OPTIONAL { ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }
  OPTIONAL { ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }
  OPTIONAL { ?mid  rdfs:label ?midLabelKo  . FILTER(LANG(?midLabelKo)  = "ko") }
  OPTIONAL { ?mid  rdfs:label ?midLabelEn  . FILTER(LANG(?midLabelEn)  = "en") }
  OPTIONAL { ?obj  rdfs:label ?objLabelKo  . FILTER(LANG(?objLabelKo)  = "ko") }
  OPTIONAL { ?obj  rdfs:label ?objLabelEn  . FILTER(LANG(?objLabelEn)  = "en") }"""


def _sitelink(var: str, wiki_var: str, lang: str) -> str:
    url = KO_WIKI if lang == "ko" else EN_WIKI
    return f"  ?{wiki_var} schema:about ?{var} ; schema:isPartOf <{url}> ."


def build_query(schema: dict, pattern: str, limit: int) -> str:
    """
    schema dict + pattern 문자열 → 실행 가능한 SPARQL SELECT 쿼리 반환.

    SPARQL row 변수명은 기존 utils/sparql.py 쿼리와 동일하게 유지:
      subj, wiki_subj, subjLabelKo, subjLabelEn
      mid,  wiki_mid,  midLabelKo,  midLabelEn
      obj,  wiki_obj,  objLabelKo,  objLabelEn
    """
    p1, p2 = schema["path"]
    subj_lang, mid_lang, obj_lang = PATTERN_LANGS[pattern]

    subj_sl = _sitelink("subj", "wiki_subj", subj_lang)
    mid_sl  = _sitelink("mid",  "wiki_mid",  mid_lang)
    obj_sl  = _sitelink("obj",  "wiki_obj",  obj_lang)

    subj_filter = schema.get("subj_filter", "")
    mid_filter  = schema.get("mid_filter",  "")
    obj_filter  = schema.get("obj_filter",  "")

    def _line(s: str) -> str:
        return f"\n  {s}" if s else ""

    return (
        f"{_PREFIXES}\n\n"
        f"SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn\n"
        f"                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn\n"
        f"                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{\n"
        f"{_line(subj_filter)}\n"
        f"{subj_sl}\n"
        f"\n"
        f"  ?subj wdt:{p1} ?mid .\n"
        f"{_line(mid_filter)}\n"
        f"{mid_sl}\n"
        f"\n"
        f"  ?mid wdt:{p2} ?obj .\n"
        f"{_line(obj_filter)}\n"
        f"{obj_sl}\n"
        f"\n"
        f"{_LABELS}\n"
        f"}}\n"
        f"LIMIT {limit}\n"
    )
