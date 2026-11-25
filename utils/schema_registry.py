"""
utils/schema_registry.py
K-CrossHop Schema Registry

각 schema는 2-hop Wikidata property chain을 정의한다.
factory를 통해 a/b/c/d 4개 pattern으로 인스턴스화된다.

필드:
  path:         [P1, P2] — 두 hop의 property
  subj_filter:  subject에 적용할 추가 SPARQL (type constraint 등)
  mid_filter:   bridge에 적용할 추가 SPARQL
  obj_filter:   answer에 적용할 추가 SPARQL
  patterns:     지원 pattern 목록 (a/b/c/d)
  limit:        SPARQL LIMIT (raw 수집 안전 상한 — sampling cap 아님)
  priority:     1=Core, 2=Diversity, 3=High-risk/Exploratory
  domain:       지식 도메인 레이블 (통계용)
  main_cap:     최종 Main-1000 sampling 시 schema별 상한 (수집 단계에서 미적용)
"""

SCHEMAS: dict[str, dict] = {

    # ── Biography / Person ────────────────────────────────────────────────────

    "person_employer_admin": {
        "path": ["P108", "P131"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1500, "priority": 1, "domain": "biography", "main_cap": 50,
    },
    "person_edu_admin": {
        # 영어권 대학(P69) → 소재 행정구역(P131)
        "path": ["P69", "P131"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "mid_filter": "?mid wdt:P17 ?_c . VALUES ?_c { wd:Q30 wd:Q145 wd:Q16 wd:Q408 }",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 1, "domain": "biography", "main_cap": 40,
    },
    "person_affiliation_country": {
        # 소속 단체(P463) → 단체 국가(P17)
        "path": ["P463", "P17"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "biography", "main_cap": 35,
    },
    "person_award_org": {
        # 수상 이력(P166) → 시상기관(P1027)
        "path": ["P166", "P1027"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "award", "main_cap": 35,
    },
    "person_employer_country": {
        # 근무 기관(P108) → 기관 국가(P17) [Core: P131 대신 P17]
        "path": ["P108", "P17"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 1, "domain": "biography", "main_cap": 40,
    },
    "person_educated_country": {
        # 졸업 대학(P69) → 대학 소재 국가(P17)
        "path": ["P69", "P17"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 1, "domain": "biography", "main_cap": 40,
    },
    "person_position_jurisdiction": {
        # 직위(P39) → 관할 지역(P1001)
        "path": ["P39", "P1001"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["b", "d"],
        "limit": 500, "priority": 2, "domain": "politics", "main_cap": 25,
    },

    # ── Film ──────────────────────────────────────────────────────────────────

    "film_director_country": {
        # 감독(P57) → 감독 국적(P27)
        "path": ["P57", "P27"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 2000, "priority": 1, "domain": "film", "main_cap": 50,
    },
    "film_director_birthplace": {
        # 감독(P57) → 감독 출생지(P19)
        "path": ["P57", "P19"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 2000, "priority": 1, "domain": "film", "main_cap": 45,
    },
    "film_director_award": {
        # 감독(P57) → 감독 수상 이력(P166) — non-location
        "path": ["P57", "P166"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1500, "priority": 2, "domain": "film", "main_cap": 40,
    },
    "film_director_educated_at": {
        # 감독(P57) → 감독 졸업 학교(P69) — non-location
        "path": ["P57", "P69"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "film", "main_cap": 35,
    },
    "film_actor_country": {
        # 출연 배우(P161) → 배우 국적(P27)
        "path": ["P161", "P27"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 2000, "priority": 1, "domain": "film", "main_cap": 45,
    },
    "film_actor_birthplace": {
        # 출연 배우(P161) → 배우 출생지(P19)
        "path": ["P161", "P19"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 2000, "priority": 1, "domain": "film", "main_cap": 45,
    },
    "film_actor_award": {
        # 출연 배우(P161) → 배우 수상 이력(P166) — non-location
        "path": ["P161", "P166"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1500, "priority": 2, "domain": "film", "main_cap": 40,
    },
    "film_production_company_country": {
        # 제작사(P272) → 제작사 국가(P17)
        "path": ["P272", "P17"],
        "patterns": ["b", "d"],
        "limit": 1500, "priority": 1, "domain": "film", "main_cap": 40,
    },
    "film_distributor_country": {
        # 배급사(P750) → 배급사 국가(P17)
        "path": ["P750", "P17"],
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "film", "main_cap": 30,
    },
    "film_original_country": {
        # 제작국(P495) → 대륙(P30)
        "path": ["P495", "P30"],
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "film", "main_cap": 25,
    },

    # ── Sports ────────────────────────────────────────────────────────────────

    "athlete_team_country": {
        # 소속 팀(P54) → 팀 국가(P17)
        "path": ["P54", "P17"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "obj_filter": "MINUS { ?mid wdt:P17 wd:Q884 . }",
        "patterns": ["b", "d"],
        "limit": 1500, "priority": 1, "domain": "sports", "main_cap": 45,
    },
    "team_league_country": {
        # 소속 리그(P118) → 리그 국가(P17)
        "path": ["P118", "P17"],
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "sports", "main_cap": 35,
    },
    "athlete_event_venue": {
        # 참가 대회(P1344) → 개최지(P276)
        "path": ["P1344", "P276"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["b", "d"],
        "limit": 1500, "priority": 1, "domain": "sports", "main_cap": 40,
    },
    "athlete_event_country": {
        # 참가 대회(P1344) → 개최 국가(P17) [Core]
        "path": ["P1344", "P17"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "mid_filter": "MINUS { ?mid wdt:P17 wd:Q884 . }",
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 1, "domain": "sports", "main_cap": 35,
    },
    "athlete_team_league": {
        # 소속 팀(P54) → 팀 리그(P118) [해외 리그, b/d]
        "path": ["P54", "P118"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "obj_filter": "MINUS { ?obj wdt:P17 wd:Q884 . }",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "sports", "main_cap": 35,
    },
    "athlete_team_league_domestic": {
        # 소속 팀(P54) → 팀 리그(P118) [국내외 리그 포함, a/c — non-location]
        "path": ["P54", "P118"],
        "subj_filter": "?subj wdt:P27 wd:Q884 ; wdt:P31 wd:Q5 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "sports", "main_cap": 35,
    },
    "team_home_venue_city": {
        # 홈 경기장(P115) → 경기장 행정구역(P131)
        "path": ["P115", "P131"],
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "sports", "main_cap": 30,
    },
    "event_venue_country": {
        # 개최지(P276) → 개최지 국가(P17)
        "path": ["P276", "P17"],
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "event", "main_cap": 30,
    },

    # ── Organization / Company ────────────────────────────────────────────────

    "org_hq_country": {
        # 본사(P159) → 국가(P17)
        "path": ["P159", "P17"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 2000, "priority": 1, "domain": "org", "main_cap": 50,
    },
    "company_founder_country": {
        # 설립자(P112) → 설립자 국적(P27)
        "path": ["P112", "P27"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "org", "main_cap": 35,
    },
    "company_founder_award": {
        # 설립자(P112) → 설립자 수상 이력(P166) — non-location
        "path": ["P112", "P166"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "org", "main_cap": 30,
    },
    "company_founder_educated_at": {
        # 설립자(P112) → 설립자 졸업 학교(P69) — non-location
        "path": ["P112", "P69"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "org", "main_cap": 35,
    },
    "company_parent_hq": {
        # 모기업(P749) → 모기업 본사(P159)
        "path": ["P749", "P159"],
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "org", "main_cap": 30,
    },
    "org_hq_admin": {
        # 본사(P159) → 행정구역(P131) [Core: org 단위 location answer]
        "path": ["P159", "P131"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 1, "domain": "org", "main_cap": 30,
    },
    "org_member_country": {
        # 소속 단체(P463) → 단체 국가(P17)
        "path": ["P463", "P17"],
        "patterns": ["b", "d"],
        "limit": 600, "priority": 2, "domain": "org", "main_cap": 25,
    },
    "company_founder_birthplace": {
        # 설립자(P112) → 설립자 출생지(P19)
        "path": ["P112", "P19"],
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "org", "main_cap": 35,
    },
    "newspaper_publisher_country": {
        # 출판사(P123) → 출판사 국가(P17) [subject=newspaper]
        "path": ["P123", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q11032 .",
        "patterns": ["b", "d"],
        "limit": 500, "priority": 2, "domain": "media", "main_cap": 30,
    },
    "newspaper_publisher_founder": {
        # 출판사(P123) → 출판사 설립자(P112) [subject=newspaper] — non-location
        "path": ["P123", "P112"],
        "subj_filter": "?subj wdt:P31 wd:Q11032 .",
        "patterns": ["b", "c", "d"],
        "limit": 500, "priority": 2, "domain": "media", "main_cap": 25,
    },

    # ── Tech / Software ───────────────────────────────────────────────────────

    "software_developer_hq": {
        # 개발사(P178) → 개발사 본사(P159) [subject=software]
        "path": ["P178", "P159"],
        "subj_filter": "?subj wdt:P31 wd:Q7397 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1500, "priority": 1, "domain": "tech", "main_cap": 40,
    },
    "game_developer_hq": {
        # 개발사(P178) → 개발사 본사(P159) [subject=video game]
        "path": ["P178", "P159"],
        "subj_filter": "?subj wdt:P31 wd:Q7889 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1500, "priority": 1, "domain": "tech", "main_cap": 40,
    },
    "software_lang_designer": {
        # 사용 언어(P277) → 언어 설계자(P287) [subject=software/app]
        "path": ["P277", "P287"],
        "patterns": ["d"],
        "limit": 800, "priority": 1, "domain": "tech", "main_cap": 40,
    },
    "software_developer_founder": {
        # 개발사(P178) → 개발사 설립자(P112) [subject=software] — non-location
        "path": ["P178", "P112"],
        "subj_filter": "?subj wdt:P31 wd:Q7397 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "tech", "main_cap": 35,
    },
    "game_developer_founder": {
        # 개발사(P178) → 개발사 설립자(P112) [subject=video game] — non-location
        "path": ["P178", "P112"],
        "subj_filter": "?subj wdt:P31 wd:Q7889 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "tech", "main_cap": 35,
    },
    "game_publisher_founder": {
        # 배급사(P123) → 배급사 설립자(P112) [subject=video game] — non-location
        "path": ["P123", "P112"],
        "subj_filter": "?subj wdt:P31 wd:Q7889 .",
        "patterns": ["b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "tech", "main_cap": 30,
    },
    "software_developer_country": {
        # 개발사(P178) → 개발사 국가(P17)
        "path": ["P178", "P17"],
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "tech", "main_cap": 35,
    },
    "game_publisher_country": {
        # 배급사(P123) → 배급사 국가(P17) [subject=video game]
        "path": ["P123", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q7889 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "tech", "main_cap": 35,
    },
    "programming_language_designer_birthplace": {
        # 설계자(P287) → 설계자 출생지(P19) [subject=programming language]
        "path": ["P287", "P19"],
        "subj_filter": "?subj wdt:P31 wd:Q9143 .",
        "patterns": ["c", "d"],
        "limit": 400, "priority": 2, "domain": "tech", "main_cap": 25,
    },
    "programming_language_designer_country": {
        # 설계자(P287) → 설계자 국적(P27) [subject=programming language]
        "path": ["P287", "P27"],
        "subj_filter": "?subj wdt:P31 wd:Q9143 .",
        "patterns": ["b", "d"],
        "limit": 400, "priority": 2, "domain": "tech", "main_cap": 25,
    },
    "programming_language_designer_educated_at": {
        # 설계자(P287) → 설계자 졸업 학교(P69) [subject=programming language] — non-location
        "path": ["P287", "P69"],
        "subj_filter": "?subj wdt:P31 wd:Q9143 .",
        "patterns": ["b", "c", "d"],
        "limit": 400, "priority": 2, "domain": "tech", "main_cap": 25,
    },
    "product_manufacturer_founder": {
        # 제조사(P176) → 제조사 설립자(P112) — non-location
        "path": ["P176", "P112"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "product", "main_cap": 30,
    },
    "product_manufacturer_country": {
        # 제조사(P176) → 제조사 국가(P17)
        "path": ["P176", "P17"],
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "product", "main_cap": 30,
    },
    "invention_inventor_country": {
        # 발명자(P61) → 발명자 국적(P27)
        "path": ["P61", "P27"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 600, "priority": 2, "domain": "science", "main_cap": 25,
    },

    # ── Literature ────────────────────────────────────────────────────────────

    "book_author_country": {
        # 저자(P50) → 저자 국적(P27)
        "path": ["P50", "P27"],
        "subj_filter": "?subj wdt:P31 wd:Q7725634 .",
        "patterns": ["b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "literature", "main_cap": 35,
    },
    "book_author_award": {
        # 저자(P50) → 저자 수상 이력(P166) — non-location
        "path": ["P50", "P166"],
        "subj_filter": "?subj wdt:P31 wd:Q7725634 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 1000, "priority": 2, "domain": "literature", "main_cap": 35,
    },
    "book_publisher_country": {
        # 출판사(P123) → 출판사 국가(P17)
        "path": ["P123", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q7725634 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "literature", "main_cap": 30,
    },
    "book_author_birthplace": {
        # 저자(P50) → 저자 출생지(P19)
        "path": ["P50", "P19"],
        "subj_filter": "?subj wdt:P31 wd:Q7725634 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "literature", "main_cap": 30,
    },

    # ── Music ─────────────────────────────────────────────────────────────────

    "album_record_label_country": {
        # 레이블(P264) → 레이블 국가(P17) [subject=album]
        "path": ["P264", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q482994 .",
        "patterns": ["b", "d"],
        "limit": 1000, "priority": 2, "domain": "music", "main_cap": 30,
    },
    "song_performer_country": {
        # 가수(P175) → 가수 국적(P27) [subject=song]
        "path": ["P175", "P27"],
        "subj_filter": "?subj wdt:P31 wd:Q7366 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "music", "main_cap": 25,
    },
    "album_performer_country": {
        # 가수(P175) → 가수 국적(P27) [subject=album]
        "path": ["P175", "P27"],
        "subj_filter": "?subj wdt:P31 wd:Q482994 .",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "music", "main_cap": 30,
    },
    "album_performer_award": {
        # 가수(P175) → 가수 수상 이력(P166) [subject=album] — non-location
        "path": ["P175", "P166"],
        "subj_filter": "?subj wdt:P31 wd:Q482994 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "music", "main_cap": 30,
    },

    # ── Award ─────────────────────────────────────────────────────────────────

    "award_org_hq": {
        # 시상기관(P1027) → 기관 본사(P159)
        "path": ["P1027", "P159"],
        "patterns": ["c", "d"],
        "limit": 500, "priority": 2, "domain": "award", "main_cap": 30,
    },

    # ── Politics ──────────────────────────────────────────────────────────────

    "politician_party_country": {
        # 소속 정당(P102) → 정당 국가(P17)
        "path": ["P102", "P17"],
        "mid_filter": "MINUS { ?mid wdt:P17 wd:Q884 . }",
        "patterns": ["b", "d"],
        "limit": 800, "priority": 2, "domain": "politics", "main_cap": 20,
    },
    "party_hq_country": {
        # 본사(P159) → 국가(P17) [subject=political party]
        "path": ["P159", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q7278 .",
        "patterns": ["b", "d"],
        "limit": 500, "priority": 2, "domain": "politics", "main_cap": 20,
    },

    # ── Architecture / Infrastructure ─────────────────────────────────────────

    "building_architect_country": {
        # 건축가(P84) → 건축가 국적(P27)
        "path": ["P84", "P27"],
        "patterns": ["b", "c", "d"],
        "limit": 800, "priority": 2, "domain": "architecture", "main_cap": 25,
    },
    "building_architect_award": {
        # 건축가(P84) → 건축가 수상 이력(P166) — non-location
        "path": ["P84", "P166"],
        "patterns": ["a", "b", "c", "d"],
        "limit": 600, "priority": 2, "domain": "architecture", "main_cap": 25,
    },
    "airport_operator_country": {
        # 운영기관(P137) → 기관 국가(P17)
        "path": ["P137", "P17"],
        "subj_filter": "?subj wdt:P31 wd:Q1248784 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 500, "priority": 2, "domain": "infrastructure", "main_cap": 25,
    },

    # ── Education ─────────────────────────────────────────────────────────────

    "university_country_continent": {
        # 국가(P17) → 대륙(P30) [subject=university]
        "path": ["P17", "P30"],
        "subj_filter": "?subj wdt:P31 wd:Q3918 .",
        "patterns": ["b", "d"],
        "limit": 500, "priority": 2, "domain": "education", "main_cap": 20,
    },
    "university_founder_country": {
        # 설립자(P112) → 설립자 국적(P27) [subject=university]
        "path": ["P112", "P27"],
        "subj_filter": "?subj wdt:P31 wd:Q3918 .",
        "patterns": ["b", "d"],
        "limit": 500, "priority": 2, "domain": "education", "main_cap": 25,
    },
    "university_founder_educated_at": {
        # 설립자(P112) → 설립자 졸업 학교(P69) [subject=university] — non-location
        "path": ["P112", "P69"],
        "subj_filter": "?subj wdt:P31 wd:Q3918 .",
        "patterns": ["a", "b", "c", "d"],
        "limit": 500, "priority": 2, "domain": "education", "main_cap": 25,
    },
}
