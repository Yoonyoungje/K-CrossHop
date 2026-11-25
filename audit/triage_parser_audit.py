#!/usr/bin/env python3
"""parser audit 전량 triage — 각 홉을 응답 구조로 분류해 판정 난이도 분포를 본다.

읽는 법: 이 분류는 **난이도**이지 라벨의 정오가 아니다. AMBIG는 "틀린 라벨"이 아니라
답과 유보가 한 응답에 공존해 기권 경계가 본질적으로 모호한 구간이며, 규칙 기반 파서와
사람 어노테이터가 같은 자리에서 갈린다(2026-07-25 수동 검수 확인). 이 출력으로
재라벨 대상을 고르지 말 것 — 실제로 그렇게 썼다가 오판한 전례가 있다.

버킷
  CLEAR_ABSTAIN : 답 후보가 전혀 없음 (기권 키워드 + 나열 없음 + 실질 내용 없음)
  CLEAR_ANSWER  : 기권 키워드 없음 (단일 답변)
  AMBIG_ENUM    : 여러 대상을 나열 → 물은 대상의 슬롯을 봐야 판정 가능
  AMBIG_HEDGE   : 기권 키워드 + 그 밖에 실질 내용 존재 (답+유보 후보)
"""
import csv, glob, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # repo root (utils 패키지)
from utils.answer_parser import normalize_entity, _check_abstain

QA = str(ROOT / 'data/dataset/qa_pairs.jsonl')
SHEETS = str(ROOT / 'reports/parser_audit/[an]*.csv')

ABSENCE = re.compile(
    r'모르겠|모르겁|알 수 없|확인할 수 없|확인되지 않|정보(가)?\s*없|명시되(어 있지|지)?\s*않|'
    r'언급되(어 있지|지)?\s*않|제공되지\s*않|나와 있지\s*않|미상|찾을 수 없|'
    r'not\s+(mentioned|specified|provided|available|found)|no\s+information|unknown|n/a', re.I)
BULLET = re.compile(r'(?m)^\s*[-*•]\s*\S')
PAIR   = re.compile(r'[^\n,;]{2,40}?\s*[-–—:]\s*[^\n,;]{1,60}')


def asked_entities(qa_path: str) -> dict:
    """(qid, hop) → 물은 대상 표기 후보 [ko, en]."""
    out = {}
    for line in open(qa_path, encoding='utf-8'):
        qa = json.loads(line)
        m = re.match(r'^(.+?)에 대해 검색하라', qa.get('question', ''))
        subj = m.group(1).strip() if m else ''
        for h in qa['hops']:
            if h['hop'] == 0:
                cands = [subj]
            else:
                cands = [h.get('sub_q_bridge_label', ''), h.get('sub_q_en_bridge_label', '')]
            out[(qa['qid'], h['hop'])] = [c for c in cands if c]
    return out


def slot_state(text: str, cands: list) -> str:
    """물은 대상이 텍스트에서 어떤 상태인가: answered / negated / absent."""
    nt = normalize_entity(text)
    for c in cands:
        nc = normalize_entity(c)
        if not nc or nc not in nt:
            continue
        # 원문에서 위치를 찾아 뒤 60자에 부재 표현이 있는지 본다
        for m in re.finditer(re.escape(c[:12]), text):
            tail = text[m.end(): m.end() + 60]
            if ABSENCE.search(tail.split('\n')[0] if '\n' in tail[:40] else tail):
                return 'negated'
        return 'answered'
    return 'absent'


def enumerated(text: str) -> bool:
    if len(BULLET.findall(text)) >= 2:
        return True
    if len(PAIR.findall(text)) >= 2:
        return True
    return len(ABSENCE.findall(text)) >= 2


def residual_content(text: str) -> bool:
    """기권 표현을 걷어내고도 실질 내용이 남는가 (답+유보 후보)."""
    stripped = ABSENCE.sub(' ', text)
    stripped = re.sub(r'[()\[\]{}“”"\'*·,\.\n\s]+', ' ', stripped)
    stripped = re.sub(r'답변\s*:?|참고\s*:?|문서(에|에서|에는)?|주어진|제공된|정확한|구체적인|'
                      r'하지만|그러나|다만|따라서|정보|내용|질문|위치|대한|이는|것은|수|있|없',
                      ' ', stripped)
    return len(stripped.strip()) >= 12


def bucket(text: str, cands: list) -> tuple:
    text = (text or '').strip()
    hedge = _check_abstain(text)
    enum = enumerated(text)
    slot = slot_state(text, cands)
    if enum:
        return 'AMBIG_ENUM', slot
    if not hedge:
        return 'CLEAR_ANSWER', slot
    if residual_content(text):
        return 'AMBIG_HEDGE', slot
    return 'CLEAR_ABSTAIN', slot


def main():
    ae = asked_entities(QA)
    rows = []
    for f in sorted(glob.glob(SHEETS)):
        for r in csv.DictReader(open(f, encoding='utf-8')):
            b, slot = bucket(r['generated_text'], ae.get((r['qid'], int(r['hop'])), []))
            r['_bucket'], r['_slot'], r['_csv'] = b, slot, f
            rows.append(r)
    json.dump([{k: v for k, v in r.items()} for r in rows],
              open(ROOT / 'reports/parser_audit/triage_rows.json',
                   'w', encoding='utf-8'), ensure_ascii=False)
    print(f"전체 {len(rows)}행\n")
    t = defaultdict(Counter)
    for r in rows:
        t[(r['parser_class'], r['stratum'])][r['_bucket']] += 1
    cols = ['CLEAR_ABSTAIN', 'CLEAR_ANSWER', 'AMBIG_ENUM', 'AMBIG_HEDGE']
    print(f"{'class':<11}{'stratum':<17}" + ''.join(f"{c:>15}" for c in cols) + f"{'계':>6}")
    for k in sorted(t):
        row = t[k]
        print(f"{k[0]:<11}{k[1]:<17}" + ''.join(f"{row[c]:>15}" for c in cols)
              + f"{sum(row.values()):>6}")
    tot = Counter()
    for k in t: tot.update(t[k])
    print(f"{'계':<28}" + ''.join(f"{tot[c]:>15}" for c in cols) + f"{sum(tot.values()):>6}")
    amb = tot['AMBIG_ENUM'] + tot['AMBIG_HEDGE']
    print(f"\n판정 애매 = {amb} / {len(rows)} = {amb/len(rows):.1%}")


if __name__ == '__main__':
    main()
