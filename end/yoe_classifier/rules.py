from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class RuleHit:
    label: int                    
    confidence: float             
    decided_by: str               
    evidence_snippet: str         
    evidence_source: str      
    min_yoe: Optional[float] = None
    max_yoe: Optional[float] = None


def yoe_to_label(yrs: float) -> int:
    
    if yrs <= 2:
        return 1
    if yrs <= 5:
        return 2
    if yrs <= 15:
        return 3
    return 4



_WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

_WORD_NUM_RE = "|".join(_WORD_NUM.keys())


def _parse_num(token: str) -> Optional[float]:
    if token is None:
        return None
    token = token.strip().lower()
    if not token:
        return None
    if token.isdigit():
        return float(token)
    if token in _WORD_NUM:
        return float(_WORD_NUM[token])
    return None


# Regex patterns


# YoE patterns
_EXP_WORD = (
    r"(?:experience|exp(?:erience)?\.?|expertise|background|"
    r"track\s+record|work\s+history|hands[-\s]?on)"
)

_DESCRIPTOR = (
    r"(?:[a-z][\w&/+\-]*[\s,;/]+){0,8}"
)

_PAREN_GAP = r"(?:\([^)]{1,40}\)\s*)?"

_BAD_TAIL = (
    r"(?:age|old|tenure\s+with|on\s+the\s+job|in\s+business|"
    r"in\s+operation|of\s+(?:trusted|combined|service|operations?|"
    r"history|leadership\s+at|excellence|growth|providing|delivering|"
    r"helping|serving|partnering|supporting|specializing|innovation|"
    r"success))"
)

_COMPANY_PREFIX = re.compile(
    r"(?:\bfor\s+(?:over|nearly|more\s+than|almost)?\s*$|"
    r"\bwith\s+(?:over|nearly|more\s+than|almost)?\s*$|"
    r"\bwe(?:'ve|\s+have)\s+been[^.]{0,40}$|"
    r"\bour\s+(?:company|firm|team|organization|practice|history)[^.]{0,40}$|"
    r"\bthe\s+(?:company|firm|organization|team)[^.]{0,40}$|"
    r"\bin\s+business\s+(?:for\s+)?$|"
    r"\bestablished\s+(?:over\s+|nearly\s+|more\s+than\s+)?$|"
    r"\bfounded\s+(?:over\s+|nearly\s+|more\s+than\s+)?$|"
    r"\b(?:celebrating|celebrate)[^.]{0,30}$)",
    re.IGNORECASE,
)

_RANGE_RE = re.compile(
    rf"(?P<lo>\d{{1,2}}|{_WORD_NUM_RE})\s*"
    rf"(?:-|\u2013|\u2014|to|\bthru\b)\s*"
    rf"(?P<hi>\d{{1,2}}|{_WORD_NUM_RE})\s*"
    rf"\+?\s*{_PAREN_GAP}(?:years?|yrs?\.?)"
    rf"(?!\s+(?:of\s+)?{_BAD_TAIL})"
    rf"\s*(?:of\s+)?{_DESCRIPTOR}{_EXP_WORD}",
    re.IGNORECASE,
)

_PLUS_RE = re.compile(
    rf"(?<![\w/$])"
    rf"(?P<n>\d{{1,2}}|{_WORD_NUM_RE})\s*\+\s*"
    rf"{_PAREN_GAP}(?:years?|yrs?\.?)"
    rf"(?!\s+(?:of\s+)?{_BAD_TAIL})"
    rf"\s*(?:of\s+)?{_DESCRIPTOR}{_EXP_WORD}",
    re.IGNORECASE,
)

_MIN_RE = re.compile(
    rf"(?:at\s+least|minimum(?:\s+of)?|min\.?|no\s+less\s+than|"
    rf"requires?|requir(?:ing|ed)|must\s+have)\s+"
    rf"(?P<n>\d{{1,2}}|{_WORD_NUM_RE})\+?\s*"
    rf"{_PAREN_GAP}(?:years?|yrs?\.?)"
    rf"(?!\s+(?:of\s+)?{_BAD_TAIL})"
    rf"\s*(?:of\s+)?{_DESCRIPTOR}{_EXP_WORD}",
    re.IGNORECASE,
)

_PLAIN_RE = re.compile(
    rf"(?<![\w/$])"
    rf"(?P<n>\d{{1,2}}|{_WORD_NUM_RE})\s*"
    rf"{_PAREN_GAP}(?:years?|yrs?\.?)"
    rf"(?!\s+(?:of\s+)?{_BAD_TAIL})"
    rf"\s*(?:of\s+)?{_DESCRIPTOR}{_EXP_WORD}",
    re.IGNORECASE,
)

_NEGATION_RES = [
    re.compile(r"\bno\s+(?:prior\s+|previous\s+)?experience\s+(?:is\s+)?"
               r"(?:necessary|needed|required|expected)\b", re.IGNORECASE),
    re.compile(r"\bno\s+experience\s+necessary\b", re.IGNORECASE),
    re.compile(r"\bentry[-\s]?level\s+(?:position|role|opportunity|job|"
               r"candidate|hire|associate|opening|opportunit(?:y|ies))\b",
               re.IGNORECASE),
    re.compile(r"\b(?:this\s+is\s+an?|ideal\s+for\s+an?|seeking\s+an?|"
               r"hiring\s+an?|looking\s+for\s+an?)\s+entry[-\s]?level\b",
               re.IGNORECASE),
    re.compile(r"\b(?:we['']?ll|we\s+will|will)\s+train\s+(?:you|the\s+right)\b",
               re.IGNORECASE),
    re.compile(r"\bon[-\s]the[-\s]job\s+training\s+(?:provided|available|"
               r"will\s+be|is\s+provided)\b", re.IGNORECASE),
    re.compile(r"\bno\s+(?:prior\s+|previous\s+)?(?:work\s+)?"
               r"experience\s+(?:in\s+\w+\s+)?(?:is\s+)?required\b",
               re.IGNORECASE),
]

_EDU_SUB_RE = re.compile(
    rf"(?:bachelor|master|phd|doctorate|ms|ma|bs|ba|mba|md|degree)"
    rf"[^.]{{0,80}}?\+\s*(?P<edu_yrs>\d{{1,2}}|{_WORD_NUM_RE})\s*"
    rf"(?:years?|yrs?\.?)"
    rf"[^.]{{0,80}}?\bor\b\s*"
    rf"(?P<alt_yrs>\d{{1,2}}|{_WORD_NUM_RE})\s*"
    rf"(?:years?|yrs?\.?)",
    re.IGNORECASE,
)

_MGMT_RE = re.compile(
    r"\b(?:manag(?:e|es|ed|ing|ement\s+of)|leads?\s+(?:a\s+)?team\s+of|"
    r"oversees?|supervis(?:e|es|ing))\s+(?:a\s+(?:team\s+of\s+)?)?"
    r"(?P<n>\d{1,3})\s+(?:direct\s+)?(?:reports?|people|employees|associates|engineers|staff)",
    re.IGNORECASE,
)

_DIRECT_REPORTS_RE = re.compile(
    r"\b(?P<n>\d{1,3})\s+direct\s+reports?\b",
    re.IGNORECASE,
)

_TITLE_PATTERNS = [
    # label 4
    (re.compile(r"\b(?:chief|c[teiof]o|cxo|cmo|ceo|cfo|coo|cto|cio|"
                r"president|svp|evp|vice\s*president|vp\b|"
                r"head\s+of|principal|distinguished|"
                r"director|managing\s+director|partner|fellow|"
                r"staff\s+(?:engineer|scientist|architect)|architect)\b",
                re.IGNORECASE), 4),
    # label 3
    (re.compile(r"\b(?:senior|sr\.?|lead|manager|mgr\.?|"
                r"iii\b|iv\b)\b", re.IGNORECASE), 3),
    # label 2
    (re.compile(r"\b(?:ii\b|mid|intermediate)\b", re.IGNORECASE), 2),
    # label 1
    (re.compile(r"\b(?:intern|internship|trainee|apprentice|"
                r"junior|jr\.?|entry|associate\s+i\b|"
                r"\bi\b)\b", re.IGNORECASE), 1),
]


_STRONG_SENIOR_RE = re.compile(
    r"\b(?:senior|sr\.?|lead(?:er)?|principal|staff|director|"
    r"chief|head\s+of|distinguished|managing\s+director|"
    r"vp\b|vice\s*president|svp|evp|partner|fellow)\b",
    re.IGNORECASE,
)


def _has_strong_senior(title: str) -> bool:
    return bool(_STRONG_SENIOR_RE.search(title or ""))



_EXECUTIVE_TITLE_RE = re.compile(
    r"\b(?:"
    r"chief\s+\w+\s+officer|ceo|cfo|coo|cto|cio|cmo|cxo|cpo|cro|"
    r"president\b|svp\b|evp\b|vice\s+president\b|"
    r"head\s+of\s+\w+|"
    r"managing\s+director|executive\s+director|"
    r"distinguished\s+(?:engineer|scientist|architect)|"
    r"principal\s+(?:engineer|scientist|architect|consultant|researcher)|"
    r"staff\s+(?:engineer|scientist|architect|researcher)"
    r")\b",
    re.IGNORECASE,
)


def _has_executive_signal(title: str) -> bool:
    return bool(_EXECUTIVE_TITLE_RE.search(title or ""))



_STRONG_TITLE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    # label 4 
    (re.compile(
        r"\b(?:"
        r"chief\s+\w+\s+officer|"
        r"ceo|cfo|coo|cto|cio|cmo|cxo|cpo|cro|cdo|cso|"
        r"president\b|svp\b|evp\b|vice\s+president\b|"
        r"head\s+of\s+\w+|"
        r"managing\s+director|executive\s+director|"
        r"distinguished\s+(?:engineer|scientist|architect|fellow)|"
        r"principal\s+(?:engineer|scientist|architect|consultant|researcher|"
        r"product\s+manager|designer)|"
        r"staff\s+(?:engineer|scientist|architect|researcher|product\s+manager|designer)"
        r")\b",
        re.IGNORECASE,
    ), 4),
    # label 1 
    (re.compile(
        r"\b(?:intern|internship|trainee|apprentice|"
        r"co[-\s]?op|summer\s+associate|graduate\s+rotation)\b",
        re.IGNORECASE,
    ), 1),
]


_TITLE_NUMERAL_RE = re.compile(r"\b(?P<num>II|III|IV)\b")
_NUMERAL_TO_LABEL = {"II": 2, "III": 3, "IV": 3}


def _scan_strong_title(title: str) -> Optional[tuple[int, str]]:
    if not title:
        return None
    for pat, label in _STRONG_TITLE_PATTERNS:
        m = pat.search(title)
        if m:
            return (label, m.group(0))
    return None


def _scan_title_numeral(title: str) -> Optional[tuple[int, str]]:
    if not title:
        return None
    matches = list(_TITLE_NUMERAL_RE.finditer(title))
    if not matches:
        return None
    last = matches[-1]
    num = last.group("num").upper()
    return (_NUMERAL_TO_LABEL[num], last.group(0))


# Sentence splitting / snippet capture


_SPLIT_RE = re.compile(r"(?:(?<=[.!?])\s+|\n+|(?:^|\n)\s*[+\-*\u2022])")


def _sentences(text: str) -> list[tuple[int, int, str]]:
    if not text:
        return []
    out: list[tuple[int, int, str]] = []
    pos = 0
    for m in _SPLIT_RE.finditer(text):
        if m.start() > pos:
            seg = text[pos:m.start()].strip()
            if seg:
                out.append((pos, m.start(), seg))
        pos = m.end()
    if pos < len(text):
        seg = text[pos:].strip()
        if seg:
            out.append((pos, len(text), seg))
    return out


def _snippet(text: str, span: tuple[int, int], max_len: int = 200) -> str:
    start, end = span
    for s_start, s_end, sent in _sentences(text):
        if s_start <= start and end <= s_end:
            sent = sent.lstrip("+-*\u2022 \t")
            if len(sent) > max_len:
                sent = sent[: max_len - 1].rstrip() + "\u2026"
            return sent
    sent = text[max(0, start - 80): start + 120].strip()
    sent = sent.lstrip("+-*\u2022 \t")
    if len(sent) > max_len:
        sent = sent[: max_len - 1].rstrip() + "\u2026"
    return sent


def _looks_like_requirements(snippet: str, full_text: str, span_start: int) -> bool:
    head = full_text[max(0, span_start - 1500): span_start].lower()
    keywords = ("requirement", "qualification", "qualifications",
                "what you'll need", "what you need", "must have",
                "minimum qualifications", "basic qualifications",
                "preferred qualifications", "you have", "you will have",
                "we are looking for", "skills required", "experience required")
    for kw in keywords:
        idx = head.rfind(kw)
        if idx == -1:
            continue
        # Only count if it's recent (within last ~1500 chars) 
        rest = head[idx:]
        if len(rest) < 1500 and (":" in rest[:60] or "\n" in rest[:60]):
            return True
    return False


# Extractors



def _is_company_tenure(text: str, span_start: int) -> bool:
    pre = text[max(0, span_start - 60): span_start]
    return bool(_COMPANY_PREFIX.search(pre))


def _scan_yoe(text: str) -> list[tuple[float, Optional[float], tuple[int, int], str]]:
    hits: list[tuple[float, Optional[float], tuple[int, int], str]] = []
    if not text:
        return hits

    seen_spans: list[tuple[int, int]] = []

    def _record(lo: float, hi: Optional[float], span: tuple[int, int], kind: str) -> None:
        if _is_company_tenure(text, span[0]):
            return
        for s, e in seen_spans:
            if not (span[1] <= s or span[0] >= e):
                return
        seen_spans.append(span)
        hits.append((lo, hi, span, kind))

    for m in _RANGE_RE.finditer(text):
        lo = _parse_num(m.group("lo"))
        hi = _parse_num(m.group("hi"))
        if lo is None or hi is None:
            continue
        if lo > hi:
            lo, hi = hi, lo
        if lo > 50 or hi > 50:
            continue
        _record(lo, hi, m.span(), "explicit_range")

    for m in _PLUS_RE.finditer(text):
        n = _parse_num(m.group("n"))
        if n is None or n > 50:
            continue
        _record(n, None, m.span(), "explicit_plus")

    for m in _MIN_RE.finditer(text):
        n = _parse_num(m.group("n"))
        if n is None or n > 50:
            continue
        _record(n, None, m.span(), "explicit_min")

    for m in _PLAIN_RE.finditer(text):
        n = _parse_num(m.group("n"))
        if n is None or n > 50:
            continue
        _record(n, None, m.span(), "explicit_yoe")

    return hits


def _scan_negation(text: str) -> Optional[tuple[int, int]]:
    if not text:
        return None
    for r in _NEGATION_RES:
        m = r.search(text)
        if m:
            return m.span()
    return None


def _scan_edu_sub(text: str) -> Optional[tuple[float, tuple[int, int]]]:
    if not text:
        return None
    for m in _EDU_SUB_RE.finditer(text):
        a = _parse_num(m.group("edu_yrs"))
        b = _parse_num(m.group("alt_yrs"))
        if a is None or b is None:
            continue
        return (min(a, b), m.span())
    return None


def _scan_mgmt(text: str) -> Optional[tuple[int, tuple[int, int]]]:
    if not text:
        return None
    for r in (_MGMT_RE, _DIRECT_REPORTS_RE):
        m = r.search(text)
        if m:
            try:
                n = int(m.group("n"))
            except (ValueError, IndexError):
                continue
            return (n, m.span())
    return None


def _title_seniority(title: str) -> Optional[tuple[int, str]]:
    if not title:
        return None
    numeral = _scan_title_numeral(title)
    if numeral is not None:
        return numeral
    for pat, label in _TITLE_PATTERNS:
        m = pat.search(title)
        if m:
            return (label, m.group(0))
    return None




def apply_rules(title: str, description: str) -> Optional[RuleHit]:
    title = title or ""
    description = description or ""

    # 1) Negation -> label 1
    neg_span = _scan_negation(description)
    neg_source_text = description
    if neg_span is None:
        neg_span = _scan_negation(title)
        neg_source_text = title
        neg_source = "title" if neg_span else None
    else:
        neg_source = "requirements" if _looks_like_requirements(
            description[max(0, neg_span[0] - 200):neg_span[1]],
            description, neg_span[0]
        ) else "description"

    # 2) Explicit numbers (description first, then title)
    desc_yoe = _scan_yoe(description)
    title_yoe = _scan_yoe(title)

    
    if desc_yoe:
    
        def _anchor(h: tuple[float, Optional[float], tuple[int, int], str]) -> float:
            lo, hi, _, _ = h
            return float(int((lo + hi) // 2)) if hi is not None else lo

        best = max(desc_yoe, key=lambda h: (_anchor(h), h[2][0]))
        lo, hi, span, kind = best
        anchor = _anchor(best)
        snippet = _snippet(description, span)
        source = "requirements" if _looks_like_requirements(
            snippet, description, span[0]
        ) else "description"
        decided_by = "explicit_range" if hi is not None else "explicit_yoe"
        label = yoe_to_label(anchor)
        # Senior-title bumps
        if _has_strong_senior(title):
            if label == 3 and anchor >= 10:
                label = 4
            elif label == 2 and anchor >= 4:
                label = 3
        return RuleHit(
            label=label,
            confidence=0.95,
            decided_by=decided_by,
            evidence_snippet=snippet,
            evidence_source=source,
            min_yoe=lo,
            max_yoe=hi,
        )

    # 3) Education substitute
    edu = _scan_edu_sub(description)
    if edu is not None:
        lo, span = edu
        snippet = _snippet(description, span)
        source = "requirements" if _looks_like_requirements(
            snippet, description, span[0]
        ) else "description"
        return RuleHit(
            label=yoe_to_label(lo),
            confidence=0.90,
            decided_by="education_substitute",
            evidence_snippet=snippet,
            evidence_source=source,
            min_yoe=lo,
        )

    # 4) Negation -> label 1 
    if neg_span is not None:
        snippet = _snippet(neg_source_text, neg_span)
        return RuleHit(
            label=1,
            confidence=0.95,
            decided_by="negation",
            evidence_snippet=snippet,
            evidence_source=neg_source,
            min_yoe=0.0,
        )

    # 5) Title YoE
    if title_yoe:
        def _anchor_t(h):
            lo, hi, _, _ = h
            return hi if hi is not None else lo
        best = max(title_yoe, key=lambda h: (_anchor_t(h), h[2][0]))
        lo, hi, span, kind = best
        snippet = _snippet(title, span)
        decided_by = "explicit_range" if hi is not None else "explicit_yoe"
        return RuleHit(
            label=yoe_to_label(_anchor_t(best)),
            confidence=0.95,
            decided_by=decided_by,
            evidence_snippet=snippet,
            evidence_source="title",
            min_yoe=lo,
            max_yoe=hi,
        )

    # 6) Strong title signal -- unambiguous C-suite 
    strong = _scan_strong_title(title)
    if strong is not None:
        label, snippet = strong
        return RuleHit(
            label=label,
            confidence=0.92,
            decided_by="strong_title",
            evidence_snippet=title,
            evidence_source="title",
        )

    # 7) Management scope
    mgmt = _scan_mgmt(description)
    if mgmt is not None:
        n, span = mgmt
        label = 4 if n >= 10 else 3
        snippet = _snippet(description, span)
        return RuleHit(
            label=label,
            confidence=0.65,
            decided_by="management_scope",
            evidence_snippet=snippet,
            evidence_source="description",
        )

    # 8) Title seniority, last resort
    ts = _title_seniority(title)
    if ts is not None:
        label, snippet = ts
        return RuleHit(
            label=label,
            confidence=0.55,
            decided_by="title_seniority",
            evidence_snippet=title,
            evidence_source="title",
        )

    return None


def extract_yoe_features(title: str, description: str) -> dict:
    title = title or ""
    description = description or ""

    desc_yoe = _scan_yoe(description)
    title_yoe = _scan_yoe(title)
    all_floors = [h[0] for h in desc_yoe] + [h[0] for h in title_yoe]
    all_ceils = [h[1] for h in desc_yoe if h[1] is not None] + \
                [h[1] for h in title_yoe if h[1] is not None]

    has_neg = _scan_negation(description) is not None or _scan_negation(title) is not None
    edu = _scan_edu_sub(description)
    mgmt = _scan_mgmt(description)
    ts = _title_seniority(title)

    return {
        "max_yoe_floor": max(all_floors) if all_floors else 0.0,
        "min_yoe_floor": min(all_floors) if all_floors else 0.0,
        "max_yoe_ceil": max(all_ceils) if all_ceils else 0.0,
        "n_yoe_matches": float(len(desc_yoe) + len(title_yoe)),
        "has_yoe_match": float(bool(all_floors)),
        "has_negation": float(has_neg),
        "has_edu_substitute": float(edu is not None),
        "edu_alt_yrs": float(edu[0]) if edu is not None else 0.0,
        "has_mgmt_scope": float(mgmt is not None),
        "mgmt_n": float(mgmt[0]) if mgmt is not None else 0.0,
        "title_seniority_label": float(ts[0]) if ts is not None else 0.0,
        "has_title_seniority": float(ts is not None),
        "jd_char_len": float(len(description)),
        "jd_word_len": float(len(description.split())),
        "title_word_len": float(len(title.split())),
        "bullet_count": float(description.count("\n+ ") + description.count("\n- ")
                              + description.count("\n* ") + description.count("\n\u2022")),
    }
