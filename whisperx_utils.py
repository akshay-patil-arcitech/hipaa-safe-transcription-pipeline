import json
import os
import re
from collections import namedtuple
from pathlib import Path

SegmentLike = namedtuple("SegmentLike", ["start_sec", "end_sec", "speaker_tag"])

DEFAULT_MASKING_ENTITIES = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "DATE_TIME",
    "LOCATION",
    "IP_ADDRESS",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
    "MEDICAL_LICENSE",
]
PII_ENTITY_SET = [
    "PERSON",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "IP_ADDRESS",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
]
PHI_ENTITY_SET = [
    "PERSON",
    "DATE_TIME",
    "LOCATION",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IP_ADDRESS",
    "CREDIT_CARD",
    "IBAN_CODE",
    "BANK_ACCOUNT",
    "US_ITIN",
    "URL",
    "MEDICAL_LICENSE",
    "DATE_OF_BIRTH",
    "MRN",
    "US_NPI",
    "ICD10",
    "CPT",
    "US_SSN_COMPACT",
    "DATE_GENERIC",
    "AGE",
    "ADDRESS",
    "ZIP",
    "STATE_ID",
]

_presidio_cache: dict[str, tuple[object, object] | None] = {}
_regex_patterns = {
    "EMAIL_ADDRESS": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_NUMBER": re.compile(r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"),
    "IP_ADDRESS": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "US_SSN_COMPACT": re.compile(r"\b\d{9}\b"),
    "US_PASSPORT": re.compile(
        r"\b(?:passport|pp)\b[:\s#-]*[0-9A-Z]{6,9}\b",
        re.IGNORECASE,
    ),
    "US_DRIVER_LICENSE": re.compile(
        r"\b(?:dl|driver'?s license|driver license|lic(?:ense)?)\b[:\s#-]*[A-Z0-9]{5,15}\b",
        re.IGNORECASE,
    ),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "IBAN_CODE": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    "BANK_ACCOUNT": re.compile(r"\b\d{8,17}\b"),
    "US_ITIN": re.compile(r"\b9\d{2}-?\d{2}-?\d{4}\b"),
    "URL": re.compile(r"\bhttps?://[^\s]+\b"),
    "DATE_OF_BIRTH": re.compile(r"\b(?:dob|date of birth)\b[:\s-]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE),
    "MRN": re.compile(r"\b(?:mrn|medical record|record #|patient id)\b[:\s-]*[A-Z0-9-]{6,15}\b", re.IGNORECASE),
    "US_NPI": re.compile(r"\b\d{10}\b"),
    "ICD10": re.compile(r"\b[A-TV-Z][0-9][0-9AB]\.?\w{0,4}\b"),
    "CPT": re.compile(r"\b\d{5}\b"),
    "DATE_GENERIC": re.compile(
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{2,4})?)\b",
        re.IGNORECASE,
    ),
    "AGE": re.compile(r"\b(?:age|aged)\s*\d{1,3}\b", re.IGNORECASE),
    "ADDRESS": re.compile(
        r"\b\d{1,6}\s+[A-Za-z0-9\s.-]+(?:st|street|ave|avenue|rd|road|blvd|lane|ln|dr|drive|ct|court)\b",
        re.IGNORECASE,
    ),
    "ZIP": re.compile(r"\b\d{5}(?:-\d{4})?\b"),
    "STATE_ID": re.compile(r"\b[A-Z]{2}\s*\d{4,10}\b"),
}
_entity_tag_map = {
    "PERSON": "[PERSON_NAME]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS": "[EMAIL]",
    "US_SSN": "[SSN]",
    "US_SSN_COMPACT": "[SSN]",
    "US_PASSPORT": "[PASSPORT]",
    "US_DRIVER_LICENSE": "[DRIVER_LICENSE]",
    "US_ITIN": "[ITIN]",
    "IP_ADDRESS": "[IP]",
    "URL": "[URL]",
    "CREDIT_CARD": "[CREDIT_CARD]",
    "IBAN_CODE": "[IBAN]",
    "BANK_ACCOUNT": "[BANK_ACCOUNT]",
    "DATE_TIME": "[DATE]",
    "LOCATION": "[LOCATION]",
    "MEDICAL_LICENSE": "[MED_LICENSE]",
    "DATE_OF_BIRTH": "[DOB]",
    "MRN": "[MRN]",
    "US_NPI": "[NPI]",
    "ICD10": "[ICD10]",
    "CPT": "[CPT]",
    "DATE_GENERIC": "[DATE]",
    "AGE": "[AGE]",
    "ADDRESS": "[ADDRESS]",
    "ZIP": "[ZIP]",
    "STATE_ID": "[STATE_ID]",
}
_spacy_model_map = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "pt": "pt_core_news_sm",
    "it": "it_core_news_sm",
    "nl": "nl_core_news_sm",
    "ru": "ru_core_news_sm",
    "uk": "uk_core_news_sm",
    "pl": "pl_core_news_sm",
    "ro": "ro_core_news_sm",
    "ca": "ca_core_news_sm",
    "el": "el_core_news_sm",
    "lt": "lt_core_news_sm",
    "ar": "ar_core_news_sm",
    "vi": "vi_core_news_sm",
    "zh": "zh_core_web_sm",
    "ja": "ja_core_news_sm",
    "ko": "ko_core_news_sm",
}

_hipaa_context_keywords = [
    "gestational diabetes",
    "diabetes",
    "hypertension",
    "pregnancy",
    "ultrasound",
    "gestational age",
    # multilingual cues (aggressive)
    "diab\u00e8te",
    "diabetes",
    "hipertensi\u00f3n",
    "hypertension",
    "embarazo",
    "grossesse",
    "\u062d\u0645\u0644",
    "\u0633\u0643\u0631",
    "ultrasonido",
    "\u00e9chographie",
    "\u8d85\u58f0\u6ce2",
    "\u8d85\u97f3\u6ce2",
]
_gestational_age_pattern = re.compile(r"\b\d{1,2}\s*(weeks?|wks?)\b", re.IGNORECASE)


def _format_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h}:{m:02d}:{s:02d}.{ms:03d}"


def _speaker_label(tag: int) -> str:
    if 0 <= tag < 26:
        return chr(ord("A") + tag)
    return str(tag)


def _duration(segment: SegmentLike) -> float:
    return segment.end_sec - segment.start_sec


def _merge_segments(a: SegmentLike, b: SegmentLike) -> SegmentLike:
    return SegmentLike(
        start_sec=min(a.start_sec, b.start_sec),
        end_sec=max(a.end_sec, b.end_sec),
        speaker_tag=a.speaker_tag,
    )


def _merge_adjacent_same_speaker(segments: list[SegmentLike], merge_gap: float) -> list[SegmentLike]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    merged: list[SegmentLike] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_sec - prev.end_sec
        if seg.speaker_tag == prev.speaker_tag and gap <= merge_gap:
            merged[-1] = SegmentLike(prev.start_sec, max(prev.end_sec, seg.end_sec), prev.speaker_tag)
        else:
            merged.append(seg)
    return merged


def reassign_minor_speakers(
    segments: list[SegmentLike],
    max_speakers: int | None,
    min_total_sec: float | None,
    merge_gap: float,
) -> list[SegmentLike]:
    if not segments:
        return []
    if max_speakers is None and (min_total_sec is None or min_total_sec <= 0):
        return segments

    totals: dict[int, float] = {}
    for seg in segments:
        totals[seg.speaker_tag] = totals.get(seg.speaker_tag, 0.0) + _duration(seg)

    keep: set[int] = set()
    if max_speakers is not None and max_speakers > 0:
        top = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:max_speakers]
        keep.update(tag for tag, _ in top)
    if min_total_sec is not None and min_total_sec > 0:
        for tag, total in totals.items():
            if total >= min_total_sec:
                keep.add(tag)

    if not keep:
        return segments

    sorted_segments = sorted(segments, key=lambda s: (s.start_sec, s.end_sec))
    reassigned: list[SegmentLike] = []
    for i, seg in enumerate(sorted_segments):
        if seg.speaker_tag in keep:
            reassigned.append(seg)
            continue

        prev_keep = next((sorted_segments[j] for j in range(i - 1, -1, -1) if sorted_segments[j].speaker_tag in keep), None)
        next_keep = next((sorted_segments[j] for j in range(i + 1, len(sorted_segments)) if sorted_segments[j].speaker_tag in keep), None)
        if prev_keep and next_keep:
            gap_prev = abs(seg.start_sec - prev_keep.end_sec)
            gap_next = abs(next_keep.start_sec - seg.end_sec)
            target = prev_keep if gap_prev <= gap_next else next_keep
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, target.speaker_tag))
        elif prev_keep:
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, prev_keep.speaker_tag))
        elif next_keep:
            reassigned.append(SegmentLike(seg.start_sec, seg.end_sec, next_keep.speaker_tag))
        else:
            reassigned.append(seg)

    return _merge_adjacent_same_speaker(reassigned, merge_gap=merge_gap)


def apply_speaker_roles(segments: list[SegmentLike], roles: list[str]) -> list[SegmentLike]:
    if not segments or not roles:
        return segments
    totals: dict[int, float] = {}
    for seg in segments:
        totals[seg.speaker_tag] = totals.get(seg.speaker_tag, 0.0) + _duration(seg)
    ranked = [tag for tag, _ in sorted(totals.items(), key=lambda kv: kv[1], reverse=True)]
    role_map: dict[int, int] = {}
    for i, tag in enumerate(ranked):
        if i >= len(roles):
            break
        role_map[tag] = i
    remapped: list[SegmentLike] = []
    for seg in segments:
        if seg.speaker_tag in role_map:
            remapped.append(SegmentLike(seg.start_sec, seg.end_sec, role_map[seg.speaker_tag]))
        else:
            remapped.append(seg)
    return remapped


def _parse_masking_entities(value: str | None) -> list[str] | None:
    if value is None:
        return None
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        return None
    items: list[str] = []
    for item in raw:
        key = item.upper()
        if key == "PII":
            items.extend(PII_ENTITY_SET)
        elif key == "PHI":
            items.extend(PHI_ENTITY_SET)
        elif key == "HIPAA":
            items.extend(PHI_ENTITY_SET)
        else:
            items.append(item)
    seen = set()
    deduped: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped or None


def _parse_blocklist(value: str | None) -> list[str]:
    if value is None:
        return []
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items


def _apply_tagged_blocklist(text: str, blocklist: list[str]) -> str:
    if not blocklist:
        return text
    masked = text
    for item in blocklist:
        if not item:
            continue
        if ":" in item:
            tag, phrase = item.split(":", 1)
            tag = tag.strip().upper()
            phrase = phrase.strip()
            tag_value = f"[{tag}]" if tag else "[REDACTED]"
            if phrase:
                masked = re.sub(re.escape(phrase), tag_value, masked, flags=re.IGNORECASE)
        else:
            masked = re.sub(re.escape(item), "[AGENCY]", masked, flags=re.IGNORECASE)
    return masked


def _apply_context_rules(text: str) -> str:
    if not text:
        return text
    lowered = text.lower()
    has_keyword = any(keyword in lowered for keyword in _hipaa_context_keywords)
    if has_keyword and _gestational_age_pattern.search(text):
        text = _gestational_age_pattern.sub("[GESTATIONAL_AGE]", text)
        for keyword in _hipaa_context_keywords:
            if keyword in lowered:
                text = re.sub(re.escape(keyword), "[DIAGNOSIS]", text, flags=re.IGNORECASE)
    return text


# Aggressive multilingual PHI keyword masking
_phi_keyword_map = {
    # English
    "name": "[PERSON_NAME]",
    "patient": "[PATIENT]",
    "dob": "[DOB]",
    "date of birth": "[DOB]",
    "mrn": "[MRN]",
    "medical record": "[MRN]",
    "address": "[ADDRESS]",
    "phone": "[PHONE]",
    "email": "[EMAIL]",
    "ssn": "[SSN]",
    "passport": "[PASSPORT]",
    "insurance": "[INSURANCE]",
    # Spanish
    "nombre": "[PERSON_NAME]",
    "paciente": "[PATIENT]",
    "fecha de nacimiento": "[DOB]",
    "historia cl\u00ednica": "[MRN]",
    "direcci\u00f3n": "[ADDRESS]",
    "tel\u00e9fono": "[PHONE]",
    "correo": "[EMAIL]",
    "seguro": "[INSURANCE]",
    # French
    "nom": "[PERSON_NAME]",
    "patient": "[PATIENT]",
    "date de naissance": "[DOB]",
    "dossier m\u00e9dical": "[MRN]",
    "adresse": "[ADDRESS]",
    "t\u00e9l\u00e9phone": "[PHONE]",
    "courriel": "[EMAIL]",
    "assurance": "[INSURANCE]",
    # Arabic (common terms)
    "\u0627\u0644\u0627\u0633\u0645": "[PERSON_NAME]",
    "\u0627\u0644\u0645\u0631\u064a\u0636": "[PATIENT]",
    "\u062a\u0627\u0631\u064a\u062e \u0627\u0644\u0645\u064a\u0644\u0627\u062f": "[DOB]",
    "\u0627\u0644\u0639\u0646\u0648\u0627\u0646": "[ADDRESS]",
    "\u0627\u0644\u0647\u0627\u062a\u0641": "[PHONE]",
    "\u0627\u0644\u0628\u0631\u064a\u062f": "[EMAIL]",
    "\u0627\u0644\u062a\u0623\u0645\u064a\u0646": "[INSURANCE]",
    # Vietnamese
    "t\u00ean": "[PERSON_NAME]",
    "b\u1ec7nh nh\u00e2n": "[PATIENT]",
    "ng\u00e0y sinh": "[DOB]",
    "\u0111\u1ecba ch\u1ec9": "[ADDRESS]",
    "s\u1ed1 \u0111i\u1ec7n tho\u1ea1i": "[PHONE]",
    "email": "[EMAIL]",
    "b\u1ea3o hi\u1ec3m": "[INSURANCE]",
    # Chinese (Mandarin/Cantonese common)
    "\u59d3\u540d": "[PERSON_NAME]",
    "\u60a3\u8005": "[PATIENT]",
    "\u51fa\u751f\u65e5\u671f": "[DOB]",
    "\u5730\u5740": "[ADDRESS]",
    "\u7535\u8bdd": "[PHONE]",
    "\u7535\u5b50\u90ae\u4ef6": "[EMAIL]",
    "\u4fdd\u9669": "[INSURANCE]",
}


def _apply_phi_keywords(text: str) -> str:
    if not text:
        return text
    masked = text
    for key, tag in _phi_keyword_map.items():
        masked = re.sub(rf"\b{re.escape(key)}\b", tag, masked, flags=re.IGNORECASE)
    return masked


def _relabel_location_as_person(text: str, results):
    if not results:
        return results
    name_cues = (
        "my name is",
        "this is",
        "i am",
        "i'm",
        "dr.",
        "doctor",
        "soy",
        "me llamo",
        "mi nombre es",
        "nombre",
    )
    lowered = text.lower()
    for res in results:
        try:
            if res.entity_type != "LOCATION":
                continue
            start = max(0, res.start - 20)
            end = min(len(text), res.end + 20)
            window = lowered[start:end]
            if any(cue in window for cue in name_cues):
                res.entity_type = "PERSON"
        except Exception:
            continue
    return results


def _suppress_weak_locations(text: str, results):
    if not results:
        return results
    location_cues = (
        "address",
        "street",
        "st.",
        "avenue",
        "ave",
        "road",
        "rd",
        "blvd",
        "lane",
        "ln",
        "drive",
        "dr",
        "city",
        "state",
        "zip",
    )
    lowered = text.lower()
    for res in results:
        try:
            if res.entity_type != "LOCATION":
                continue
            start = max(0, res.start - 25)
            end = min(len(text), res.end + 25)
            window = lowered[start:end]
            if not any(cue in window for cue in location_cues):
                res.entity_type = "PERSON"
        except Exception:
            continue
    return results


def _get_presidio(language: str | None):
    lang = (language or "en").lower()
    cached = _presidio_cache.get(lang)
    if cached is not None:
        return cached
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine
    except ImportError as exc:
        raise RuntimeError(
            "Presidio is not installed. Install presidio-analyzer and presidio-anonymizer to enable masking."
        ) from exc
    model_name = _spacy_model_map.get(lang)
    if not model_name:
        _presidio_cache[lang] = None
        return None
    nlp_config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": lang, "model_name": model_name}],
        "ner_model_configuration": {
            "labels_to_ignore": ["PERSON_NAME", "ID_N", "FAC", "LANGUAGE", "MISC", "ORDINAL", "CARDINAL"],
        },
    }
    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=[lang])
        anonymizer = AnonymizerEngine()
        _presidio_cache[lang] = (analyzer, anonymizer)
        return analyzer, anonymizer
    except Exception:
        _presidio_cache[lang] = None
        return None


def _regex_mask(text: str, entities: list[str]) -> str:
    masked = text
    for entity in entities:
        pattern = _regex_patterns.get(entity)
        if pattern:
            tag = _entity_tag_map.get(entity, "[REDACTED]")
            masked = pattern.sub(tag, masked)
    return masked


def mask_text(
    text: str,
    entities: list[str],
    language: str | None,
    blocklist: list[str] | None = None,
    context_rules: bool = True,
) -> str:
    if not text:
        return text
    if blocklist:
        text = _apply_tagged_blocklist(text, blocklist)
    if context_rules:
        text = _apply_context_rules(text)
    text = _apply_phi_keywords(text)
    engine = _get_presidio(language)
    if engine is None:
        return _regex_mask(text, entities)
    analyzer, anonymizer = engine
    lang = (language or "en").lower()
    try:
        supported = set(analyzer.get_supported_entities(language=lang))
        entities = [e for e in entities if e in supported] or entities
    except Exception:
        pass
    try:
        results = analyzer.analyze(text=text, entities=entities, language=lang)
    except ValueError:
        return _regex_mask(text, entities)
    results = _relabel_location_as_person(text, results)
    results = _suppress_weak_locations(text, results)
    if not results:
        return _regex_mask(text, entities)
    from presidio_anonymizer.entities import OperatorConfig
    operators = {
        entity: OperatorConfig("replace", {"new_value": _entity_tag_map.get(entity, "[REDACTED]")})
        for entity in entities
    }
    operators["DEFAULT"] = OperatorConfig("replace", {"new_value": "[REDACTED]"})
    return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text


def write_json(path: Path, data):
    try:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
    except OSError:
        pass
