"""Project-wide constants."""

IGNORE_INDEX = -100
DEFAULT_IMAGE_SIZE = 224
DEFAULT_LABEL2ID = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-LOC": 3,
    "I-LOC": 4,
    "B-ORG": 5,
    "I-ORG": 6,
    "B-OTHER": 7,
    "I-OTHER": 8,
}
COARSE_ENTITY_TYPE_ALIASES = {
    "LOC": "LOC",
    "LOCATION": "LOC",
    "BUILDING": "LOC",
    "PER": "PER",
    "PERSON": "PER",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",
    "OTHER": "OTHER",
    "ART": "OTHER",
    "EVENT": "OTHER",
    "PRODUCT": "OTHER",
}
ENTITY_TYPE2ID = {
    "LOC": 0,
    "PER": 1,
    "ORG": 2,
    "OTHER": 3,
    "O": 4,
}
ID2ENTITY_TYPE = {value: key for key, value in ENTITY_TYPE2ID.items()}


def normalize_entity_type(raw_type: str) -> str:
    """Map dataset-specific coarse/fine super types onto GMNER's four coarse types."""

    normalized = str(raw_type or "").strip().replace("-", "_").upper()
    return COARSE_ENTITY_TYPE_ALIASES.get(normalized, normalized if normalized in ENTITY_TYPE2ID else "OTHER")


def normalize_bio_label(label: str) -> str:
    """Normalize BIO tags such as B-person/B-building to the 4-way GMNER schema."""

    text = str(label or "O").strip()
    if text == "O" or "-" not in text:
        return "O"
    prefix, entity_type = text.split("-", 1)
    prefix = prefix.upper()
    if prefix not in {"B", "I"}:
        return "O"
    return f"{prefix}-{normalize_entity_type(entity_type)}"


def strip_bio_prefix(label: str) -> str:
    text = str(label or "O").strip()
    if text == "O":
        return "O"
    if "-" in text:
        return text.split("-", 1)[1]
    return text
