"""Lightweight entity-type to VinVL object/attribute compatibility rules."""

from __future__ import annotations

from gmner.constants import ENTITY_TYPE2ID, ID2ENTITY_TYPE


TYPE_STRONG_COMPATIBILITY_TERMS = {
    "PER": {
        "person",
        "man",
        "woman",
        "boy",
        "girl",
        "people",
        "player",
        "athlete",
        "actor",
        "singer",
        "face",
        "head",
        "body",
        "human",
        "skier",
        "surfer",
    },
    "LOC": {
        "building",
        "street",
        "road",
        "city",
        "town",
        "bridge",
        "tower",
        "beach",
        "field",
        "court",
        "stadium",
        "park",
        "room",
        "house",
        "mountain",
        "sky",
        "water",
        "land",
        "sign",
    },
    "ORG": {
        "logo",
        "sign",
        "banner",
        "team",
        "uniform",
        "jersey",
        "shirt",
        "building",
        "company",
        "brand",
        "helmet",
        "cap",
        "hat",
        "bus",
        "car",
        "airplane",
    },
    "OTHER": {
        "product",
        "book",
        "movie",
        "poster",
        "award",
        "trophy",
        "phone",
        "computer",
        "camera",
        "food",
        "ball",
        "car",
        "event",
        "flag",
        "screen",
        "tv",
        "television",
        "stage",
        "microphone",
        "instrument",
        "guitar",
        "text",
        "letter",
    },
}

TYPE_WEAK_COMPATIBILITY_TERMS = {
    "PER": set(),
    "LOC": {"logo", "banner"},
    "ORG": {
        "person",
        "people",
        "player",
        "athlete",
        "ball",
        "court",
        "stadium",
        "field",
        "grass",
        "racket",
        "bat",
        "glove",
        "goal",
    },
    "OTHER": {
        "person",
        "people",
        "character",
        "sign",
        "banner",
        "image",
        "picture",
        "painting",
        "art",
        "music",
        "song",
        "game",
    },
}


def normalize_region_text(value: object) -> str:
    return str(value or "").strip().replace("_", " ").lower()


def compatibility_score(entity_type: str | int | None, label: object, attribute: object = "") -> float:
    if isinstance(entity_type, int):
        entity_type = ID2ENTITY_TYPE.get(entity_type, "O")
    entity_type = str(entity_type or "O").upper()
    if entity_type == "O" or entity_type not in TYPE_STRONG_COMPATIBILITY_TERMS:
        return 0.0

    text = f"{normalize_region_text(label)} {normalize_region_text(attribute)}"
    if not text.strip():
        return 0.0

    strong_terms = TYPE_STRONG_COMPATIBILITY_TERMS[entity_type]
    for term in strong_terms:
        if term in text:
            return 1.0
    weak_terms = TYPE_WEAK_COMPATIBILITY_TERMS.get(entity_type, set())
    for term in weak_terms:
        if term in text:
            return 0.5
    return 0.0


def type_id_to_name(type_id: int) -> str:
    return ID2ENTITY_TYPE.get(int(type_id), "O")


def type_name_to_id(type_name: str) -> int:
    return ENTITY_TYPE2ID.get(str(type_name).upper(), ENTITY_TYPE2ID["O"])
