"""Semantic type and subtype descriptions used to initialize prototypes."""

from __future__ import annotations

TYPE_DESCRIPTIONS = {
    "LOC": "A location entity refers to a country, city, region, landmark, venue, or geographical place.",
    "PER": "A person entity refers to a named human individual.",
    "ORG": "An organization entity refers to a company, team, institution, agency, group, or brand.",
    "OTHER": "An other named entity refers to an event, product, creative work, nationality, language, award, or named concept.",
}

SUBTYPE_DESCRIPTIONS = {
    "LOC": {
        "country": "A country, nation, or sovereign state.",
        "city": "A city, town, or urban settlement.",
        "region": "A geographical region, state, province, or district.",
        "landmark": "A named landmark or famous geographical place.",
        "venue": "A named venue, stadium, park, building, or event location.",
    },
    "PER": {
        "athlete": "A person who participates in sports or physical competitions.",
        "politician": "A politician, government leader, or public official.",
        "artist": "A singer, actor, musician, writer, or creative artist.",
        "celebrity": "A famous public figure or media celebrity.",
        "ordinary_person": "A named person without a more specific public role.",
    },
    "ORG": {
        "company": "A company or business organization that provides goods or services.",
        "sports_team": "A sports team, football club, or athletic organization.",
        "institution": "A university, school, hospital, charity, or social institution.",
        "government_agency": "A government agency, authority, or public administration body.",
        "brand": "A commercial brand, media organization, band, or named group.",
    },
    "OTHER": {
        "event": "A named event, festival, competition, ceremony, or public occasion.",
        "product": "A named product, device, service, vehicle, or consumer item.",
        "work": "A named movie, song, book, television show, or creative work.",
        "nationality_language": "A named nationality, ethnic group, language, or cultural identity.",
        "award": "A named prize, award, title, or honor.",
    },
}

