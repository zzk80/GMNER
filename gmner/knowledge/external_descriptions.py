"""Dataset-independent explanatory drafts for the FMNERG subtype schema.

The texts are intentionally generic: they define categories and decision
boundaries without copying entity mentions or contexts from any dataset split.
They are an offline assistant-authored draft and require human review before a
formal publication claim.
"""

from __future__ import annotations


EXPLANATION_KINDS = ("definition", "attributes", "boundary")


EXTERNAL_SUBTYPE_EXPLANATIONS: dict[
    tuple[str, str],
    dict[str, str],
] = {
    ("LOC", "building_other"): {
        "definition": "A building is a named constructed physical structure or complex that can be treated as a place.",
        "attributes": "Building references often involve an address, entrance, floor, hall, tower, construction, opening, renovation, or physical occupancy.",
        "boundary": "Use this subtype for the physical site, not for the organization operating there, and prefer cultural_place or sports_facility when that function defines the place.",
    },
    ("LOC", "city"): {
        "definition": "A city is a named urban settlement with a concentrated population and local administration.",
        "attributes": "City contexts commonly mention municipal government, residents, neighborhoods, downtown areas, metropolitan transport, or travel to an urban destination.",
        "boundary": "A city is below a state or country in administrative scale and is not merely a building, road, park, or unnamed geographic area.",
    },
    ("LOC", "continent"): {
        "definition": "A continent is a named very large geographic division that contains multiple countries and regions.",
        "attributes": "Continental references often describe cross-country geography, populations, climates, trade, travel, or events spanning a major world region.",
        "boundary": "A continent is broader than a country or state and should not be used for political alliances or organizations that share a continental name.",
    },
    ("LOC", "country"): {
        "definition": "A country is a named sovereign state or nationally governed territorial entity.",
        "attributes": "Country contexts often involve national government, borders, citizenship, foreign policy, currency, elections, or international representation.",
        "boundary": "Use country for a sovereign national territory; use state for a subnational unit and organization types for governments, agencies, or political groups.",
    },
    ("LOC", "cultural_place"): {
        "definition": "A cultural place is a named physical site whose identity is primarily historical, artistic, religious, memorial, or heritage-related.",
        "attributes": "Typical cues include exhibitions, collections, monuments, heritage status, worship, archives, visitors, preservation, or cultural performances at a site.",
        "boundary": "Classify the physical venue here, while the institution managing it may be an organization and an entertainment-focused venue may be entertainment_place.",
    },
    ("LOC", "entertainment_place"): {
        "definition": "An entertainment place is a named physical venue designed mainly for leisure, shows, nightlife, amusement, or audience experiences.",
        "attributes": "Contexts may mention tickets, audiences, screenings, stages, rides, performances, clubs, opening hours, or visiting a leisure venue.",
        "boundary": "Use sports_facility for athletic venues and cultural_place for heritage or cultural identity; the company operating a venue remains an organization.",
    },
    ("LOC", "location_other"): {
        "definition": "Location_other is a residual subtype for a named physical or geographic place that does not fit a more specific location category.",
        "attributes": "It may cover named districts, islands, natural areas, neighborhoods, sites, or geographic features when no dedicated subtype applies.",
        "boundary": "Use a specific location subtype whenever the text supports city, country, state, road, park, building, cultural place, entertainment place, or sports facility.",
    },
    ("LOC", "park"): {
        "definition": "A park is a named area of land designated for public recreation, landscape protection, conservation, or managed open space.",
        "attributes": "Park contexts often mention trails, gardens, wildlife, green space, visitors, recreation, conservation, national park status, or outdoor facilities.",
        "boundary": "A park is a place rather than its managing agency, and a stadium or arena inside it should be sports_facility when that venue is the entity.",
    },
    ("LOC", "road"): {
        "definition": "A road is a named transport route or thoroughfare, including streets, highways, avenues, lanes, and similar linear places.",
        "attributes": "Road references commonly involve traffic, intersections, exits, routes, addresses, closures, driving directions, or infrastructure work.",
        "boundary": "Classify the physical route here, not a transport authority, company, event, or neighborhood that happens to share its name.",
    },
    ("LOC", "sports_facility"): {
        "definition": "A sports facility is a named physical venue built or designated for athletic training, competition, or spectators.",
        "attributes": "Typical cues include stadium, arena, court, field, track, capacity, seating, home venue, match location, or training complex.",
        "boundary": "The venue is a location; the team is sports_team, the governing competition is sports_league, and a tournament or match is sports_event.",
    },
    ("LOC", "state"): {
        "definition": "A state is a named first-level or comparable subnational administrative territory within a country.",
        "attributes": "State contexts often mention governors, provincial or state law, regional administration, counties, capitals, or relations with the national government.",
        "boundary": "Use country for sovereign national entities and city for urban settlements; a government agency belonging to a state is an organization.",
    },
    ("ORG", "band"): {
        "definition": "A band is a named organized group of musicians who perform or record collectively.",
        "attributes": "Band contexts often mention members, albums, tours, concerts, formation, reunion, genre, recording, or collective performance.",
        "boundary": "The group is a band, an individual member is a musician, and a song or album is music rather than an organization.",
    },
    ("ORG", "company"): {
        "definition": "A company is a named commercial or legally constituted organization that produces goods, services, or business activity.",
        "attributes": "Company contexts frequently mention employees, executives, revenue, ownership, headquarters, products, customers, acquisition, or corporate operations.",
        "boundary": "Use company for the organization, brand_name_products or product_other for an offered item, and building_other for the physical premises.",
    },
    ("ORG", "educational_institution"): {
        "definition": "An educational institution is a named organization whose main purpose is teaching, training, study, or academic research.",
        "attributes": "Typical cues include students, faculty, courses, campus, admissions, degrees, departments, schools, colleges, universities, or academies.",
        "boundary": "The institution is an organization; its campus building is a location and an individual scholar or teacher is a person.",
    },
    ("ORG", "government_agency"): {
        "definition": "A government agency is a named public-sector body authorized to administer policy, regulation, services, enforcement, or official programs.",
        "attributes": "Agency contexts often mention departments, ministries, authorities, commissions, public officials, regulation, investigations, budgets, or government services.",
        "boundary": "Use political_party for an electoral organization, country or state for a territory, and ordinance for the rule or law issued by a government body.",
    },
    ("ORG", "news_agency"): {
        "definition": "A news agency is a named organization that gathers, produces, publishes, or distributes journalism and news reports.",
        "attributes": "Common cues include reporters, correspondents, newsroom, coverage, press reports, broadcasting, publishing, editorial operations, or news distribution.",
        "boundary": "The organization is a news agency, an individual reporter is a journalist, and a named periodical publication may be a magazine.",
    },
    ("ORG", "organization_other"): {
        "definition": "Organization_other is a residual subtype for a named collective, institution, association, or formal group without a more specific organization subtype.",
        "attributes": "Evidence of organization includes membership, leadership, coordinated activity, offices, governance, a shared mission, or acting as a collective agent.",
        "boundary": "Prefer company, band, school, agency, party, social organization, league, team, or news agency whenever those narrower functions are supported.",
    },
    ("ORG", "political_party"): {
        "definition": "A political party is a named organization that seeks political influence or public office through elections, representation, and a shared platform.",
        "attributes": "Party contexts often mention candidates, campaigns, voters, manifestos, ideology, party leaders, seats, coalitions, or electoral results.",
        "boundary": "A party is not the government agency it may control, the country where it operates, or the ordinance and policies it supports.",
    },
    ("ORG", "social_organization"): {
        "definition": "A social organization is a named nonprofit, civic, charitable, advocacy, community, religious, or membership-based group.",
        "attributes": "Typical cues include members, volunteers, donations, campaigns, chapters, community services, advocacy, charity, or a noncommercial mission.",
        "boundary": "Use company for primarily commercial entities, political_party for electoral groups, and organization_other when no social or civic purpose is evident.",
    },
    ("ORG", "sports_league"): {
        "definition": "A sports league is a named organization that structures, governs, or administers recurring competition among multiple teams or participants.",
        "attributes": "League contexts often mention standings, seasons, divisions, schedules, member clubs, rules, commissioners, promotion, playoffs, or championships.",
        "boundary": "A league organizes competition, a sports_team competes in it, a sports_event is an occurrence, and a sports_facility is a venue.",
    },
    ("ORG", "sports_team"): {
        "definition": "A sports team is a named organized group of athletes that competes as one unit or club.",
        "attributes": "Team contexts commonly mention players, coaches, roster, wins, losses, signing, club ownership, home games, supporters, or competition results.",
        "boundary": "Use athlete or coach for individuals, sports_league for the governing competition, sports_event for a match, and sports_facility for the venue.",
    },
    ("OTHER", "animal"): {
        "definition": "An animal entity is a named nonhuman animal, animal character, breed, or recognized animal subject treated as a proper entity.",
        "attributes": "Animal contexts may mention species, breed, habitat, owner, zoo, wildlife, veterinary care, behavior, or the name of an individual animal.",
        "boundary": "Use character for a primarily fictional person-like role and product or organization types when the animal name denotes a mascot, brand, or title instead.",
    },
    ("OTHER", "art_other"): {
        "definition": "Art_other covers a named artistic work or art entity not represented by music, film and television, or written work subtypes.",
        "attributes": "It can involve paintings, sculptures, installations, visual artworks, exhibitions, artistic movements, or other named creative artifacts.",
        "boundary": "Prefer music, film_and_television_works, or written_work for those media, and cultural_place when the entity is the physical gallery, monument, or heritage site.",
    },
    ("OTHER", "award"): {
        "definition": "An award is a named prize, honor, medal, title, distinction, or formal recognition granted for achievement.",
        "attributes": "Award contexts often mention winners, nominees, ceremonies, categories, recipients, judging, honors, prizes, or being awarded.",
        "boundary": "The award itself is OTHER; the presenting body is an organization, the recipient may be a person, and the ceremony may be an event.",
    },
    ("OTHER", "brand_name_products"): {
        "definition": "A brand-name product is a named commercial model, product line, or branded item identified by a proprietary market name.",
        "attributes": "Typical cues include model, version, launch, manufacturer, price, features, customers, purchase, device, vehicle, or consumer product family.",
        "boundary": "Use company for the producer or brand owner, software for a program, and product_other when the item is named but not clearly a branded product line.",
    },
    ("OTHER", "event_other"): {
        "definition": "Event_other is a named occurrence, campaign, conference, ceremony, incident, or organized occasion not covered by festival or sports_event.",
        "attributes": "Event contexts often include dates, venues, participants, attendance, schedules, opening, cancellation, commemoration, or something taking place.",
        "boundary": "Prefer festival for a celebratory recurring program and sports_event for athletic competition; an organizer is an organization, not the event itself.",
    },
    ("OTHER", "festival"): {
        "definition": "A festival is a named organized celebration or recurring program centered on culture, art, food, religion, community, or entertainment.",
        "attributes": "Festival contexts commonly mention annual editions, performances, screenings, celebrations, attendees, programs, venues, tickets, or opening dates.",
        "boundary": "Use sports_event for athletic tournaments, event_other for nonfestival occasions, and entertainment_place for the physical venue hosting the festival.",
    },
    ("OTHER", "film_and_television_works"): {
        "definition": "This subtype covers a named film, television series, episode, program, documentary, or other screen-based creative work.",
        "attributes": "Typical cues include cast, director, episode, season, premiere, cinema, broadcast, streaming, screenplay, ratings, or a screen release.",
        "boundary": "An actor or director is a person, the production company is an organization, and the cinema or studio site may be a location.",
    },
    ("OTHER", "game"): {
        "definition": "A game is a named interactive game title or codified play system, including video, board, card, and similar games.",
        "attributes": "Game contexts often mention players, gameplay, rules, levels, matches, release, platform, developer, score, expansion, or competition.",
        "boundary": "Use software when the named entity is primarily an application or platform, and sports_event when it denotes a particular real-world athletic match or tournament.",
    },
    ("OTHER", "magazine"): {
        "definition": "A magazine is a named periodical publication issued in recurring editions for a readership.",
        "attributes": "Magazine contexts commonly mention issues, editions, covers, articles, editors, subscribers, circulation, publication dates, or print and digital releases.",
        "boundary": "The publication is a magazine, its publisher may be a company or news agency, and an individual article may be a written_work.",
    },
    ("OTHER", "medical_thing"): {
        "definition": "Medical_thing covers a named disease, condition, treatment, drug, procedure, diagnostic concept, or other medically relevant entity.",
        "attributes": "Medical contexts may mention symptoms, diagnosis, patients, clinical studies, dosage, treatment, prevention, health effects, or medical professionals.",
        "boundary": "A hospital or health agency is an organization, a named medicine can also behave as a branded product, and a person remains PER.",
    },
    ("OTHER", "music"): {
        "definition": "Music covers a named song, album, composition, recording, soundtrack, or other musical work.",
        "attributes": "Music contexts often mention release, track, lyrics, album, chart, recording, performance, composer, producer, genre, or listening.",
        "boundary": "A musician is a person, a band is an organization, and a concert or festival is an event rather than the musical work.",
    },
    ("OTHER", "ordinance"): {
        "definition": "An ordinance is a named law, regulation, official order, policy instrument, treaty, or formally issued rule.",
        "attributes": "Typical cues include enactment, legislation, legal provisions, compliance, court review, regulation, amendment, repeal, or government approval.",
        "boundary": "The rule is an ordinance, while the government agency issuing or enforcing it is an organization and the jurisdiction is a location.",
    },
    ("OTHER", "product_other"): {
        "definition": "Product_other is a named manufactured item, service, device, vehicle, or commercial offering without a more specific product subtype.",
        "attributes": "Product contexts often mention design, manufacture, launch, price, features, customers, sales, use, replacement, or availability.",
        "boundary": "Use company for the producer, software for a program, and brand_name_products when a proprietary model or branded product family is clearly intended.",
    },
    ("OTHER", "software"): {
        "definition": "Software is a named computer program, application, operating system, digital platform, or software package.",
        "attributes": "Software contexts commonly mention installation, version, update, users, code, application, operating system, compatibility, download, or digital features.",
        "boundary": "The program is software, its developer may be a company, its online destination may be a website, and a video game may use the game subtype.",
    },
    ("OTHER", "sports_event"): {
        "definition": "A sports event is a named athletic match, race, tournament, championship, competition, or edition of a sporting contest.",
        "attributes": "Sports-event contexts often mention competitors, fixtures, rounds, scores, medals, qualification, dates, hosts, winners, or tournament stages.",
        "boundary": "A league is the governing organization, a team is a participant, a facility is the venue, and the event is the occurrence itself.",
    },
    ("OTHER", "website"): {
        "definition": "A website is a named online destination, domain-based publication, web portal, or web service presented as a site.",
        "attributes": "Website contexts often mention pages, domain, visitors, online content, posts, accounts, browsing, web traffic, links, or launching a site.",
        "boundary": "Use software for an application or operating platform, company for the organization behind the site, and magazine for a periodical publication identity.",
    },
    ("OTHER", "written_work"): {
        "definition": "A written work is a named book, novel, poem, play, article, essay, report, document, or other authored text.",
        "attributes": "Written-work contexts commonly mention author, title, publication, edition, chapter, translation, publisher, reading, manuscript, or literary genre.",
        "boundary": "The author is a person, a recurring periodical is a magazine, and an adaptation for screen belongs to film_and_television_works.",
    },
    ("PER", "actor"): {
        "definition": "An actor is a named person whose professional role is performing characters in film, television, theatre, or related productions.",
        "attributes": "Actor contexts often mention cast, role, performance, character, film, series, stage, audition, award nomination, or appearing in a production.",
        "boundary": "The actor is the performer, the fictional role is a character, and the person directing the production is a director.",
    },
    ("PER", "artist"): {
        "definition": "An artist is a named person known for creating or practicing visual, performing, or other creative art when no narrower role dominates.",
        "attributes": "Artist contexts may mention artwork, exhibitions, creative practice, style, studio, performance, design, commissions, or a body of work.",
        "boundary": "Prefer actor, author, musician, or director when the text clearly identifies that specific profession; use artist for broader or other creative roles.",
    },
    ("PER", "athlete"): {
        "definition": "An athlete is a named person who competes professionally or prominently in a sport or athletic discipline.",
        "attributes": "Athlete contexts often mention team membership, position, competition, training, records, medals, scores, transfer, injury, or athletic performance.",
        "boundary": "A coach trains participants, a sports_team is an organization, and a sports_event is the competition rather than the individual competitor.",
    },
    ("PER", "author"): {
        "definition": "An author is a named person recognized for writing books, articles, scripts, poems, or other textual works.",
        "attributes": "Author contexts commonly mention writing, publication, books, novels, articles, bibliography, literary awards, editors, or a new work.",
        "boundary": "The person is an author, the text is a written_work, and journalist should be used when reporting news is the central professional role.",
    },
    ("PER", "businessman"): {
        "definition": "A businessperson is a named person known for founding, owning, managing, investing in, or leading commercial enterprises.",
        "attributes": "Business contexts often mention founder, executive, chief officer, entrepreneur, investor, company ownership, wealth, deals, or corporate leadership.",
        "boundary": "The individual is a businessperson while the enterprise is a company; prefer politician, artist, or another profession when that role is central in context.",
    },
    ("PER", "character"): {
        "definition": "A character is a named fictional or dramatized person-like entity appearing in a story, game, film, television work, or other narrative.",
        "attributes": "Character contexts often mention plot, fictional role, portrayal, episode, novel, franchise, protagonist, antagonist, or relationships within a narrative.",
        "boundary": "The character is distinct from the actor portraying it, the author creating it, and the creative work in which it appears.",
    },
    ("PER", "coach"): {
        "definition": "A coach is a named person responsible for training, selecting, directing, or tactically leading athletes or a sports team.",
        "attributes": "Coach contexts commonly mention training, tactics, lineup, appointment, dismissal, season, players, staff, management, or leading a team.",
        "boundary": "A coach directs competitors, an athlete competes, and a sports_team is the organization employing or represented by them.",
    },
    ("PER", "director"): {
        "definition": "A director is a named person who provides primary creative direction for film, television, theatre, or another production.",
        "attributes": "Director contexts often mention directing, production, screenplay, cast, scenes, filmmaking, episodes, theatre, premiere, or creative control.",
        "boundary": "The director guides the production, an actor performs a role, and a company or studio is the organization producing the work.",
    },
    ("PER", "intellectual"): {
        "definition": "An intellectual is a named scholar, thinker, academic, scientist, or public figure known mainly for intellectual or research contributions.",
        "attributes": "Typical contexts include research, theory, scholarship, lectures, university work, publications, discovery, expertise, philosophy, or public ideas.",
        "boundary": "Prefer author, journalist, politician, or artist when the context centers on that specific occupation rather than scholarly or intellectual activity.",
    },
    ("PER", "journalist"): {
        "definition": "A journalist is a named person who reports, investigates, writes, edits, presents, or comments on news and public affairs.",
        "attributes": "Journalist contexts often mention reporting, interview, correspondent, anchor, editor, article, investigation, newsroom, broadcast, or press coverage.",
        "boundary": "The individual is a journalist, the employing outlet is a news agency, and author is better for primarily literary or non-news writing.",
    },
    ("PER", "musician"): {
        "definition": "A musician is a named person who sings, plays instruments, composes, produces, or performs music professionally or prominently.",
        "attributes": "Musician contexts commonly mention songs, albums, instruments, vocals, recording, concert, composition, producer, chart, or musical performance.",
        "boundary": "The individual is a musician, the collective may be a band, and a song or album belongs to the music subtype.",
    },
    ("PER", "person_other"): {
        "definition": "Person_other is a residual subtype for a named human individual whose role is unknown or not covered by another person subtype.",
        "attributes": "Person evidence includes personal actions, speech, relationships, biography, pronouns, titles, or being treated as an individual agent.",
        "boundary": "Use a specific occupation such as actor, athlete, author, coach, director, journalist, musician, politician, businessperson, artist, or intellectual when supported.",
    },
    ("PER", "politician"): {
        "definition": "A politician is a named person who holds, seeks, or exercises elected or senior political office and public governmental leadership.",
        "attributes": "Political contexts often mention elections, campaigns, office, parliament, legislation, government, voters, party leadership, policy, or official statements.",
        "boundary": "The person is a politician, their political_party is an organization, a government_agency is a public body, and a country or state is a location.",
    },
}


def validate_external_descriptions() -> None:
    """Fail fast when a subtype lacks one of the three independent views."""

    for key, explanations in EXTERNAL_SUBTYPE_EXPLANATIONS.items():
        missing = [kind for kind in EXPLANATION_KINDS if not explanations.get(kind)]
        if missing:
            raise ValueError(f"External description {key} is missing: {missing}")


validate_external_descriptions()
