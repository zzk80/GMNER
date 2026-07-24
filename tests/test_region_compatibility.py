from gmner.knowledge.region_compatibility import compatibility_score


def test_region_compatibility_matches_type_terms():
    assert compatibility_score("PER", "person", "standing") == 1.0
    assert compatibility_score("LOC", "building", "tall") == 1.0
    assert compatibility_score("ORG", "logo", "red") == 1.0
    assert compatibility_score("OTHER", "trophy", "") == 1.0
    assert compatibility_score("ORG", "player", "running") == 0.5
    assert compatibility_score("OTHER", "painting", "") == 0.5
    assert compatibility_score("PER", "building", "") == 0.0
