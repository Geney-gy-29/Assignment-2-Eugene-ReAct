from react_repro.metrics import em, fever_acc


def test_em_exact_match():
    assert em("Paris", "Paris") == 1


def test_em_case_and_article_insensitive():
    assert em("the Paris", "PARIS") == 1


def test_em_punctuation_insensitive():
    assert em("Paris.", "Paris") == 1


def test_em_mismatch():
    assert em("London", "Paris") == 0


def test_em_whitespace_normalized():
    assert em("New  York City", "new york city") == 1


def test_fever_acc_match():
    assert fever_acc("SUPPORTS", "supports") == 1


def test_fever_acc_mismatch():
    assert fever_acc("REFUTES", "SUPPORTS") == 0
