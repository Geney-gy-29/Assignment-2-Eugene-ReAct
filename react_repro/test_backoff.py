"""[A3-IMPROVEMENT] Unit tests for the generalized back-off signals.

Fixtures are drawn from the real Assignment-2 n=10 trajectories so the tests
pin the exact failure modes the improvement targets, not synthetic ideals.
"""

import pytest

from react_repro.backoff import (
    TAU_ENTROPY,
    TRIGGERS,
    is_informative,
    n_informative,
    s1_exhausted,
    s2_thin_evidence,
    sc_lowconf_cga,
    sc_lowconf_paper,
    vote_entropy,
)

# Real FEVER trajectory: one search, immediate Finish[REFUTES] on a claim whose
# gold label is NOT ENOUGH INFO. The paper's trigger cannot see this.
FEVER_PREMATURE = {
    "answer": "REFUTES",
    "n_steps": 2,
    "exhausted": False,
    "actions": ["Search[New York City Landmarks Preservation Commission]", "Finish[REFUTES]"],
    "observations": [
        "The New York City Landmarks Preservation Commission (LPC) is the "
        "New York City agency charged with administering the city's Landmarks Preservation Law."
    ],
}

# Real HotpotQA trajectory: burned all 7 steps, reissued Search[Hardley Flood]
# verbatim. The paper's trigger DOES fire here.
HOTPOT_EXHAUSTED = {
    "answer": "",
    "n_steps": 7,
    "exhausted": True,
    "actions": [
        "Search[Hardley Flood]", "Lookup[waterfowl]", "Search[Hardley Flood waterfowl]",
        "Search[Hardley Flood birds]", "Search[Hardley Flood]", "Lookup[lagoons]",
        "Search[Hardley Flood goose]",
    ],
    "observations": [
        "Hardley Flood is a nature reserve in Norfolk, England.",
        "No more results.",
        "Could not find Hardley Flood waterfowl. Similar: ['Hardley', 'Hardley Flood'].",
        "Could not find Hardley Flood birds. Similar: ['Hardley Flood'].",
        "Hardley Flood is a nature reserve in Norfolk, England.",
        "No more results.",
        "Could not find Hardley Flood goose. Similar: ['Hardley Flood'].",
    ],
}

# Real HotpotQA trajectory: well-grounded, 2 informative searches then Finish.
HOTPOT_GROUNDED = {
    "answer": "James Worthy",
    "n_steps": 4,
    "exhausted": False,
    "actions": [
        "Search[Danny Green]", "Search[Danny Green (basketball)]",
        "Search[James Worthy]", "Finish[James Worthy]",
    ],
    "observations": [
        "Could not find Danny Green. Similar: ['Danny Green (basketball)'].",
        "Daniel Richard Green Jr. is an American professional basketball player.",
        "James Ager Worthy is an American former professional basketball player.",
    ],
}


class TestIsInformative:
    def test_failed_search_is_not_evidence(self):
        assert not is_informative("Could not find X. Similar: ['Y'].")

    def test_exhausted_lookup_is_not_evidence(self):
        assert not is_informative("No more results.")

    def test_invalid_action_is_not_evidence(self):
        assert not is_informative("Invalid action: Search Hardley")

    def test_empty_is_not_evidence(self):
        assert not is_informative("")
        assert not is_informative("   ")

    def test_page_content_is_evidence(self):
        assert is_informative("Hardley Flood is a nature reserve in Norfolk, England.")

    def test_case_insensitive(self):
        assert not is_informative("could not find x. Similar: [].")


class TestS1Exhausted:
    def test_fires_on_step_burn(self):
        assert s1_exhausted(HOTPOT_EXHAUSTED)

    def test_silent_on_premature_finish(self):
        """The core limitation: the paper's trigger is blind to this case."""
        assert not s1_exhausted(FEVER_PREMATURE)

    def test_silent_on_grounded_answer(self):
        assert not s1_exhausted(HOTPOT_GROUNDED)

    def test_falls_back_to_empty_answer(self):
        """Callers predating the `exhausted` flag still work."""
        assert s1_exhausted({"answer": ""})
        assert not s1_exhausted({"answer": "Paris"})


class TestS2ThinEvidence:
    def test_fires_on_premature_finish(self):
        """The case S1 misses: 1 informative retrieval, then Finish."""
        assert n_informative(FEVER_PREMATURE) == 1
        assert s2_thin_evidence(FEVER_PREMATURE)

    def test_silent_on_grounded_answer(self):
        assert n_informative(HOTPOT_GROUNDED) == 2
        assert not s2_thin_evidence(HOTPOT_GROUNDED)

    def test_disjoint_from_s1(self):
        """Exhaustion is S1's responsibility; S2 must not double-count it."""
        assert not s2_thin_evidence(HOTPOT_EXHAUSTED)

    def test_failed_searches_do_not_count_as_evidence(self):
        """3 searches but only 1 returned a page -> still thin."""
        thin = {
            "answer": "Yes", "exhausted": False,
            "observations": [
                "Could not find A. Similar: [].",
                "Could not find B. Similar: [].",
                "Real page content about the subject.",
            ],
        }
        assert n_informative(thin) == 1
        assert s2_thin_evidence(thin)

    def test_tau_is_configurable(self):
        assert not s2_thin_evidence(FEVER_PREMATURE, tau=1)
        assert s2_thin_evidence(HOTPOT_GROUNDED, tau=3)


class TestTriggerRegistry:
    def test_paper_arm_matches_s1(self):
        for traj in (FEVER_PREMATURE, HOTPOT_EXHAUSTED, HOTPOT_GROUNDED):
            assert TRIGGERS["paper"](traj) == s1_exhausted(traj)

    def test_s1s2_catches_both_failure_modes(self):
        assert TRIGGERS["s1s2"](HOTPOT_EXHAUSTED)
        assert TRIGGERS["s1s2"](FEVER_PREMATURE)

    def test_s1s2_is_selective(self):
        """A trigger that fires on everything degenerates to plain CoT-SC."""
        assert not TRIGGERS["s1s2"](HOTPOT_GROUNDED)

    def test_all_arms_registered(self):
        assert set(TRIGGERS) == {"paper", "s1s2", "s1s3", "cga", "s3only", "cga_tau1"}

    def test_tau1_arm_is_stricter_than_cga(self):
        """At tau=1 only zero-evidence Finishes count as thin, so the FEVER
        1-search case stops firing on S2 (S3 may still catch it).

        Tests the S2 component directly: the composite arms containing S3
        would issue a live API call.
        """
        assert s2_thin_evidence(FEVER_PREMATURE, tau=2)
        assert not s2_thin_evidence(FEVER_PREMATURE, tau=1)

    def test_tau1_still_catches_zero_evidence(self):
        zero = {
            "answer": "REFUTES", "exhausted": False,
            "observations": ["Could not find X. Similar: []."],
        }
        assert s2_thin_evidence(zero, tau=1)


class TestVoteEntropy:
    def test_unanimous_is_zero(self):
        assert vote_entropy(["a"] * 21) == 0.0

    def test_uniform_scatter_is_one(self):
        """All-distinct answers = maximum uncertainty."""
        assert vote_entropy(["a", "b", "c", "d"]) == pytest.approx(1.0)

    def test_all_empty_is_max_uncertainty(self):
        assert vote_entropy(["", "", ""]) == 1.0

    def test_empty_votes_are_excluded(self):
        assert vote_entropy(["a", "a", "", ""]) == 0.0

    def test_clean_split_below_scatter(self):
        """Two confident hypotheses should score lower than broad scatter,
        yet both fall below the paper's n/2 majority threshold."""
        split = vote_entropy(["a"] * 5 + ["b"] * 5)
        scatter = vote_entropy(["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
        assert split < scatter


class TestScTriggers:
    def test_paper_fires_below_majority(self):
        assert sc_lowconf_paper({"majority_count": 9}, 21)
        assert not sc_lowconf_paper({"majority_count": 11}, 21)

    def test_cga_agrees_when_below_majority(self):
        assert sc_lowconf_cga({"majority_count": 9, "all_answers": ["a"] * 9}, 21)

    def test_cga_catches_scattered_vote_above_majority(self):
        """Majority passes n/2, but the remaining votes are fully scattered --
        invisible to the paper's top-bin-only threshold."""
        answers = ["a"] * 11 + list("bcdefghij")
        sc = {"majority_count": 11, "all_answers": answers}
        assert not sc_lowconf_paper(sc, 21)
        assert vote_entropy(answers) > TAU_ENTROPY
        assert sc_lowconf_cga(sc, 21)

    def test_cga_silent_on_confident_vote(self):
        sc = {"majority_count": 19, "all_answers": ["a"] * 19 + ["b", "c"]}
        assert not sc_lowconf_cga(sc, 21)
