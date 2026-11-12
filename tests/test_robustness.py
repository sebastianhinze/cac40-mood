"""Robustness checks: masking for the memory check, printed release times, Loughran-McDonald
scores, GARCH-X, and the column sets of the nonlinear-mood comparison."""

import numpy as np
import pandas as pd
import pytest
from arch import arch_model

from cac40_mood.scoring import loughran, masking, memory_report
from cac40_mood.vol import garchx, nonlinear_run

# ---------------------------------------------------------------- masking


def test_years_become_distances_to_the_publication_year() -> None:
    text = "En 2021 le chiffre d'affaires dépasse 2020 et 2019; objectif 2023. Réf 120210."
    masked = masking.mask_years(text, 2021)
    assert masked == (
        "En [year 0] le chiffre d'affaires dépasse [year -1] et [year -2]; "
        "objectif [year +2]. Réf 120210."
    )


def test_placeholders_and_replacement() -> None:
    names = [
        {"text": "LVMH", "kind": "issuer"},
        {"text": "Tiffany", "kind": "company"},
        {"text": "Tiffany & Co.", "kind": "company"},
        {"text": "tiffany", "kind": "company"},  # same string in another case: no new number
        {"text": "Bernard Arnault", "kind": "person"},
        {"text": "SA", "kind": "company"},  # too short: would damage ordinary words
    ]
    mapping = masking.placeholders_for(names, ["LVMH Moët Hennessy Louis Vuitton"])
    assert mapping["LVMH"] == masking.ISSUER_PLACEHOLDER
    assert mapping["Tiffany"] == "Company 1"
    assert mapping["Tiffany & Co."] == "Company 2"
    assert mapping["Bernard Arnault"] == "Person 1"
    assert "SA" not in mapping and "tiffany" not in mapping
    text = "LVMH SA rachète TIFFANY & CO. en 2021; Tiffany reste à New York. Bernard Arnault: +12%."
    masked = masking.apply_mask(text, mapping, 2021)
    # The longer name is replaced first, so "Tiffany & Co." does not become "Company 1 & Co.".
    assert masked == (
        "the Company SA rachète Company 2 en [year 0]; Company 1 reste à New York. Person 1: +12%."
    )
    # Only whole words: a name inside another word is left alone.
    assert masking.apply_mask("Capitalisation", {"Cap": "X"}, 2021) == "Capitalisation"


def test_guess_matching_ignores_accents_case_and_legal_suffixes() -> None:
    assert masking.guess_is_right("Société Générale S.A.", "Societe Generale")
    assert masking.guess_is_right("L'Oreal", "L'Oréal")
    assert not masking.guess_is_right("Bouygues", "Vinci")
    assert not masking.guess_is_right("", "Vinci")


def test_memory_sample_takes_its_quota_from_each_surprise_group() -> None:
    rng = np.random.default_rng(0)
    surprise = np.concatenate([np.zeros(300), rng.uniform(0.1, 1, 100), rng.uniform(-1, -0.1, 100)])
    documents = pd.DataFrame({"doc_id": [f"d{n}" for n in range(500)], "surprise": surprise})
    sample = masking.memory_sample(documents)
    counts = masking.surprise_group(sample["surprise"]).value_counts().to_dict()
    assert counts == masking.SAMPLE_QUOTAS
    assert sample["doc_id"].is_unique
    assert list(sample["doc_id"]) == list(masking.memory_sample(documents)["doc_id"])  # fixed draw


# ---------------------------------------------------------- printed times


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Vélizy-Villacoublay, le 27 février 2019 17:40 Chiffre d'affaires", "2019-02-27 17:40"),
        ("Paris, le 1er mars 2021 - 7h30 Résultats annuels", "2021-03-01 07:30"),
        ("Clichy, 6 February 2020 at 6:00 p.m.", "2020-02-06 06:00"),
        ("Résultats au 31 décembre 2022 COMMUNIQUÉ", None),  # a date without a time
        (
            "Comptes au 31 décembre 2022. Paris, le 9 février 2023 18h Communiqué",
            "2023-02-09 18:00",
        ),
    ],
)
def test_printed_release_time(text: str, expected: str | None) -> None:
    found = memory_report.printed_release_time(text)
    assert found == (pd.Timestamp(expected) if expected else None)


def test_agreement_rows_measure_how_far_masked_scores_moved() -> None:
    table = pd.DataFrame(
        {
            "tone": [0.5, -0.5, 0.0],
            "masked_tone": [0.6, -0.5, 0.0],
            "surprise": [0.3, -0.3, 0.0],
            "masked_surprise": [0.0, -0.3, 0.0],
            "uncertainty": [0.2, 0.4, 0.6],
            "masked_uncertainty": [0.2, 0.4, 0.6],
        }
    )
    rows = pd.DataFrame(memory_report.agreement_rows(table, "all")).set_index("field")
    assert rows.loc["tone", "mean_abs_difference"] == pytest.approx(0.033, abs=0.001)
    assert rows.loc["surprise", "same_direction"] == pytest.approx(0.667, abs=0.001)
    assert rows.loc["uncertainty", "correlation"] == 1.0


# ------------------------------------------------------- Loughran-McDonald

LISTS = {
    "positive": frozenset({"improved", "strong"}),
    "negative": frozenset({"loss", "impairment"}),
    "uncertainty": frozenset({"may", "uncertain"}),
}


def test_loughran_scores_count_words_and_skip_negated_positives() -> None:
    text = "Strong sales. Margins have not improved. A loss and an impairment may follow."
    scores = loughran.loughran_scores(text, LISTS)
    assert (scores["lmd_positive"], scores["lmd_negative"], scores["lmd_uncertain"]) == (1, 2, 1)
    assert scores["lmd_tone"] == pytest.approx(-1 / 3)
    assert scores["lmd_uncertainty"] == pytest.approx(100 / 13)
    assert loughran.loughran_scores("", LISTS)["lmd_tone"] == 0.0


def test_word_lists_are_read_from_the_dictionary_file(tmp_path) -> None:
    path = tmp_path / "dictionary.csv"
    path.write_text(
        "Word,Negative,Positive,Uncertainty\n"
        "LOSS,2009,0,0\nIMPROVED,0,2009,0\nMAY,0,0,2009\nNULL,2009,0,0\nTABLE,0,0,0\n"
    )
    lists = loughran.load_word_lists(path)
    assert lists["negative"] == {"loss", "null"}  # "NULL" is a word, not a missing value
    assert lists["positive"] == {"improved"}
    assert lists["uncertainty"] == {"may"}


def test_english_text_comes_from_the_original_or_from_the_official_twin() -> None:
    filings = pd.DataFrame(
        {"doc_id": ["en1", "fr1", "fr2", "fr3"], "lang": ["en", "fr", "fr", "fr"]}
    )
    texts = pd.DataFrame(
        {
            "doc_id": ["en1", "fr1", "fr2", "fr3"],
            "text": ["english", "f", "f", "f"],
            "text_quality": "ok",
        }
    )
    twins = pd.DataFrame(
        {"doc_id": ["fr1", "fr2"], "text": ["twin text", ""], "text_quality": ["ok", "empty"]}
    )
    english = loughran.english_texts(filings, texts, twins).set_index("doc_id")
    assert english.loc["en1", "english_source"] == "original"
    assert english.loc["fr1", "text"] == "twin text"
    assert "fr2" not in english.index  # the twin could not be read
    assert "fr3" not in english.index  # no English version exists


# ----------------------------------------------------------------- GARCH-X


def simulated_garchx(
    gamma: float, days: int = 3000, seed: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = (rng.random(days) < 0.1).astype(float)
    returns, variance = np.empty(days), 1.0
    for day in range(days):
        returns[day] = rng.normal(0, np.sqrt(variance))
        variance = 0.05 + 0.08 * returns[day] ** 2 + 0.85 * variance + gamma * x[day]
    return returns, x[:, None]


def test_without_x_columns_it_is_ordinary_garch() -> None:
    returns, _ = simulated_garchx(gamma=0.0)
    ours = garchx.fit_garchx(returns, np.empty((len(returns), 0)))
    reference = arch_model(returns, mean="Constant", vol="GARCH", p=1, q=1, rescale=False).fit(
        disp="off"
    )
    assert ours.alpha == pytest.approx(reference.params["alpha[1]"], abs=0.02)
    assert ours.beta == pytest.approx(reference.params["beta[1]"], abs=0.03)
    # The multi-day formula without x is the plain GARCH one: towards omega / (1 - alpha - beta).
    far = garchx.average_variance_ahead(np.array([5.0]), ours, np.zeros(0), 2000)
    assert far[0] == pytest.approx(ours.omega / (1 - ours.alpha - ours.beta), rel=0.05)


def test_it_finds_a_variance_jump_after_announcement_days() -> None:
    returns, x = simulated_garchx(gamma=1.0)
    fitted = garchx.fit_garchx(returns, x)
    assert fitted.gamma[0] == pytest.approx(1.0, abs=0.4)
    plain = garchx.fit_garchx(returns, np.empty((len(returns), 0)))
    with_x = garchx.negative_log_likelihood(returns, x, fitted)
    without_x = garchx.negative_log_likelihood(returns, np.empty((len(returns), 0)), plain)
    assert with_x < without_x - 10


def test_variance_never_falls_below_the_floor() -> None:
    returns = np.array([0.1, -0.2, 0.1, 0.0, 0.3])
    parameters = garchx.GarchXParameters(0.0, 0.001, 0.0, 0.0, np.array([-5.0]))
    variances = garchx.next_day_variances(returns, np.ones((5, 1)), parameters)
    floor = garchx.VARIANCE_FLOOR_SHARE * np.mean(returns**2)
    np.testing.assert_allclose(variances, floor)


# ------------------------------------------------- nonlinear-mood columns


def test_nonlinear_comparison_uses_the_forecast_of_its_own_horizon() -> None:
    models = nonlinear_run.models_for(5)
    assert models["M3"][-2:] == ["sv_log_forecast_5", "sv_mood"]
    assert models["M+SV0"][-1] == "sv0_log_forecast_5"
    assert set(models["M"]) < set(models["M3"])  # M3 is M plus the two new columns
    frame = pd.DataFrame(
        {"firm_id": ["A", "A"], "date": pd.to_datetime(["2023-01-02", "2023-01-03"])}
    )
    nonlinear = frame.assign(sv_mood=[0.1, np.nan], sv_log_forecast_5=-8.0, sv0_log_forecast_5=-8.1)
    assert len(nonlinear_run.add_nonlinear_columns(frame, nonlinear)) == 1
