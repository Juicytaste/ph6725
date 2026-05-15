import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from science_explainer import (
    answer_science_question,
    build_ai_stack_cards,
    build_research_insight_cards,
    format_prediction_context,
)


class ScienceExplainerTests(unittest.TestCase):
    def test_ai_stack_cards_cover_all_requested_ai_layers(self):
        cards = build_ai_stack_cards()
        titles = {card["title"] for card in cards}

        self.assertIn("Machine Learning", titles)
        self.assertIn("Deep Learning", titles)
        self.assertIn("Reinforcement Learning", titles)
        self.assertIn("LLM Science Explainer", titles)
        self.assertTrue(all(card["maturity"] for card in cards))
        deep_card = next(card for card in cards if card["title"] == "Deep Learning")
        self.assertIn("MLPClassifier", deep_card["mvp"])
        self.assertIn("Implemented", deep_card["maturity"])

    def test_science_explainer_answers_transit_depth_with_grounded_context(self):
        answer = answer_science_question(
            "What is transit depth?",
            prediction_context=format_prediction_context(
                "CONFIRMED",
                0.82,
                {
                    "orbital_period": 17.0,
                    "transit_duration": 9.0,
                    "transit_depth": 1000.0,
                    "planet_radius": 7.0,
                },
            ),
        )

        self.assertIn("transit depth", answer.lower())
        self.assertIn("brightness", answer.lower())
        self.assertIn("1000", answer)
        self.assertIn("CONFIRMED", answer)

    def test_science_explainer_handles_false_positive_questions(self):
        answer = answer_science_question("Why could this be a false positive?")

        self.assertIn("false positive", answer.lower())
        self.assertIn("eclipsing binary", answer.lower())
        self.assertIn("model", answer.lower())

    def test_research_insight_cards_include_source_backed_presentation_points(self):
        cards = build_research_insight_cards()
        combined = "\n".join(card["body"] + " " + card["source_label"] for card in cards)

        self.assertIn("ExoMiner++", combined)
        self.assertIn("7,000", combined)
        self.assertIn("370", combined)
        self.assertIn("6,000", combined)
        self.assertIn("8,000", combined)
        self.assertIn("Reinforcement Learning", combined)
        self.assertTrue(all(card["source_url"].startswith("https://") for card in cards))

    def test_science_explainer_answers_exominer_and_open_data_questions(self):
        exominer = answer_science_question("What is ExoMiner++?")
        open_data = answer_science_question("Why is NASA open data important?")

        self.assertIn("ExoMiner++", exominer)
        self.assertIn("7,000", exominer)
        self.assertIn("370", exominer)
        self.assertIn("MLP neural network", exominer)
        self.assertIn("6,000", open_data)
        self.assertIn("educational prototype", open_data.lower())
