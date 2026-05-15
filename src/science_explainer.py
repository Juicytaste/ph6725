from __future__ import annotations

from dataclasses import dataclass


FEATURE_LABELS = {
    "orbital_period": "orbital period",
    "transit_duration": "transit duration",
    "transit_depth": "transit depth",
    "planet_radius": "planet radius",
}


@dataclass(frozen=True)
class AIStackCard:
    title: str
    role: str
    mvp: str
    next_step: str
    maturity: str

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "role": self.role,
            "mvp": self.mvp,
            "next_step": self.next_step,
            "maturity": self.maturity,
        }


@dataclass(frozen=True)
class ResearchInsightCard:
    title: str
    subtitle: str
    body: str
    presentation_takeaway: str
    source_label: str
    source_url: str

    def as_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "subtitle": self.subtitle,
            "body": self.body,
            "presentation_takeaway": self.presentation_takeaway,
            "source_label": self.source_label,
            "source_url": self.source_url,
        }


def build_ai_stack_cards() -> list[dict[str, str]]:
    cards = [
        AIStackCard(
            title="Machine Learning",
            role="Classifies NASA Kepler/TESS candidates into confirmed planets, candidates, and false positives.",
            mvp="A Random Forest model trained on orbital period, transit duration, transit depth, and planet radius.",
            next_step="Add feature importance and model comparison for stronger scientific transparency.",
            maturity="Implemented in this MVP",
        ),
        AIStackCard(
            title="Deep Learning",
            role="Adds a neural-network classifier alongside the Random Forest baseline.",
            mvp="An MLPClassifier with two hidden layers, [64, 32], trained on the same NASA tabular features.",
            next_step="Upgrade from tabular MLP inputs to CNN or Transformer models over raw/folded light curves.",
            maturity="Implemented in this MVP",
        ),
        AIStackCard(
            title="Reinforcement Learning",
            role="Prioritizes which candidates deserve scarce follow-up telescope time.",
            mvp="Explained as a scheduling agent that balances confidence, scientific value, and observation cost.",
            next_step="Simulate follow-up decisions with rewards for confirmed discoveries and efficient scheduling.",
            maturity="Future work",
        ),
        AIStackCard(
            title="LLM Science Explainer",
            role="Turns model outputs and astronomy terms into student-friendly explanations.",
            mvp="A local, grounded explainer that answers common questions without external API keys.",
            next_step="Upgrade to retrieval-augmented generation over NASA docs, ADS papers, and Exoplanet Archive metadata.",
            maturity="Implemented as a local MVP",
        ),
    ]
    return [card.as_dict() for card in cards]


def build_research_insight_cards() -> list[dict[str, str]]:
    cards = [
        ResearchInsightCard(
            title="Deep Learning Research Inspiration",
            subtitle="NASA ExoMiner and ExoMiner++",
            body=(
                "NASA introduced ExoMiner++ in 2026 as an AI/deep learning tool trained for Kepler and TESS data. "
                "Its initial TESS run identified 7,000 targets as exoplanet candidates. NASA also reports that the "
                "earlier ExoMiner system validated 370 exoplanets from Kepler data, while a 2021 NASA article describes "
                "a deep neural network approach that added 301 planets to Kepler's validated count."
            ),
            presentation_takeaway=(
                "SWAI now includes a tabular MLP neural-network MVP; the deeper research path is to move from tabular "
                "features toward light-curve CNN/Transformer models inspired by ExoMiner-style workflows."
            ),
            source_label="NASA ExoMiner++ / NASA ExoMiner",
            source_url="https://science.nasa.gov/open-science/deep-learning-exoplanets-tess/",
        ),
        ResearchInsightCard(
            title="AI in Science and Open Data",
            subtitle="Educational prototype for public science",
            body=(
                "NASA announced that its official exoplanet tally reached 6,000 confirmed planets, with more than "
                "8,000 additional candidate planets still awaiting confirmation. That gap makes the science story "
                "clear: AI systems help researchers and students screen huge public archives before expensive follow-up."
            ),
            presentation_takeaway=(
                "Frame SWAI as an educational prototype that helps students and space enthusiasts understand how AI "
                "supports open-science discovery from NASA datasets."
            ),
            source_label="NASA 6,000 exoplanets milestone",
            source_url="https://www.nasa.gov/universe/exoplanets/nasas-tally-of-planets-outside-our-solar-system-reaches-6000/",
        ),
        ResearchInsightCard(
            title="Reinforcement Learning Future Work",
            subtitle="Follow-up observation scheduling",
            body=(
                "Do not claim SWAI already uses Reinforcement Learning. The accurate story is that RL fits a future "
                "follow-up layer: choosing which candidate to observe, when to observe it, and how to balance limited "
                "telescope time against scientific value. SkAI's Intelligent Scheduling project explicitly explores "
                "RL algorithms for dynamic, real-time astronomical observation scheduling."
            ),
            presentation_takeaway=(
                "SWAI classifies candidates today; an RL scheduler could later prioritize the most valuable candidates "
                "for telescope follow-up."
            ),
            source_label="SkAI Intelligent Scheduling for Astronomical Surveys",
            source_url="https://skai-institute.org/research/skai-funded-projects-year-1-2-2024-2026/project-9-intelligent-scheduling-for-astronomical-surveys/",
        ),
    ]
    return [card.as_dict() for card in cards]


def format_prediction_context(prediction: str | None, confidence: float | None, inputs: dict[str, float] | None) -> str:
    if not prediction or confidence is None or not inputs:
        return "No manual prediction is available yet. Ask about the model, NASA data, or exoplanet terms."

    parts = [
        f"Current prediction: {prediction}",
        f"trust/confidence: {confidence:.0%}",
    ]
    for key, value in inputs.items():
        label = FEATURE_LABELS.get(key, key.replace("_", " "))
        parts.append(f"{label}: {value:g}")
    return "; ".join(parts)


def answer_science_question(question: str, prediction_context: str = "") -> str:
    normalized = question.strip().lower()
    context = prediction_context.strip()
    context_note = f"\n\nCurrent SWAI context: {context}" if context else ""

    if not normalized:
        return "Ask me about the prediction, transit depth, false positives, NASA data, or how AI fits into SWAI."

    if any(term in normalized for term in ("transit depth", "depth", "brightness", "ppm")):
        return (
            "Transit depth is the tiny drop in a star's brightness when a planet crosses in front of it. "
            "A deeper transit usually means a larger object blocked more light, although star size and noise also matter. "
            "SWAI uses transit depth as one of four model features, so it helps the classifier separate planet-like signals "
            "from weaker or suspicious events."
            f"{context_note}"
        )

    if any(term in normalized for term in ("false positive", "imposter", "binary", "eclipsing")):
        return (
            "A false positive is a signal that looks planet-like but is probably caused by something else. "
            "Common causes include eclipsing binary stars, background stars contaminating the image, instrument artifacts, "
            "or noisy light curves. The model estimates this from the same transit features, but a final scientific claim "
            "still needs follow-up evidence."
            f"{context_note}"
        )

    if any(term in normalized for term in ("prediction", "confidence", "trust", "explain result", "current")):
        return (
            "The current SWAI prediction comes from a supervised machine learning classifier. "
            "It compares your input features with patterns learned from NASA Kepler and TESS candidate data, then returns "
            "the class with the highest probability. Treat the confidence as model trust, not as final astronomical proof."
            f"{context_note}"
        )

    if any(term in normalized for term in ("machine learning", "ml", "random forest", "classifier")):
        return (
            "Machine learning is the working core of this MVP. SWAI trains a Random Forest classifier on labeled NASA "
            "candidate records and predicts whether a new input resembles a confirmed planet, candidate, or false positive. "
            "This is supervised learning because the model learns from examples with known labels."
            f"{context_note}"
        )

    if any(term in normalized for term in ("exominer", "deep learning", "cnn", "transformer", "light curve")):
        return (
            "Deep learning is the next scientific upgrade path. NASA's ExoMiner++ is an AI/deep learning tool for "
            "Kepler and TESS data; NASA says its first TESS run identified 7,000 targets as exoplanet candidates, "
            "and the earlier ExoMiner validated 370 exoplanets from Kepler data. In this SWAI MVP, deep learning is "
            "implemented as an MLP neural network with hidden layers [64, 32] trained on the same four tabular inputs; "
            "the next research step is CNN or Transformer learning directly from light curves."
            f"{context_note}"
        )

    if any(term in normalized for term in ("reinforcement", "rl", "schedule", "telescope", "follow-up")):
        return (
            "Reinforcement learning fits the follow-up stage: an agent could choose which candidate to observe next under "
            "limited telescope time. A good reward could combine scientific value, predicted confidence, visibility windows, "
            "and the cost of observation."
            f"{context_note}"
        )

    if any(term in normalized for term in ("llm", "language model", "rag", "assistant", "copilot")):
        return (
            "In SWAI, the LLM layer is best used as a science explainer, not as the planet classifier. "
            "The classifier makes the prediction; the language assistant explains the terms, summarizes evidence, and can "
            "later use retrieval-augmented generation over NASA documentation and Exoplanet Archive metadata."
            f"{context_note}"
        )

    if any(term in normalized for term in ("nasa", "archive", "kepler", "tess", "data")):
        return (
            "SWAI uses open NASA exoplanet data from Kepler Objects of Interest and TESS Objects of Interest. "
            "NASA's official exoplanet count has reached 6,000 confirmed planets, while more than 8,000 candidates "
            "still await confirmation. That makes SWAI a strong educational prototype: it helps students and the public "
            "understand how AI can screen large open-science archives before follow-up observations."
            f"{context_note}"
        )

    return (
        "I can help explain SWAI's AI stack, the current prediction, and core exoplanet terms. "
        "Try asking about transit depth, false positives, confidence, deep learning, reinforcement learning, or LLMs."
        f"{context_note}"
    )
