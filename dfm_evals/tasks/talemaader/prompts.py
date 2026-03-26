from __future__ import annotations

JUDGE_PROMPT_TEMPLATE_DA = """Du er dommer for en eval af danske talemåder.

Vurder modelsvaret mod facit og giv en semantisk score fra {min_score} til {max_score}:
- {max_score}: Samme betydning som facit (kan være omformulering).
- mellemhøj score: Overvejende korrekt betydning, men mindre mangel/uklarhed.
- mellem score: Delvist korrekt, men væsentlige elementer mangler eller er forkerte.
- lav score: Mest forkert betydning.
- {min_score}: Forkert eller irrelevant betydning.

Talemåde: {talemaade_udtryk}
Facit: {reference}
Modelsvar: {prediction}

Svar KUN med gyldig JSON i dette format:
{{"score": <heltal fra {min_score} til {max_score}>, "reason": "kort begrundelse"}}
"""
