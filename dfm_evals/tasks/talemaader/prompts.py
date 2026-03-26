from __future__ import annotations

JUDGE_TEMPLATE_DA = """\
Du vurderer et indsendt svar mod et ekspertsvar om en dansk talemåde.

[BEGIN DATA]
************
[Talemåde]: {talemaade_udtryk}
************
[Facit]: {criterion}
************
[Modelsvar]: {answer}
************
[END DATA]

Sammenlign det semantiske indhold i modelsvaret med facit. \
Ignorer forskelle i stil, grammatik eller tegnsætning.

Indeholder modelsvaret den korrekte betydning af talemåden?

{instructions}
"""

JUDGE_INSTRUCTIONS_DA = (
    "Svar med 'GRADE: $LETTER' hvor LETTER er en af CPI. "
    'Vælg én mulighed: "C" for korrekte svar, '
    '"P" for delvist korrekte svar, '
    'eller "I" for forkerte svar.'
)
