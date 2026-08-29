"""
prep_data.py
============

Writes ``data/c2_seed_data.jsonl`` — the seed corpus for the QLoRA fine-tune.

Run:

    python prep_data.py

Each line is a JSON object of the form::

    {"messages": [{"role": "system", ...},
                  {"role": "user", ...},
                  {"role": "assistant", ...}]}

which is exactly what ``tokenizer.apply_chat_template`` expects for
Llama-3-Instruct, and what ``train.py`` converts into a flat ``text`` column
for ``SFTTrainer``.

The examples deliberately teach *distinctions* rather than one-to-one
substitution: several entries end with a "Note:" paragraph spelling out where
the alternatives are not interchangeable.
"""

from __future__ import annotations

import json
import os

from c2_engine import (
    DATA_PATH,
    build_enhance_messages,
    build_explanation_messages,
    build_synonym_messages,
)

# --------------------------------------------------------------------------- #
# MODE 1 — Enhance Phrase
# --------------------------------------------------------------------------- #

ENHANCE_EXAMPLES = [
    (
        "This is an ideal opportunity.",
        None,
        "This constitutes an exceptionally auspicious opportunity.",
    ),
    (
        "I want to understand this problem better.",
        None,
        "I seek to develop a more nuanced understanding of this predicament.",
    ),
    (
        "The mountain top was unreachable because the weather was really bad.",
        None,
        "The summit remained inaccessible, the weather having proved insurmountable.",
    ),
    (
        "It is important that we change our approach, because the old one is "
        "difficult to maintain.",
        None,
        "A recalibration of our approach is paramount, given how arduous the "
        "existing one has become to sustain.",
    ),
    (
        "The view was very beautiful and it made me want to stay longer.",
        None,
        "The view was so exquisite that I found myself reluctant to leave.",
    ),
    (
        "Thanks so much for your help with this, I really appreciate it.",
        None,
        "I am most grateful for your assistance with this; it has been "
        "genuinely appreciated.",
    ),
    (
        "We had a problem with the supplier, so the launch will be late.",
        None,
        "An impediment on the supplier's side has obliged us to defer the launch.",
    ),
    (
        "This is an ideal opportunity.",
        "warmer and more personal",
        "What a wonderfully opportune moment this is for us.",
    ),
]

# --------------------------------------------------------------------------- #
# MODE 2 — C2 Synonyms
# --------------------------------------------------------------------------- #

SYNONYM_EXAMPLES = [
    (
        "ideal",
        """OPTIMAL
The most favourable or effective option under a given set of constraints; analytical, faintly technical in register.
Example: Dawn is the optimal time to photograph the valley.

EXEMPLARY
Serving as an outstanding model that others ought to emulate; carries evaluative, often moral praise.
Example: Her conduct throughout the crisis was exemplary.

QUINTESSENTIAL
Representing the purest and most characteristic instance of a category, rather than the best one.
Example: He is the quintessential English gentleman.

CONSUMMATE
Displaying the highest attainable degree of skill or polish; applied to people and to performances.
Example: A consummate negotiator, she never conceded a point cheaply.

AUSPICIOUS
Promising a favourable outcome; concerns omens, beginnings and prospects rather than intrinsic quality.
Example: The talks opened on an auspicious note.

Note: these five diverge sharply. 'Optimal' judges efficiency, 'exemplary' judges standard, 'quintessential' judges typicality, 'consummate' judges mastery, and 'auspicious' judges prospects. An ideal candidate is an exemplary candidate, never an auspicious one; an ideal moment to begin is an auspicious one, never a consummate one.""",
    ),
    (
        "unreachable",
        """UNATTAINABLE
Beyond achievement; takes goals, standards and ambitions as its objects rather than places.
Example: Perfection is an unattainable standard, though a useful direction of travel.

INACCESSIBLE
Impossible or very difficult to reach or make use of, whether physically or intellectually.
Example: The summit is inaccessible until the snow clears.

ELUSIVE
Persistently escaping capture, definition or recollection; implies repeated near-misses rather than outright impossibility.
Example: A precise definition of the term remains elusive.

INSURMOUNTABLE
Said of obstacles rather than destinations: an impediment that cannot be overcome.
Example: The technical hurdles proved insurmountable within the budget.

BEYOND ONE'S KEN
Outside the limits of one's knowledge or comprehension; idiomatic and literary, with a faintly old-fashioned colour.
Example: The mathematics of the proof lies well beyond my ken.

Note: the object determines the word. A peak is inaccessible; the record it represents is unattainable; the weather that stops you is insurmountable; a subject you cannot follow is beyond your ken. 'Elusive' alone implies that the thing keeps almost being reached.""",
    ),
    (
        "important",
        """PARAMOUNT
Of the highest priority, overriding every competing consideration; usually predicative.
Example: In this laboratory, safety is paramount.

PIVOTAL
Important because subsequent events turn on it; a claim about causation, not about size.
Example: The 1989 election proved pivotal for the region.

SALIENT
Standing out as the most noticeable or relevant point within an argument or a body of evidence.
Example: Let me restate the salient objections before we vote.

MOMENTOUS
Carrying grave historical weight; reserved for occasions and decisions, not for details.
Example: It was a momentous day in the country's constitutional history.

INTEGRAL
Indispensable as a constituent part of a whole; structural rather than evaluative.
Example: Redundancy is integral to the design of the system.

Note: these are not interchangeable. 'Paramount' ranks a priority, 'pivotal' explains a consequence, 'salient' concerns prominence in discussion, 'momentous' concerns historical gravity, and 'integral' concerns structure. An integral component may be entirely unmomentous.""",
    ),
    (
        "difficult",
        """ARDUOUS
Demanding sustained physical or mental effort over time; the difficulty is one of endurance.
Example: The ascent was arduous but never technically dangerous.

INTRICATE
Difficult because elaborately detailed and interconnected, not because strenuous.
Example: The judgment rests on an intricate chain of statutory reasoning.

FORMIDABLE
Inspiring respect or apprehension by sheer scale, strength or accomplishment.
Example: She is a formidable opponent in cross-examination.

EXACTING
Demanding precision and high standards from whoever undertakes it.
Example: Restoration work of this kind is exacting and poorly paid.

INTRACTABLE
Resisting solution or management despite sustained effort; used of problems and disputes.
Example: The dispute over water rights has proved intractable.

Note: 'arduous' describes effort, 'intricate' describes complexity, 'formidable' describes the impression something makes, 'exacting' describes the standard imposed, and 'intractable' describes resistance to any solution. A task can be intricate without being arduous, and formidable without being intractable.""",
    ),
    (
        "beautiful",
        """EXQUISITE
Beautiful through fineness of detail and delicacy of craftsmanship; suits small things.
Example: The embroidery on the cuffs is exquisite.

SUBLIME
Beautiful on a scale that inspires awe bordering on unease; a Romantic register.
Example: There is something sublime in the indifference of those mountains.

RESPLENDENT
Dazzling by brilliance of light, colour or ornament; often used of people in ceremonial dress.
Example: The choir processed in, resplendent in scarlet.

ELEGANT
Beautiful through restraint, proportion and economy; applies equally to dresses, arguments and proofs.
Example: His solution is shorter than mine and considerably more elegant.

PICTURESQUE
Pleasing in a composed, scenic way, as though arranged for a painting; can carry a faintly patronising edge.
Example: They retired to a picturesque fishing village on the coast.

Note: scale and quality of the beauty govern the choice. 'Exquisite' rewards close inspection, 'sublime' overwhelms, 'resplendent' glitters, 'elegant' withholds, and 'picturesque' charms. Calling a cathedral picturesque diminishes it; calling a brooch sublime inflates it.""",
    ),
    (
        "understand",
        """COMPREHEND
To grasp something complex in full; formal, and common in negative constructions.
Example: Few of us can comprehend the scale of the losses.

GRASP
To seize the essential point, often suddenly; idiomatic and slightly less formal.
Example: It took me a second reading to grasp what he was proposing.

DISCERN
To perceive something subtle that is not immediately obvious.
Example: One can discern a pattern in the last four quarters.

FATHOM
To reach the bottom of something puzzling; overwhelmingly used in the negative.
Example: I cannot fathom why she agreed to those terms.

APPREHEND
To perceive and understand directly, often intuitively; philosophical and literary in register.
Example: We apprehend the danger long before we can articulate it.

Note: 'comprehend' concerns completeness, 'grasp' concerns the essential point, 'discern' concerns subtlety, 'fathom' concerns motive and mystery, and 'apprehend' concerns immediate perception. 'I cannot fathom the instructions' says something different from 'I cannot comprehend the instructions': the first suspects hidden sense, the second reports simple failure.""",
    ),
    (
        "problem",
        """PREDICAMENT
An unwelcome situation in which one is caught and from which escape is awkward.
Example: His resignation left the board in an unenviable predicament.

QUANDARY
A state of indecision between two or more courses of action; the difficulty is internal.
Example: I am in something of a quandary about whether to tell her.

CONUNDRUM
An intellectually puzzling problem, often one admitting no clean answer.
Example: Pricing the externality remains an economic conundrum.

IMPEDIMENT
Something that obstructs progress toward a goal; the emphasis is on blockage.
Example: The licensing regime is the chief impediment to entry.

SETBACK
A reversal that delays progress rather than a standing difficulty.
Example: The failed trial was a setback, not a refutation.

Note: 'predicament' describes a situation you are in, 'quandary' a decision you cannot make, 'conundrum' a puzzle you cannot solve, 'impediment' an obstacle in your path, and 'setback' a delay you have suffered. Only 'conundrum' is primarily intellectual, and only 'setback' is inherently temporary.""",
    ),
    (
        "change",
        """TRANSFORM
To change something so thoroughly in nature or character that the result is effectively a new thing.
Example: Cheap sequencing transformed the discipline within a decade.

ALTER
To change something in a particular respect while it remains recognisably itself.
Example: We have altered the opening hours, not the service.

MODIFY
To make limited, deliberate adjustments, usually to improve or adapt something.
Example: The airframe was modified for high-altitude work.

AMEND
To correct or improve a text, rule or agreement; formal and largely legal or parliamentary.
Example: The clause was amended to include subcontractors.

RECALIBRATE
To adjust expectations, standards or instruments against a new reference point; figurative in business register.
Example: We should recalibrate our forecasts after this quarter.

Note: the scale and object differ. 'Transform' implies a change of kind; 'alter' and 'modify' a change of degree, with 'modify' the more purposive of the two; 'amend' takes documents and rules; 'recalibrate' takes expectations and measurements. Amending a policy is not transforming it.""",
    ),
]

# --------------------------------------------------------------------------- #
# Optional: explanation of an enhancement
# --------------------------------------------------------------------------- #

EXPLANATION_EXAMPLES = [
    (
        "This is an ideal opportunity.",
        "This constitutes an exceptionally auspicious opportunity.",
        "'Constitutes' replaces the flat copula and asserts that the thing "
        "qualifies as such, which is more precise than merely equating it. "
        "'Auspicious' captures the forward-looking promise that 'ideal' leaves "
        "implicit, where 'optimal' would have made a claim about efficiency "
        "instead. 'Exceptionally' does the intensifying work without the "
        "vagueness of 'very'. The register rises while the factual claim is "
        "unchanged.",
    ),
    (
        "I want to understand this problem better.",
        "I seek to develop a more nuanced understanding of this predicament.",
        "'Seek to' is more deliberate than 'want to' and suits written register. "
        "Nominalising to 'a more nuanced understanding' allows the comparative "
        "to modify the quality of understanding rather than the vague adverb "
        "'better'. 'Predicament' specifies a situation one is caught in, which "
        "'problem' leaves open. Nothing has been added to the original claim.",
    ),
]


def build_records() -> list[dict]:
    records: list[dict] = []

    for text, tone, answer in ENHANCE_EXAMPLES:
        records.append({"messages": build_enhance_messages(text, tone) + [
            {"role": "assistant", "content": answer}
        ]})

    for word, answer in SYNONYM_EXAMPLES:
        records.append({"messages": build_synonym_messages(word, n=5) + [
            {"role": "assistant", "content": answer}
        ]})

    for original, rewritten, answer in EXPLANATION_EXAMPLES:
        records.append({"messages": build_explanation_messages(original, rewritten) + [
            {"role": "assistant", "content": answer}
        ]})

    return records


def main() -> None:
    records = build_records()
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    with open(DATA_PATH, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Read it back so the script fails loudly if anything is malformed.
    with open(DATA_PATH, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            payload = json.loads(line)
            roles = [m["role"] for m in payload["messages"]]
            assert roles == ["system", "user", "assistant"], (index, roles)

    print(f"Wrote {len(records)} examples to {DATA_PATH}")
    print(
        f"  {len(ENHANCE_EXAMPLES)} enhancement, "
        f"{len(SYNONYM_EXAMPLES)} synonym, "
        f"{len(EXPLANATION_EXAMPLES)} explanation"
    )


if __name__ == "__main__":
    main()
