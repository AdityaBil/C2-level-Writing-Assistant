"""
prompts.py
==========

System instructions, mode prompts, style modifiers, and parsing utilities
for the Elite C1/C2 English Writing Assistant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Elite C1/C2 System Prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are an Elite C1/C2 English Writing and Vocabulary Assistant.

Your purpose is to help users express themselves with sophisticated, precise, elegant, idiomatic, and natural English.

When the user provides a phrase or sentence, elevate it to C1/C2 English while preserving the original meaning, factual content, intent, and emotional register.

Do not merely replace ordinary words with obscure synonyms. Prefer contextual refinement, sophisticated syntax, precise vocabulary, idiomatic expression, and elegant phrasing.

When the user provides a single word, provide several sophisticated C1/C2 alternatives and explain the semantic distinction between them.

Never assume synonyms are universally interchangeable. Always consider context, register, connotation, grammatical compatibility, and intended meaning.

Avoid archaic, pretentious, unnatural, or unnecessarily convoluted language.

The goal is not to make English complicated.
The goal is to make English precise, elegant, expressive, intellectually mature, and natural to an educated native speaker."""


# --------------------------------------------------------------------------- #
# Style Presets & Guidelines
# --------------------------------------------------------------------------- #

STYLE_GUIDELINES: Dict[str, str] = {
    "Natural C2": (
        "Adopt an effortless, naturally sophisticated C2 register typical of an "
        "educated native speaker. Emphasize idiomatic precision, syntactic grace, and fluid cadence."
    ),
    "Academic": (
        "Adopt a rigorous scholarly tone. Prioritize analytical precision, academic "
        "nominalizations, objective phrasing, and formal intellectual discourse."
    ),
    "Literary": (
        "Infuse the prose with subtle stylistic depth, evocative texture, rhythmic cadence, "
        "and nuanced phrasing suitable for high-end literary prose."
    ),
    "Philosophical": (
        "Emphasize conceptual clarity, dialectical rigor, epistemological nuance, "
        "and deep intellectual precision."
    ),
    "Poetic": (
        "Harmonize evocative vocabulary with lyrical cadence, acoustic resonance, "
        "and resonant imagery without becoming esoteric or opaque."
    ),
    "Formal": (
        "Maintain an immaculate, diplomatic, and executive register with clear structure, "
        "unambiguous decorum, and authoritative polish."
    ),
    "Concise": (
        "Maximize syntactic and lexical efficiency. Eliminate redundancies and tautologies, "
        "delivering high information density with crisp, punchy C2 elegance."
    ),
}

AVAILABLE_STYLES = list(STYLE_GUIDELINES.keys())
DEFAULT_STYLE = "Natural C2"


# --------------------------------------------------------------------------- #
# Mode 1: Enhance Phrase Instructions
# --------------------------------------------------------------------------- #

ENHANCE_USER_TEMPLATE = """Elevate the following text into sophisticated C1/C2 English.

Style guidance: {style_guideline}

Guidelines:
1. Preserve the core meaning, factual accuracy, emotional tone, and communicative intent.
2. Refine syntax, cadence, and vocabulary to achieve effortless sophistication and precision.
3. Avoid overly verbose or archaic words that distract from natural clarity.

Format your response exactly as follows:

### C2 VERSION
[Provide the single best rewritten sentence or passage here, with no surrounding quotes]

### VOCABULARY & SYNTAX NOTE
[Provide 2-3 brief bullet points highlighting specific lexical upgrades or syntactic restructuring used and why they enhance precision or register]

---
Text to enhance:
"{text}"
"""


# --------------------------------------------------------------------------- #
# Mode 2: C2 Synonyms Instructions
# --------------------------------------------------------------------------- #

SYNONYMS_USER_TEMPLATE = """Provide {num_alternatives} sophisticated C1/C2 alternatives for the following word or concept: "{word}"

Style context: {style_guideline}

Crucial Requirements:
- Explain the precise semantic distinction, subtle nuance, connotation, register (e.g. academic, literary, formal, conversational), and exact situational fit for each alternative.
- Provide a natural, high-caliber example sentence demonstrating each word in authentic C2 context.
- Conclude with a dedicated note explaining why these synonyms are NOT universally interchangeable and how improper substitution causes semantic drift.

Format your response exactly in this structured format:

### C2 SYNONYMS

1. **[WORD_1]**
- **Nuance & Register:** [Specific nuance, connotation, and appropriate context]
- **Example:** "[Natural C2 example sentence]"

2. **[WORD_2]**
- **Nuance & Register:** [Specific nuance, connotation, and appropriate context]
- **Example:** "[Natural C2 example sentence]"

(Continue for all {num_alternatives} words)

### CONTEXTUAL INTERCHANGEABILITY NOTE
[Explain why these terms cannot be swapped blindly, detailing key context/collocation boundaries]
"""


# --------------------------------------------------------------------------- #
# Mode 3: Polish Instructions
# --------------------------------------------------------------------------- #

POLISH_USER_TEMPLATE = """Polish and refine the following text while strictly preserving the author's authentic personal voice, perspective, and core sentence structure.

Style guidance: {style_guideline}

Guidelines:
1. Do NOT fundamentally rewrite the text into a completely different voice.
2. Fix subtle awkwardness, improve rhythmic flow, tighten word choice, and heighten elegance.
3. Maintain the original author's intimacy and natural style while raising it to C1/C2 level polish.

Format your response exactly as follows:

### POLISHED VERSION
[Provide the polished text here, with no surrounding quotes]

### REFINEMENT HIGHLIGHTS
[Provide 2-3 brief bullet points summarizing subtle changes made to rhythm, word choice, or clarity]

---
Text to polish:
"{text}"
"""


# --------------------------------------------------------------------------- #
# Prompt Construction Helpers
# --------------------------------------------------------------------------- #

def build_enhance_prompt(text: str, style: str = DEFAULT_STYLE) -> List[Dict[str, str]]:
    """Build messages payload for Mode 1: Enhance Phrase."""
    style_guideline = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES[DEFAULT_STYLE])
    user_prompt = ENHANCE_USER_TEMPLATE.format(
        text=text.strip(),
        style_guideline=style_guideline,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_synonyms_prompt(word: str, style: str = DEFAULT_STYLE, n: int = 6) -> List[Dict[str, str]]:
    """Build messages payload for Mode 2: C2 Synonyms."""
    style_guideline = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES[DEFAULT_STYLE])
    user_prompt = SYNONYMS_USER_TEMPLATE.format(
        word=word.strip(),
        style_guideline=style_guideline,
        num_alternatives=n,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_polish_prompt(text: str, style: str = DEFAULT_STYLE) -> List[Dict[str, str]]:
    """Build messages payload for Mode 3: Polish."""
    style_guideline = STYLE_GUIDELINES.get(style, STYLE_GUIDELINES[DEFAULT_STYLE])
    user_prompt = POLISH_USER_TEMPLATE.format(
        text=text.strip(),
        style_guideline=style_guideline,
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --------------------------------------------------------------------------- #
# Response Parsing & Structuring Helpers
# --------------------------------------------------------------------------- #

@dataclass
class SynonymItem:
    term: str
    nuance: str
    example: str


@dataclass
class ParsedSynonymResult:
    items: List[SynonymItem]
    note: str
    raw_text: str


@dataclass
class ParsedTextResult:
    main_text: str
    notes: str
    raw_text: str


def parse_enhance_or_polish_response(raw_text: str, mode: str = "enhance") -> ParsedTextResult:
    """Extract main text and accompanying notes from LLM output with fallback."""
    cleaned = raw_text.strip()

    main_header_pattern = r"###\s*(?:C2\s*VERSION|POLISHED\s*VERSION)"
    notes_header_pattern = r"###\s*(?:VOCABULARY\s*&\s*SYNTAX\s*NOTE|REFINEMENT\s*HIGHLIGHTS|VOCABULARY\s*NOTE|NOTES?)"

    # Split on the notes header
    parts = re.split(notes_header_pattern, cleaned, flags=re.IGNORECASE)

    if len(parts) >= 2:
        main_part = parts[0]
        notes_part = parts[1].strip()

        # Remove main header if present
        main_part = re.sub(main_header_pattern, "", main_part, flags=re.IGNORECASE).strip()
        return ParsedTextResult(main_text=main_part, notes=notes_part, raw_text=cleaned)

    # Fallback if no explicit headers found
    return ParsedTextResult(main_text=cleaned, notes="", raw_text=cleaned)


def parse_synonyms_response(raw_text: str) -> ParsedSynonymResult:
    """Parse structured synonyms output into item cards and contextual note."""
    cleaned = raw_text.strip()
    items: List[SynonymItem] = []
    note = ""

    # Split out the interchangeability note
    note_split = re.split(r"###\s*(?:CONTEXTUAL\s*INTERCHANGEABILITY\s*NOTE|NOTE:?|INTERCHANGEABILITY)", cleaned, flags=re.IGNORECASE)
    body_text = note_split[0]
    if len(note_split) > 1:
        note = note_split[1].strip()

    # Match numbered items: e.g. 1. **WORD** or 1. WORD
    # and subsequent nuance and example lines
    item_blocks = re.split(r"\n(?=\d+[\.\)]\s+)", body_text)

    for block in item_blocks:
        block = block.strip()
        if not block or not re.match(r"^\d+[\.\)]", block):
            continue

        # Extract word term
        first_line = block.split("\n")[0]
        term_match = re.search(r"^\d+[\.\)]\s*(?:\*\*)?([A-Za-z\s\-\'/]+)(?:\*\*)?", first_line)
        term = term_match.group(1).strip() if term_match else first_line

        # Extract nuance
        nuance_match = re.search(r"(?:-\s*)?\*\*Nuance.*?\:\*\*\s*(.*?)(?=\n(?:-\s*)?\*\*Example|\Z)", block, flags=re.DOTALL | re.IGNORECASE)
        if not nuance_match:
            nuance_match = re.search(r"(?:Nuance|Meaning|Context)\:\s*(.*?)(?=\n.*Example|\Z)", block, flags=re.DOTALL | re.IGNORECASE)
        nuance = nuance_match.group(1).strip() if nuance_match else ""

        # Extract example
        example_match = re.search(r"(?:-\s*)?\*\*Example.*?\:\*\*\s*(.*?)(?=\Z|\n\d+[\.\)])", block, flags=re.DOTALL | re.IGNORECASE)
        if not example_match:
            example_match = re.search(r"(?:Example)\:\s*(.*?)(?=\Z|\n\d+[\.\)])", block, flags=re.DOTALL | re.IGNORECASE)
        example = example_match.group(1).strip().strip('"').strip("'") if example_match else ""

        if term:
            items.append(SynonymItem(term=term, nuance=nuance, example=example))

    return ParsedSynonymResult(items=items, note=note, raw_text=cleaned)
