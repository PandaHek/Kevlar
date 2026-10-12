"""Kevlar trust scorer: proximity + density of danger words → risk/trust."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Match:
    word: str
    weight: float
    index: int
    category: str = "unknown"


@dataclass
class ScoreResult:
    trust: float
    risk: float
    raw: float
    density: float
    pair_sum: float
    matches: list[Match] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    level: str = "safe"

    def to_dict(self) -> dict:
        return {
            "trust": round(self.trust, 1),
            "risk": round(self.risk, 1),
            "raw": round(self.raw, 3),
            "density": round(self.density, 3),
            "pair_sum": round(self.pair_sum, 3),
            "level": self.level,
            "reasons": self.reasons,
            "matches": [
                {"word": m.word, "weight": m.weight, "index": m.index}
                for m in self.matches
            ],
        }


def load_config(path: Path | None = None) -> dict:
    config_path = path or Path(__file__).with_name("words.json")
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def tokenize(text: str) -> list[str]:
    text = (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
    return WORD_RE.findall(text.lower())


def _phrase_starts(tokens: list[str], phrase: str) -> list[int]:
    parts = phrase.lower().split()
    width = len(parts)
    if width == 0:
        return []
    return [
        i
        for i in range(len(tokens) - width + 1)
        if tokens[i : i + width] == parts
    ]


def _negated(tokens: list[str], index: int) -> bool:
    """True when don't / never / not sits just before the phrase."""
    prev = tokens[max(0, index - 4) : index]
    return any(tok in {"not", "never", "don't", "dont"} for tok in prev)


def _secrecy_phrase_matches(tokens: list[str], phrases: Iterable[str]) -> list[Match]:
    found: list[Match] = []
    for phrase in phrases:
        for start in _phrase_starts(tokens, phrase):
            if _negated(tokens, start):
                continue
            found.append(
                Match(word=phrase, weight=0.6, index=start, category="secrecy")
            )
    return found


def _request_phrase_bonus(
    tokens: list[str],
    matches: list[Match],
    phrases: Iterable[str],
    window: int,
    bonus: float,
) -> tuple[float, list[str]]:
    """Bonus when the caller asks someone to read a code aloud.

    'Read it out' next to a code or the word verification counts.
    'Never read it out' and 'do not read it out' do not.
    """
    credential_indexes = [m.index for m in matches if m.category == "credential"]
    for phrase in phrases:
        for start in _phrase_starts(tokens, phrase):
            if _negated(tokens, start):
                continue
            near_credential = any(
                abs(idx - start) <= window for idx in credential_indexes
            )
            if not near_credential:
                lo = max(0, start - window)
                hi = min(len(tokens), start + window)
                near_credential = "verification" in tokens[lo:hi]
            if not near_credential:
                continue
            return bonus, [
                f"Asked to read a code aloud (request, +{bonus:.2f})"
            ]
    return 0.0, []


def _upi_pin_request_bonus(
    tokens: list[str],
    verbs: Iterable[str],
    window: int,
    bonus: float,
) -> tuple[float, list[str]]:
    """Bonus when the caller asks someone to authorise or enter a UPI PIN.

    'Never authorise with your UPI PIN' does not count.
    A door PIN, or the word pin with no UPI beside it, does not count.
    """
    verb_set = {verb.lower() for verb in verbs}
    pin_indexes = [
        i
        for i, tok in enumerate(tokens)
        if tok == "pin" and "upi" in tokens[max(0, i - 3) : min(len(tokens), i + 4)]
    ]
    if not pin_indexes:
        return 0.0, []
    for i, tok in enumerate(tokens):
        if tok not in verb_set or _negated(tokens, i):
            continue
        if any(abs(i - pin_i) <= window for pin_i in pin_indexes):
            return bonus, [
                f"Asked to authorise with a UPI PIN (request, +{bonus:.2f})"
            ]
    return 0.0, []


def level_for_trust(trust: float) -> str:
    if trust >= 80:
        return "safe"
    if trust >= 50:
        return "caution"
    if trust >= 20:
        return "suspicious"
    return "likely_scam"


def _category_lookup(config: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for category, words in config.get("categories", {}).items():
        for word in words:
            lookup[word] = category
    return lookup


def _find_matches(
    tokens: list[str], weights: dict[str, float], categories: dict[str, str]
) -> list[Match]:
    matches: list[Match] = []
    for i, tok in enumerate(tokens):
        if tok in weights:
            matches.append(
                Match(
                    word=tok,
                    weight=float(weights[tok]),
                    index=i,
                    category=categories.get(tok, "unknown"),
                )
            )
    return matches


def _pair_contributions(
    matches: list[Match],
    window_size: int,
    *,
    same_category_scale: float = 0.25,
    cross_category_scale: float = 1.0,
) -> tuple[float, list[str]]:
    """Sum weight_a * weight_b * closeness * category_scale for nearby pairs.

    Same-category pairs (otp+sms+code) count little.
    Cross-category pairs (otp+blocked, transfer+secrecy) count fully.
    """
    pair_sum = 0.0
    reasons: list[str] = []
    seen_reason_keys: set[tuple[str, str]] = set()

    for i, a in enumerate(matches):
        for b in matches[i + 1 :]:
            distance = b.index - a.index
            if distance <= 0 or distance >= window_size:
                continue
            closeness = 1.0 - (distance / window_size)
            same = a.category == b.category and a.category != "unknown"
            scale = same_category_scale if same else cross_category_scale
            contribution = a.weight * b.weight * closeness * scale
            pair_sum += contribution

            key = tuple(sorted((a.word, b.word)))
            if key not in seen_reason_keys and contribution >= 0.12:
                seen_reason_keys.add(key)
                tag = "same-family" if same else "cross-signal"
                reasons.append(
                    f"'{a.word}' near '{b.word}' ({tag}, gap={distance}, +{contribution:.2f})"
                )

    reasons.sort(key=lambda r: -float(r.rsplit("+", 1)[-1].rstrip(")")))
    return pair_sum, reasons[:8]


def score_text(
    text: str,
    config: dict | None = None,
    *,
    previous_risk: float = 0.0,
    seconds_since_last_hit: float | None = None,
) -> ScoreResult:
    """
    Score transcript text.

    risk = 100 * (1 - e^(-raw))
    raw  = pair contributions + density + request-phrase bonus
    density = (# danger words in token window) / window_size  (capped view of packing)

    Optional decay: if seconds_since_last_hit is set and no new matches,
    previous_risk *= decay_factor each decay_seconds interval.
    """
    cfg = config or load_config()
    weights: dict[str, float] = cfg["weights"]
    window_size = int(cfg.get("window_size", 12))
    decay_factor = float(cfg.get("decay_factor", 0.8))
    decay_seconds = float(cfg.get("decay_seconds", 5))

    tokens = tokenize(text)
    categories = _category_lookup(cfg)
    matches = _find_matches(tokens, weights, categories)
    matches.extend(
        _secrecy_phrase_matches(tokens, cfg.get("secrecy_phrases", []))
    )
    matches.sort(key=lambda m: m.index)
    request_window = int(cfg.get("request_window", 20))
    request_bonus_value = float(cfg.get("request_bonus", 0.9))
    request_bonus, request_reasons = _request_phrase_bonus(
        tokens,
        matches,
        cfg.get("request_phrases", []),
        request_window,
        request_bonus_value,
    )
    if request_bonus == 0:
        request_bonus, request_reasons = _upi_pin_request_bonus(
            tokens,
            cfg.get("pin_request_verbs", []),
            request_window,
            request_bonus_value,
        )

    risk = float(previous_risk)
    if not matches and request_bonus == 0:
        if seconds_since_last_hit is not None and seconds_since_last_hit > 0:
            steps = int(seconds_since_last_hit // decay_seconds)
            for _ in range(steps):
                risk *= decay_factor
        trust = max(0.0, min(100.0, 100.0 - risk))
        return ScoreResult(
            trust=trust,
            risk=risk,
            raw=0.0,
            density=0.0,
            pair_sum=0.0,
            matches=[],
            reasons=["No danger words detected"],
            level=level_for_trust(trust),
        )

    pair_sum, pair_reasons = _pair_contributions(
        matches,
        window_size,
        same_category_scale=float(cfg.get("same_category_pair_scale", 0.25)),
        cross_category_scale=float(cfg.get("cross_category_pair_scale", 1.0)),
    )
    # Density: how packed danger tokens are relative to span they occupy.
    # Same-family-only clusters (otp+sms+code) get a weaker density push than
    # multi-signal scam scripts (credential + threat + secrecy).
    span = max(1, matches[-1].index - matches[0].index + 1)
    density = len(matches) / max(span, window_size)
    categories_hit = {m.category for m in matches}
    density_scale = 0.6 if len(categories_hit) <= 1 else 1.8
    raw = pair_sum + (density_scale * density) + request_bonus
    chunk_risk = 100.0 * (1.0 - math.exp(-raw))
    # Rising risk takes the max of decayed previous and new chunk signal,
    # then blends upward so repeated scam chunks escalate.
    risk = max(risk * decay_factor, chunk_risk)
    if len(categories_hit) >= 2:
        risk = max(risk, min(100.0, risk + 0.2 * chunk_risk))
    risk = min(100.0, risk)

    best_weight: dict[str, float] = {}
    for m in matches:
        best_weight[m.word] = max(best_weight.get(m.word, 0.0), m.weight)
    unique_words = sorted(best_weight, key=lambda w: -best_weight[w])
    reasons = (
        [f"Danger words: {', '.join(unique_words[:10])}"]
        + request_reasons
        + pair_reasons
    )

    trust = max(0.0, min(100.0, 100.0 - risk))
    return ScoreResult(
        trust=trust,
        risk=risk,
        raw=raw,
        density=density,
        pair_sum=pair_sum,
        matches=matches,
        reasons=reasons,
        level=level_for_trust(trust),
    )


def score_chunks(
    chunks: Iterable[str], config: dict | None = None
) -> list[ScoreResult]:
    """Score a sequence of transcript chunks, carrying risk forward with decay."""
    cfg = config or load_config()
    results: list[ScoreResult] = []
    risk = 0.0
    for chunk in chunks:
        result = score_text(chunk, cfg, previous_risk=risk)
        risk = result.risk
        results.append(result)
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Kevlar trust scorer")
    parser.add_argument("text", nargs="?", help="Transcript text to score")
    parser.add_argument(
        "-f", "--file", type=Path, help="Read transcript from a text file"
    )
    parser.add_argument(
        "-c", "--config", type=Path, default=None, help="Path to words.json"
    )
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text or --file")

    result = score_text(text, load_config(args.config) if args.config else None)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
