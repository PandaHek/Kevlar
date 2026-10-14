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


def _ask_negated(tokens: list[str], index: int) -> bool:
    """True when a refusal governs this ask.

    Looks back far enough for 'I will not ask you to share any code',
    including won't, never, and nobody. Stops at a new instruction, so
    'I will not ask for a code. Please share the OTP' is still an ask.
    'If you do not share the code' is still a demand.
    """
    stops = {
        "disconnect", "hang", "helpline", "hotline", "lawyer", "family",
        "anyone", "video", "camera", "branch", "bank", "relative", "relatives",
        "wife", "husband", "son", "daughter", "parents", "brother", "sister",
        "friend", "friends", "worry", "worried", "alone", "confidential",
        "privately", "call", "inform", "leave", "step",
        "please", "but", "however", "instead", "then", "now",
    }
    negations = {
        "not", "never", "don't", "dont", "won't", "wont",
        "can't", "cant", "cannot", "without", "refrain",
        "nobody", "noone",
    }
    for j in range(index - 1, max(-1, index - 13), -1):
        tok = tokens[j]
        if tok in stops:
            return False
        if tok in negations:
            earlier = tokens[max(0, j - 4) : j]
            if "if" in earlier:
                return False
            return True
    return False


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


def _read_aloud_bonus(
    tokens: list[str],
    *,
    window: int,
    bonus: float,
    objects: Iterable[str],
    number_guards: Iterable[str],
    context_words: Iterable[str],
) -> tuple[float, list[str]]:
    """Bonus when a caller tells the user to read a code or number aloud.

    Matches 'read me the code', 'read the number aloud', and 'read out the numeric'.
    Skips 'account number' and 'reference number'. Skips 'do not read the code'.
    Reading a number counts in a banking context. Claiming to be family
    does not soften an ask: the call is from an unknown number.
    """
    object_set = {word.lower() for word in objects}
    guard_set = {word.lower() for word in number_guards}
    context_set = {word.lower() for word in context_words}
    if not any(tok in context_set for tok in tokens):
        return 0.0, []
    for i, tok in enumerate(tokens):
        if tok != "read" or _negated(tokens, i):
            continue
        lo = max(0, i - window)
        hi = min(len(tokens), i + window + 1)
        for j in range(lo, hi):
            if j == i:
                continue
            word = tokens[j]
            if word not in object_set:
                continue
            if word == "number" and j > 0 and tokens[j - 1] in guard_set:
                continue
            return bonus, [
                f"Asked to read a code aloud (request, +{bonus:.2f})"
            ]
    return 0.0, []


def _disclose_number_bonus(
    tokens: list[str],
    *,
    window: int,
    bonus: float,
    otp_bonus: float,
    verbs: Iterable[str],
    number_guards: Iterable[str],
) -> tuple[float, list[str]]:
    """Bonus when any ask sits next to a one-time secret.

    The ask can be read, tell, dictate, convey, spell, share, and the rest
    of the verb list. The secret can be an OTP, code, PIN, password, or
    the digits of a number that just arrived.
    Account numbers and reference numbers do not count.
    'I will not ask you to share any code' does not count.
    A later instruction still does: 'I will not ask for a code. Please share the OTP.'
    'If you do not share the code' is still a demand.
    A secret that merely arrives, with nobody asking for it, does not count.
    """
    verb_set = {verb.lower() for verb in verbs}
    # 'Type the OTP' and 'enter the code' are asks. They must not also
    # match 'enter it only on the official page' next to the word number.
    keyed = {"type", "enter"}
    verb_set.update(keyed)
    guard_set = {word.lower() for word in number_guards}
    plain = {"numeric", "number", "digit", "digits"}
    named = {"otp", "code", "pin", "password", "passcode"}
    secrets: list[tuple[int, float, bool]] = []
    for i, tok in enumerate(tokens):
        if tok in plain:
            if tok == "number" and i > 0 and tokens[i - 1] in guard_set:
                continue
            secrets.append((i, bonus, False))
        elif tok in named:
            secrets.append((i, otp_bonus, True))
    if not secrets:
        return 0.0, []
    best = 0.0
    for i, tok in enumerate(tokens):
        if tok not in verb_set or _ask_negated(tokens, i):
            continue
        for index, amount, is_named in secrets:
            if tok in keyed:
                # The secret has to be what they are told to type.
                # 'A code will appear. Enter it only on the official page' is not an ask.
                if not is_named or index <= i or index - i > window:
                    continue
                best = max(best, amount)
                continue
            if abs(i - index) <= window:
                best = max(best, amount)
    if best <= 0:
        return 0.0, []
    return best, [f"Asked for a one-time number or code (request, +{best:.2f})"]


def _window_has(tokens: list[str], index: int, words: set[str], gap: int) -> bool:
    lo = max(0, index - gap)
    hi = min(len(tokens), index + gap + 1)
    return any(tokens[j] in words and j != index for j in range(lo, hi))


def _money_window_has(tokens: list[str], index: int, words: set[str], gap: int) -> bool:
    """Like _window_has, but a rupee amount does not use up the gap.

    'transfer the extra 63000 back to the recovery account' stays one ask
    even though 63,000 is two tokens.
    """
    for direction in (-1, 1):
        seen = 0
        j = index
        while seen < gap:
            j += direction
            if j < 0 or j >= len(tokens):
                break
            if tokens[j].isdigit():
                continue
            seen += 1
            if tokens[j] in words:
                return True
    return False


def _skip_money_move(tokens: list[str], index: int) -> bool:
    """Drops lookalikes that are not the caller asking to be paid.

    'Transfer you' is a call handoff. 'Will be credited' and 'credit card'
    are not 'please credit this account'. 'Put the account number' is
    writing a number down.
    """
    tok = tokens[index]
    nxt = tokens[index + 1] if index + 1 < len(tokens) else ""
    prev = tokens[index - 1] if index else ""
    if tok == "transfer" and nxt == "you":
        return True
    if tok == "put" and nxt in {"you", "me"}:
        return True
    if tok == "credit" and (nxt == "card" or prev == "card"):
        return True
    if tok == "credit" and prev not in {"please", "kindly", "now", "you", "must", "just", "simply"}:
        return True
    if tok in {"return", "reverse"} and prev in {"will", "be", "being", "been"}:
        return True
    if tok == "put" and _window_has(tokens, index, {"number"}, 4):
        if not _window_has(tokens, index, {"money", "funds", "rupees", "wallet", "upi"}, 6):
            return True
    # 'Pay through the official app or at the branch' names their own channel.
    # 'Pay into this account' does not.
    if tok == "pay" and _window_has(tokens, index, {"official", "branch"}, 4):
        return True
    return False


def _kept_quiet(
    tokens: list[str],
    index: int,
    tok: str,
    topic_set: set[str],
    negation_set: set[str],
    gap: int,
) -> bool:
    """True when this token is the caller cutting the user off from help.

    A named person (lawyer, son, branch, bank) counts only next to a
    contact verb, so 'don't worry, your son is fine' stays ordinary.
    """
    if tok in topic_set and _window_has(tokens, index, negation_set, gap):
        return True
    person = {
        "bank", "branch", "lawyer", "relative", "relatives",
        "wife", "husband", "son", "daughter", "parents",
        "brother", "sister", "friend", "friends",
    }
    contact = {
        "call", "contact", "tell", "inform", "speak", "discuss",
        "visit", "check", "verify", "talk", "informing",
    }
    if (
        tok in person
        and _window_has(tokens, index, negation_set, gap)
        and _window_has(tokens, index, contact, gap)
    ):
        return True
    if (
        tok == "leave"
        and _window_has(tokens, index, negation_set, gap)
        and _window_has(tokens, index, {"room", "house", "home"}, gap)
    ):
        return True
    if tok == "step" and index + 1 < len(tokens) and tokens[index + 1] == "out":
        if _window_has(tokens, index, negation_set, gap):
            return True
    if tok in {"video", "camera"} and (
        _window_has(tokens, index, negation_set, gap)
        or _window_has(tokens, index, {"keep", "stay"}, 4)
    ):
        return True
    return False


def _money_negated(tokens: list[str], index: int) -> bool:
    """True when the payment instruction itself is cancelled.

    Looks back far enough for 'will not ask you to move any money'.
    Stops early if another order sits in between, so 'do not disconnect,
    then transfer the money' is still a payment ask.
    """
    stops = {
        "disconnect", "hang", "helpline", "hotline", "lawyer", "family",
        "anyone", "video", "camera", "branch", "bank", "relative", "relatives",
        "wife", "husband", "son", "daughter", "parents", "brother", "sister",
        "friend", "friends", "worry", "worried", "alone", "confidential",
        "privately", "call", "tell", "inform", "leave", "step",
    }
    negations = {"not", "never", "don't", "dont", "without", "refrain"}
    for j in range(index - 1, max(-1, index - 9), -1):
        tok = tokens[j]
        if tok in stops:
            return False
        if tok in negations:
            # "If you do not pay" is still a demand. "We will not ask you to pay" is not.
            earlier = tokens[max(0, j - 4) : j]
            if "if" in earlier:
                return False
            return True
    return False


def _structure_bonus(
    tokens: list[str],
    *,
    bonus: float,
    payment_bonus: float,
    verbs: Iterable[str],
    pronouns: Iterable[str],
    secrets: Iterable[str],
    lookback: int,
    verb_gap: int,
    money_moves: Iterable[str],
    money_targets: Iterable[str],
    money_gap: int,
    isolation_marks: Iterable[str],
    isolation_topics: Iterable[str],
    isolation_negations: Iterable[str],
    isolation_gap: int,
) -> tuple[float, list[str]]:
    """One call-level pattern, shared by polite code asks and money demands.

    Handoff: the caller names a code, then asks the user to read or share
    'that' or 'it'. The word code and the ask can sit a sentence apart.
    Coercion: the caller asks for money to move and also tells the user
    to stay alone, not to tell anyone, not to hang up, or not to check
    with a lawyer, the bank, or a relative. Those two facts can sit
    anywhere in the call.
    """
    verb_set = {word.lower() for word in verbs}
    pronoun_set = {word.lower() for word in pronouns}
    secret_set = {word.lower() for word in secrets}
    move_set = {word.lower() for word in money_moves}
    target_set = {word.lower() for word in money_targets}
    mark_set = {word.lower() for word in isolation_marks}
    topic_set = {word.lower() for word in isolation_topics}
    negation_set = {word.lower() for word in isolation_negations}

    for i, tok in enumerate(tokens):
        if tok not in verb_set or _negated(tokens, i):
            continue
        lo = i
        hi = min(len(tokens), i + verb_gap + 1)
        for j in range(lo, hi):
            if tokens[j] not in pronoun_set:
                continue
            start = max(0, j - lookback)
            if any(tokens[k] in secret_set for k in range(start, j)):
                return bonus, [
                    f"Asked to hand over a code just mentioned (request, +{bonus:.2f})"
                ]

    money = False
    isolated = any(tok in mark_set for tok in tokens)
    for i, tok in enumerate(tokens):
        if (
            tok in move_set
            and not _skip_money_move(tokens, i)
            and not _money_negated(tokens, i)
            and _money_window_has(tokens, i, target_set, money_gap)
        ):
            if tok == "send" and not _money_window_has(
                tokens, i, target_set - {"account"}, money_gap
            ):
                continue
            money = True
        if (
            tok in {"gpay", "phonepe", "paytm", "googlepay"}
            and not _money_negated(tokens, i)
            and _money_window_has(
                tokens, i, {"money", "thousand", "lakh", "lakhs", "rupees", "rupee"}, money_gap
            )
        ):
            money = True
        if not isolated and _kept_quiet(tokens, i, tok, topic_set, negation_set, isolation_gap):
            isolated = True
        if money and isolated:
            return bonus, [
                f"Asked for money while saying to keep it quiet (request, +{bonus:.2f})"
            ]
    if money:
        return payment_bonus, [
            f"Asked to pay into an account (request, +{payment_bonus:.2f})"
        ]
    return 0.0, []


_PAY_WORDS = {
    "pay", "payment", "payments", "paying", "paid", "repay", "repayment",
    "settle", "settlement", "dues",
}


def _pay_language_near(tokens: list[str], index: int, gap: int) -> bool:
    """True when a pay request sits near this word.

    Covers pay, payment, settle, dues, and 'clear the balance'.
    The word balance on its own does not count.
    """
    if _window_has(tokens, index, _PAY_WORDS, gap):
        return True
    lo = max(0, index - gap)
    hi = min(len(tokens), index + gap + 1)
    for j in range(lo, hi):
        if tokens[j] == "clear" and _window_has(tokens, j, {"balance"}, 4):
            return True
    return False


def _payment_app_in_call(tokens: list[str]) -> bool:
    """True when the caller points at a UPI or payment app."""
    if any(tok in {"upi", "phonepe", "gpay", "paytm", "googlepay"} for tok in tokens):
        return True
    return any(
        tok == "payment" and _window_has(tokens, i, {"app"}, 2)
        for i, tok in enumerate(tokens)
    )


def _checklist_bonus(tokens: list[str], bonus: float) -> tuple[float, list[str], list[str]]:
    """Flags asks that are scams even with no OTP word.

    A call to action is installing an app, opening a file they send, scanning a
    QR or entering a PIN to receive money, or opening a link.
    Police, arrest, customs, a prize, or a QR that is only mentioned are
    returned separately and do not score unless the call also makes an ask.
    A refund that simply arrives, and 'update the official app', do not count.
    """
    found: list[str] = []
    context: list[str] = []
    mandate_in_call = any(tok in {"mandate", "autopay"} for tok in tokens)
    payment_app_in_call = _payment_app_in_call(tokens)
    fillers = {
        "the", "a", "an", "your", "this", "that", "it", "from", "small",
        "i", "i'd", "i'll", "i'm", "i've", "we", "we'd", "we'll", "you",
        "he", "she", "they", "and", "or", "but", "so", "if", "to",
    }
    # "the update failed to install" is not "please install QuickSupport"
    not_a_request = {"failed", "unable", "completed", "finished"}

    for i, tok in enumerate(tokens):
        if tok == "apk":
            found.append("an apk file")
            break
        if tok in {"install", "download"} and not _negated(tokens, i):
            if _window_has(tokens, i, {"app", "apk", "file", "application"}, 6):
                found.append("install an app")
                break
            nxt = i + 1
            already_happened = _window_has(tokens, i, not_a_request, 3)
            if (
                not already_happened
                and nxt < len(tokens)
                and tokens[nxt] not in fillers
            ):
                found.append("install an app")
                break
        if tok == "file" and not _negated(tokens, i):
            opened = _window_has(tokens, i, {"click", "open", "tap"}, 6)
            incoming = _window_has(
                tokens, i, {"send", "sending", "sent", "whatsapp", "sms", "telegram"}, 8
            )
            if opened and incoming:
                found.append("open a file they send")
                break

    receive = {"receive", "receiving", "cashback", "prize", "refund", "money"}
    for i, tok in enumerate(tokens):
        if _negated(tokens, i):
            continue
        qr_here = tok == "qr" or (tok == "scan" and _window_has(tokens, i, {"qr"}, 4))
        if qr_here:
            if _window_has(tokens, i, receive | {"pin", "transfer"}, 10) or _pay_language_near(
                tokens, i, 10
            ):
                found.append("scan a QR")
                break
            context.append("a QR")
            continue
        if tok in {"enter", "type", "put"} and _window_has(tokens, i, {"pin"}, 6):
            if _window_has(tokens, i, receive, 10):
                found.append("enter a PIN to receive money")
                break

    for i, tok in enumerate(tokens):
        if tok in {"http", "https", "www", "bitly", "tinyurl"} or (
            tok == "bit" and i + 1 < len(tokens) and tokens[i + 1] == "ly"
        ):
            found.append("a link to click")
            break
        if tok == "link" and not _negated(tokens, i) and _window_has(
            tokens,
            i,
            {"click", "tap", "open", "sent", "send", "sending", "sms", "message", "text", "whatsapp", "telegram"},
            6,
        ):
            found.append("a link to click")
            break
        if tok in {"portal", "website", "url", "form", "site", "page"} and not _negated(tokens, i) and _window_has(
            tokens,
            i,
            {"send", "sending", "sent", "sms", "click", "tap", "message", "text", "whatsapp", "telegram"},
            8,
        ):
            found.append("a link to click")
            break

    for i, tok in enumerate(tokens):
        if tok in {"share", "sharing"} and not _negated(tokens, i) and _window_has(
            tokens, i, {"screen"}, 3
        ):
            found.append("share your screen")
            break
        if not _negated(tokens, i) and (
            tok == "voucher" or (tok == "gift" and _window_has(tokens, i, {"card", "cards"}, 2))
        ):
            found.append("buy a gift card")
            break
        if tok in {"aadhaar", "aadhar", "pan", "passbook"} and not _negated(tokens, i) and _window_has(
            tokens, i, {"send", "sending", "sent", "share", "whatsapp", "forward", "show", "hold"}, 6
        ):
            found.append("send or show an identity document")
            break
        # Tapping accept or confirm on a payment-app popup is the payment.
        # Callers call that popup a cancellation, a reversal, or a pending item,
        # and they never have to say UPI PIN.
        if tok in {"approve", "accept", "confirm", "authorise", "authorize"} and not _ask_negated(tokens, i):
            popup = {
                "pending", "notification", "cancellation", "collect",
                "reversal", "mandate", "autopay", "request", "pops", "popup",
            }
            near_popup = _window_has(tokens, i, popup, 8)
            near_app = _window_has(
                tokens, i, {"upi", "phonepe", "gpay", "paytm", "googlepay", "payment"}, 8
            )
            names_the_mandate = tok in {"confirm", "authorise", "authorize"} and (
                _window_has(tokens, i, {"mandate", "autopay"}, 4)
                or (
                    mandate_in_call
                    and _window_has(tokens, i, {"request", "cancellation"}, 8)
                )
            )
            approves_collect = tok in {"approve", "accept"} and (
                _window_has(tokens, i, {"collect", "mandate", "autopay"}, 8)
                or (
                    _window_has(tokens, i, {"request"}, 4)
                    and _window_has(tokens, i, {"phonepe", "gpay", "paytm", "googlepay", "upi"}, 8)
                )
            )
            taps_the_popup = (tok in {"approve", "accept"} and near_app) or (
                near_popup and payment_app_in_call
            )
            if names_the_mandate or approves_collect or taps_the_popup:
                found.append("approve a collect or autopay")
                break
        if tok == "read" and not _negated(tokens, i) and _window_has(
            tokens, i, {"message", "messages", "sms"}, 6
        ) and _window_has(tokens, i, {"me", "out", "aloud"}, 6):
            found.append("read a message aloud")
            break

    if any(
        tok in {"police", "cbi", "customs", "arrest"} and not _negated(tokens, i)
        for i, tok in enumerate(tokens)
    ):
        context.append("police, CBI, customs, or arrest")

    if any(tok in {"prize", "cashback", "lottery"} for tok in tokens):
        context.append("a prize or cashback")
    for i, tok in enumerate(tokens):
        if tok == "returns" and _window_has(tokens, i, {"high", "guaranteed"}, 3):
            context.append("very high returns")
            break

    if not found:
        return 0.0, [], context
    return bonus, [
        f"Scam checks: {', '.join(found)} (request, +{bonus:.2f})"
    ], context

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
    # A refused 'share' or 'tell' is not a danger word. A later real ask still is.
    disclose_verbs = {word.lower() for word in cfg.get("disclose_verbs", [])}
    matches = [
        m
        for m in matches
        if m.word not in disclose_verbs or not _ask_negated(tokens, m.index)
    ]
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
    if request_bonus == 0:
        request_bonus, request_reasons = _read_aloud_bonus(
            tokens,
            window=int(cfg.get("read_aloud_window", 6)),
            bonus=float(cfg.get("read_aloud_bonus", 1.15)),
            objects=cfg.get(
                "read_aloud_objects",
                ["code", "otp", "pin", "numeric", "number"],
            ),
            number_guards=cfg.get(
                "read_aloud_number_guards",
                ["account", "reference", "mobile", "phone", "contact", "card", "case"],
            ),
            context_words=cfg.get(
                "read_aloud_context",
                ["account", "verification", "verify", "kyc", "refund", "upi"],
            ),
        )
    if request_bonus == 0:
        request_bonus, request_reasons = _disclose_number_bonus(
            tokens,
            window=int(cfg.get("read_aloud_window", 6)),
            bonus=float(cfg.get("disclose_bonus", 1.7)),
            otp_bonus=float(cfg.get("otp_ask_bonus", 0.85)),
            verbs=cfg.get(
                "disclose_verbs",
                [
                    "read",
                    "tell",
                    "say",
                    "share",
                    "give",
                    "repeat",
                    "dictate",
                    "speak",
                    "provide",
                    "what",
                    "what's",
                ],
            ),
            number_guards=cfg.get(
                "read_aloud_number_guards",
                ["account", "reference", "mobile", "phone", "contact", "card", "case"],
            ),
        )
    if request_bonus == 0:
        request_bonus, request_reasons = _structure_bonus(
            tokens,
            bonus=float(cfg.get("structure_bonus", 1.25)),
            payment_bonus=float(cfg.get("payment_ask_bonus", 0.55)),
            verbs=cfg.get("disclose_verbs", []),
            pronouns=cfg.get("handoff_pronouns", ["that", "it"]),
            secrets=cfg.get("handoff_secrets", ["code", "otp", "pin", "numeric"]),
            lookback=int(cfg.get("handoff_lookback", 20)),
            verb_gap=int(cfg.get("handoff_verb_gap", 4)),
            money_moves=cfg.get(
                "money_moves",
                [
                    "transfer", "deposit", "pay", "move", "send",
                    "neft", "imps", "rtgs", "return", "reverse",
                    "credit", "put", "load",
                ],
            ),
            money_targets=cfg.get(
                "money_targets",
                [
                    "account", "money", "funds", "bond", "rupees", "rupee",
                    "upi", "wallet", "gpay", "phonepe", "paytm", "googlepay",
                    "thousand", "lakh", "lakhs",
                ],
            ),
            money_gap=int(cfg.get("money_gap", 8)),
            isolation_marks=cfg.get("isolation_marks", ["alone", "confidential", "privately"]),
            isolation_topics=cfg.get(
                "isolation_topics",
                [
                    "family",
                    "parties",
                    "disclose",
                    "tell",
                    "anyone",
                    "informing",
                    "contact",
                    "helpline",
                    "hotline",
                    "hang",
                    "disconnect",
                    "video",
                    "camera",
                ],
            ),
            isolation_negations=cfg.get(
                "isolation_negations",
                ["not", "never", "don't", "dont", "without", "refrain"],
            ),
            isolation_gap=int(cfg.get("isolation_gap", 6)),
        )
    checklist_bonus = float(cfg.get("checklist_bonus", 1.65))
    check_bonus, check_reasons, context = _checklist_bonus(tokens, checklist_bonus)
    if context and (request_bonus > 0 or check_bonus > 0):
        if request_bonus > 0 and check_bonus < checklist_bonus:
            check_bonus = checklist_bonus
        check_reasons = [
            *check_reasons,
            "Scam checks: "
            + ", ".join(dict.fromkeys(context))
            + f" with an ask (request, +{check_bonus:.2f})",
        ]
    if check_bonus > request_bonus:
        request_bonus = check_bonus
    request_reasons = [*request_reasons, *check_reasons]
    # Unknown number only. "Dad, I'm your son, transfer the money" is a scam,
    # including a cloned voice. "Hi dad, see you for dinner" has no ask.
    family_claim = {
        "dad", "daddy", "papa", "mom", "mummy", "mama", "mother", "father",
        "beta", "son", "daughter", "uncle", "aunt", "brother", "sister", "papa",
    }
    if "withdraw" in tokens and 0 < request_bonus < checklist_bonus:
        request_bonus = checklist_bonus
        request_reasons.append(
            "Asked to pay in order to withdraw "
            f"(request, +{request_bonus:.2f})"
        )
    if any(tok in family_claim for tok in tokens) and 0 < request_bonus < checklist_bonus:
        request_bonus = checklist_bonus
        request_reasons.append(
            "Caller claims to be family and makes an ask "
            f"(request, +{request_bonus:.2f})"
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

    if matches:
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
    else:
        pair_sum = 0.0
        pair_reasons = []
        density = 0.0
        categories_hit = set()
        density_scale = 0.0
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
    reasons: list[str] = []
    if unique_words:
        reasons.append(f"Danger words: {', '.join(unique_words[:10])}")
    reasons.extend(request_reasons)
    reasons.extend(pair_reasons)

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
