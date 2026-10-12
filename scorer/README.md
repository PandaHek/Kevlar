# Kevlar Scorer (Step 1)

Proximity-based scam-intent scorer. No Android, no mic — paste transcript text and get a trust score.

## Idea

- Danger words have weights and categories (`credential`, `account_threat`, `secrecy`, …).
- Words **closer together** raise risk more.
- **Same-family** pairs (e.g. `otp` + `sms` + `code`) count little → son asking for OTP stays mostly safe.
- **Cross-signal** pairs (e.g. `account` + `blocked` + `otp` + `don't tell`) count fully → bank KYC scam goes red.
- `risk = 100 * (1 - e^(-raw))`, `trust = 100 - risk`.

## Run

```bash
cd scorer
pip install pytest
python score.py "Your account will be blocked. Share the OTP. Do not tell anyone."
python -m pytest test_score.py -v
```

## Expected demo numbers

| Transcript | Trust (approx) |
|---|---|
| Normal family chat | ~100 (safe) |
| Son asks for OTP only | ~80 (caution) |
| Bank KYC + OTP + secrecy | ~0–20 (likely scam) |

## Next step

Wire this logic into the Android app (Kotlin port of the same math + `words.json`).
