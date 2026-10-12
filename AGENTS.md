# Kevlar — agent notes

## Product
Android mid-call scam-intent assistant. Speech chunks → STT → proximity trust scorer → alert.

## Build order
1. Scorer (done in `scorer/`) — text only
2. App shell + demo transcripts
3. Mic capture without a call
4. Cloud STT on 8s chunks
5. AccessibilityService for live-call mic (Cube ACR pattern)
6. Wake on call / unknown number; polish

## Platform constraints
- Ordinary apps cannot use mic during cellular calls on many Samsungs.
- AccessibilityService + `VOICE_RECOGNITION` can keep mic alive during calls (device-dependent).
- Sideload for college demo; Play Store policy blocks accessibility-for-recording.

## Scorer rules of thumb
- Same-category clusters (otp/sms/code) = mild
- Cross-category clusters (credential + account_threat + secrecy) = severe
- Closer words = higher risk
