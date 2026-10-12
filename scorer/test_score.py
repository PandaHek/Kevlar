"""Tests for Kevlar proximity trust scorer."""

from score import load_config, score_chunks, score_text


CFG = load_config()


def test_normal_family_call_stays_safe():
    text = (
        "Hi dad, how are you? I will come home for dinner tomorrow. "
        "Please buy milk and vegetables if you go to the shop."
    )
    result = score_text(text, CFG)
    assert result.trust >= 80
    assert result.level == "safe"
    assert result.risk < 20


def test_son_asking_otp_alone_is_not_hard_alert():
    # One credential ask without account-threat / secrecy cluster
    text = (
        "Dad, can you read me the OTP from the SMS? "
        "I am buying headphones online and need the code."
    )
    result = score_text(text, CFG)
    assert result.trust >= 50, result.to_dict()
    assert result.level in {"safe", "caution"}
    assert result.risk < 60


def test_bank_kyc_scam_triggers_high_risk():
    text = (
        "This is calling from your bank. Your account will be blocked today "
        "if KYC is not completed. Please share the OTP from the SMS now. "
        "Do not tell anyone about this verification call."
    )
    result = score_text(text, CFG)
    assert result.risk >= 60, result.to_dict()
    assert result.trust <= 40
    assert result.level in {"suspicious", "likely_scam"}
    assert any("otp" in r.lower() or "account" in r.lower() for r in result.reasons)


def test_closer_words_are_riskier_than_far_words():
    close = score_text("your account blocked share otp immediately", CFG)
    far = score_text(
        "your account "
        + ("hello " * 20)
        + "blocked "
        + ("hello " * 20)
        + "share otp immediately",
        CFG,
    )
    assert close.risk > far.risk, (close.risk, far.risk)


def test_fake_relative_money_secrecy_cluster():
    text = (
        "Uncle please send money now on GPay. There was an accident. "
        "Do not tell anyone in the family. Transfer immediately."
    )
    result = score_text(text, CFG)
    assert result.risk >= 45, result.to_dict()
    assert result.trust <= 55


def test_remote_access_app_is_high_signal():
    text = "Install AnyDesk and tell me the code so I can verify your refund."
    result = score_text(text, CFG)
    assert result.risk >= 40, result.to_dict()


def test_chunk_escalation_across_call():
    chunks = [
        "Hello sir this is regarding your bank account.",
        "Your account may be blocked if you do not complete KYC.",
        "Share the OTP from SMS and do not tell anyone.",
    ]
    results = score_chunks(chunks, CFG)
    assert results[0].risk < results[-1].risk
    assert results[-1].risk >= 55, [r.to_dict() for r in results]


def test_polite_bank_code_ask_is_a_scam():
    # KYC + refund + "read the verification code out" + "between us".
    # The word OTP never appears.
    text = (
        "Good morning, Mrs. Rao, this is Priya from the Canara Bank customer care team. "
        "I’m calling about two small things that I’d like to sort out for you today, "
        "so you don’t have to worry about them later. First, our system shows your KYC "
        "update is pending, and accounts with pending KYC can have transactions restricted "
        "from the end of this month. Second, there was a failed transaction and the amount "
        "was deducted but has not been refunded. I can push that refund through today as well. "
        "To confirm that I’m speaking to the right person, a six-digit verification code will "
        "arrive on your phone in a moment. Could you just read it out to me? One last thing: "
        "I’d request you to keep this between us until the ticket is closed, since discussing "
        "it with others can delay the refund."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.trust <= 20
    assert result.level == "likely_scam"
    assert any("read a code" in r.lower() for r in result.reasons)


def test_genuine_bank_update_stays_below_scam():
    # A real bank can mention KYC, a restriction, a refund, and an in-app code.
    text = (
        "This is Canara Bank. Your KYC is pending. Please update it in the official app. "
        "Transactions may be restricted until then. A refund of twelve hundred rupees will "
        "be credited today. You may see a verification code inside the app. Enter it only "
        "on the official page."
    )
    result = score_text(text, CFG)
    # KYC + refund + an in-app code can look uneasy. It must not become a scam alert.
    assert result.risk < 70, result.to_dict()
    assert result.level != "likely_scam"


def test_do_not_read_it_out_is_not_a_request():
    base = (
        "Your KYC is pending and a refund is due today. A verification code will arrive "
        "in the app."
    )
    warned = score_text(base + " Do not read it out to any caller.", CFG)
    asked = score_text(base + " Could you read it out to me?", CFG)
    assert not any("read a code" in r.lower() for r in warned.reasons)
    assert any("read a code" in r.lower() for r in asked.reasons)
    assert asked.risk > warned.risk


def test_read_it_out_without_a_code_stays_safe():
    text = "Hi dad, can you read it out to me, the grocery list for dinner tomorrow?"
    result = score_text(text, CFG)
    assert result.trust >= 80, result.to_dict()
    assert result.level == "safe"


def test_polite_upi_pin_ask_is_a_scam():
    text = (
        "Good afternoon, Mr. Kumar, and thank you for taking my call. My name is Anil Mehta, "
        "and I’m with the HDFC Bank customer protection team. Please don’t be alarmed, there’s "
        "nothing wrong with your account. This is a courtesy call, because our monitoring system "
        "flagged an unusual UPI payment of twenty-four thousand nine hundred ninety-nine rupees "
        "earlier today, and we’d like to make sure it was genuinely you. I fully understand if "
        "you’d like to verify me, and you absolutely may, but I’d gently mention that the hold "
        "on that transaction is time-sensitive, so it would be best to resolve it within the hour. "
        "As a precaution, we’ve placed a temporary restriction on the account, which we can lift "
        "together in a couple of minutes. For your safety, could you kindly confirm the last four "
        "digits of your account number? Thank you, that’s very helpful. I can see the amount is "
        "eligible for an immediate reversal. You’ll receive a notification on your payment app, "
        "and to accept the refund, you’d simply authorise it with your UPI PIN. I know that sounds "
        "unusual, so let me reassure you, this is a standard security step that our customers "
        "complete every day, and it’s perfectly safe on your end. One small request: since the "
        "matter is under active review by our fraud team, I’d appreciate your discretion until "
        "it’s closed, as discussing it with others can complicate the investigation. If we can "
        "complete the authorisation now, I’ll personally ensure your account is cleared and you "
        "won’t need to visit the branch at all."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.trust <= 20
    assert result.level == "likely_scam"
    assert any("upi pin" in r.lower() for r in result.reasons)


def test_never_authorise_upi_pin_is_not_a_request():
    warned = score_text(
        "Never authorise a refund with your UPI PIN if a caller asks you to.",
        CFG,
    )
    asked = score_text(
        "You would simply authorise it with your UPI PIN to accept the refund.",
        CFG,
    )
    assert not any("asked to authorise" in r.lower() for r in warned.reasons)
    assert any("asked to authorise" in r.lower() for r in asked.reasons)


def test_decay_without_new_danger_words():
    hot = score_text(
        "account blocked share otp do not tell anyone", CFG, previous_risk=0
    )
    cooled = score_text(
        "okay thank you for your time goodbye",
        CFG,
        previous_risk=hot.risk,
        seconds_since_last_hit=20,
    )
    assert cooled.risk < hot.risk
