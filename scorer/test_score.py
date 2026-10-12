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
