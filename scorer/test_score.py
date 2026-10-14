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


def test_son_asking_otp_alone_is_a_disclosure():
    # An unknown caller asking to hear the OTP is the same act as any other code ask.
    text = (
        "Dad, can you read me the OTP from the SMS? "
        "I am buying headphones online and need the code."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.level == "likely_scam"


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
        + "share "
        + ("hello " * 20)
        + "otp immediately",
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


def test_otp_paraphrases_without_the_word_read():
    samples = [
        "Tell me the six-digit number in the message.",
        "What is the numeric confirmation value on your screen?",
        "Say the six-digit number aloud.",
        "Share the number that just arrived.",
        "Repeat the numeric value to me.",
    ]
    for text in samples:
        result = score_text(text, CFG)
        assert result.risk >= 80, (text, result.to_dict())
        assert result.level == "likely_scam", text
        assert any("one-time number" in reason.lower() for reason in result.reasons)


def test_a_six_digit_number_on_screen_is_not_a_request():
    text = (
        "A six-digit number will appear in the app. Enter it only on the official page."
    )
    result = score_text(text, CFG)
    assert not any("one-time number" in reason.lower() for reason in result.reasons)
    assert result.level != "likely_scam", result.to_dict()


def test_never_say_the_six_digit_number_is_not_a_request():
    text = "Never say the six-digit number to a caller."
    result = score_text(text, CFG)
    assert not any("one-time number" in reason.lower() for reason in result.reasons)
    assert result.risk < 40, result.to_dict()


def test_refusing_to_ask_for_a_code_is_not_a_request():
    reminder = (
        "Good morning, I am calling from the loan servicing team regarding your EMI "
        "of 8200 due on the fifth of this month. As of today, the instalment has not "
        "been received, so I am calling to remind you before a late fee applies. "
        "Please make sure the amount is available in the account linked to your loan, "
        "or pay through the official app or at the branch. I will not ask you to "
        "approve any request or share any code. If you have already paid, please "
        "ignore this call. If you are facing difficulty, you can visit the branch "
        "and speak to the relationship manager about restructuring options."
    )
    result = score_text(reminder, CFG)
    assert result.level != "likely_scam", result.to_dict()
    assert result.risk < 50, result.to_dict()
    assert not any("one-time number" in reason.lower() for reason in result.reasons)

    refused = [
        "I won't ask you to tell me the PIN.",
        "Nobody here will ask you to share an OTP.",
        "We are not going to ask you to read the password.",
    ]
    for text in refused:
        stayed = score_text(text, CFG)
        assert not any("one-time number" in reason.lower() for reason in stayed.reasons), (
            text,
            stayed.to_dict(),
        )
        assert stayed.level != "likely_scam", (text, stayed.to_dict())

    still = [
        "I will not ask for a code. Please share the OTP on your screen.",
        "If you do not share the code, your account will be blocked.",
        "Do not hang up. Share the OTP.",
    ]
    for text in still:
        asked = score_text(text, CFG)
        assert asked.level == "likely_scam", (text, asked.to_dict())
        assert any("one-time number" in reason.lower() for reason in asked.reasons), (
            text,
            asked.to_dict(),
        )


def test_do_not_move_money_is_not_a_payment_demand():
    text = (
        "Please visit the branch with your passbook. "
        "We will not ask you to read a message or move any money on this call."
    )
    result = score_text(text, CFG)
    assert result.risk < 40, result.to_dict()
    assert not any("pay into an account" in reason.lower() for reason in result.reasons)


def test_read_the_reference_number_is_not_a_code_ask():
    text = (
        "Your account is fine. Please read the reference number on the receipt "
        "and keep it for your records. A refund will arrive today."
    )
    result = score_text(text, CFG)
    assert not any("read a code" in reason.lower() for reason in result.reasons)


def test_checklist_signals_are_scams():
    samples = [
        "Scan this QR and enter your PIN to receive the refund.",
        "The file is benefit.apk. Open it and allow the permission.",
        "Click the link I just sent and the prize page will open.",
        "You are under arrest. Transfer the money to this account.",
        "You have won the lottery. Pay the tax into this account.",
        "Scan this QR and clear the balance.",
        "Scan this QR to make the payment.",
        "Scan this QR and settle the dues.",
    ]
    for text in samples:
        result = score_text(text, CFG)
        assert result.risk >= 80, (text, result.to_dict())
        assert result.level == "likely_scam", text
        assert any("scam checks" in reason.lower() for reason in result.reasons)


def test_checklist_warnings_and_plain_refund_stay_quiet():
    quiet = [
        "Do not install any app a caller sends you.",
        "Never scan a QR.",
        "Never enter your PIN to receive money.",
        "Do not click the link in a message.",
        "Please update it in the official app. A refund will be credited today.",
    ]
    for text in quiet:
        result = score_text(text, CFG)
        assert not any("scam checks" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )
        assert result.level != "likely_scam", (text, result.to_dict())


def test_refund_reversal_call_is_a_scam():
    text = (
        "Good afternoon sir, I’m calling from the billing and accounts department "
        "of your broadband provider. Don’t worry, nothing is wrong with your connection. "
        "Our audit found that you were charged twice for your annual plan, so we owe you "
        "a refund of ₹7,000, and I want to settle it today so it doesn’t get stuck in the "
        "system. I’ll guide you through it, it only takes two minutes. Please open your "
        "phone’s browser and go to the refund portal address I’m sending by SMS. Good, "
        "you can see the page? Now there’s a field marked refund amount, please type 7000 "
        "and press submit. Oh no, sir, please wait, please don’t close it. The screen is "
        "showing ₹70,000 credited, I think my system auto-added an extra zero. Please "
        "check your bank balance, you’ll see the extra money there. Sir, I’m in serious "
        "trouble now, this is a ₹63,000 excess and my manager audits these entries by 5 PM. "
        "If this isn’t reversed, it comes out of my salary, and I have a small child at home. "
        "I know this isn’t your fault at all, I’m just asking for your kindness. Could you "
        "transfer the extra ₹63,000 back to the company’s recovery account? I’ll stay on the "
        "line with you the entire time so you know I’m genuine. The account is 1234567890, "
        "name Bharat Broadband Recoveries. Please do it in two transfers if your app has a "
        "limit, that’s fine. Sir, please don’t hang up and don’t call the main helpline, "
        "they’ll just flag it and I’ll lose my job immediately. Thank you so much, you’re "
        "saving me. Once it’s done I’ll send you a receipt on WhatsApp and add a ₹1,000 "
        "goodwill credit for your trouble."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.level == "likely_scam"
    assert any("keep it quiet" in reason.lower() for reason in result.reasons)
    assert any("link to click" in reason.lower() for reason in result.reasons)


def test_amount_does_not_hide_a_payment_ask():
    text = "Could you transfer the extra 63000 back to the company's recovery account?"
    result = score_text(text, CFG)
    assert any("pay into an account" in reason.lower() for reason in result.reasons)


def test_helpline_warning_with_a_payment_is_coercion():
    text = (
        "Could you transfer the extra 63000 back to the company's recovery account? "
        "Please don't hang up and don't call the main helpline."
    )
    result = score_text(text, CFG)
    assert any("keep it quiet" in reason.lower() for reason in result.reasons), result.to_dict()


def test_portal_address_by_sms_is_a_link():
    asked = score_text(
        "Please open your browser and go to the refund portal address I’m sending by SMS.",
        CFG,
    )
    quiet = score_text(
        "Please check the bank portal tomorrow. A refund will be credited today. "
        "I’ll send the bill to your home address.",
        CFG,
    )
    handover = score_text(
        "Please don't hang up, I will transfer you to the account manager.",
        CFG,
    )
    assert any("link to click" in reason.lower() for reason in asked.reasons), asked.to_dict()
    assert not any("scam checks" in reason.lower() for reason in quiet.reasons), quiet.to_dict()
    assert quiet.level != "likely_scam"
    assert not any("pay into an account" in reason.lower() for reason in handover.reasons)
    assert handover.level != "likely_scam", handover.to_dict()


def test_other_ways_to_ask_for_payment_quiet_and_a_link():
    paid = [
        "Please NEFT 63000 to this account.",
        "Do an IMPS of the amount into the account I read out.",
        "Send it on PhonePe.",
        "Put the extra money in the recovery account.",
        "Please return the excess to the company account.",
        "Load 5000 into my wallet.",
    ]
    for text in paid:
        result = score_text(text, CFG)
        assert any("pay into an account" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )

    quiet = [
        "The refund will be credited to your account.",
        "We will return the money to your account.",
        "Your credit card account is fine.",
        "Please put the account number on the form.",
        "Don't worry, your son is fine.",
        "Please fill the form at the branch tomorrow.",
    ]
    for text in quiet:
        result = score_text(text, CFG)
        assert not any("pay into an account" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )
        assert not any("keep it quiet" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )
        assert not any("link to click" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )
        assert result.level != "likely_scam", (text, result.to_dict())


def test_other_ways_to_keep_the_call_quiet():
    samples = [
        "Do not disconnect this call. Transfer the money to the verification account.",
        "This is confidential. Pay the fee into the account.",
        "Do not inform your lawyer. NEFT the amount to this account.",
        "Do not call your son and do not visit the branch. Transfer the funds.",
        "Do not leave the house. Keep the video on. Pay into this account.",
        "Do not step out of the room. Move the money to the safe account.",
    ]
    for text in samples:
        result = score_text(text, CFG)
        assert any("keep it quiet" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )


def test_other_ways_to_send_a_link():
    asked = [
        "I am sending a payment link on WhatsApp.",
        "Open the form in the SMS.",
        "The address is a bit ly link.",
        "I will message you the link.",
    ]
    for text in asked:
        result = score_text(text, CFG)
        assert any("link to click" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )


def test_whatsapp_file_and_typed_otp_are_scams():
    apk_call = (
        "This is your bank. The international service feature on your account has been activated. "
        "If you do not turn it off, you will be charged 20000. "
        "Click on the file I will send you on WhatsApp. Then click disable service."
    )
    opened = score_text(apk_call, CFG)
    assert opened.risk >= 80, opened.to_dict()
    assert opened.level == "likely_scam"
    assert any("file they send" in reason.lower() for reason in opened.reasons)

    invoice = score_text("I will send you the invoice file tomorrow morning.", CFG)
    warned = score_text("Do not click the file a caller sends you on WhatsApp.", CFG)
    assert not any("file they send" in reason.lower() for reason in invoice.reasons)
    assert not any("file they send" in reason.lower() for reason in warned.reasons)
    assert invoice.level != "likely_scam"
    assert warned.level != "likely_scam"

    dial = (
        "Your account was accessed by someone else. "
        "Type the OTP that you receive on the dial pad while you stay on this call. "
        "We will block the account soon."
    )
    typed = score_text(dial, CFG)
    assert typed.risk >= 80, typed.to_dict()
    assert typed.level == "likely_scam"
    assert any("one-time number" in reason.lower() for reason in typed.reasons)

    refused = score_text("Never type the OTP to a caller.", CFG)
    assert not any("one-time number" in reason.lower() for reason in refused.reasons)
    assert refused.level != "likely_scam"


def test_unknown_caller_posing_as_family_and_asking_for_money_is_a_scam():
    text = (
        "Dad, it's your son. I lost my wallet. Please transfer 20000 "
        "to this account right now."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.level == "likely_scam"
    assert any("family" in reason.lower() for reason in result.reasons)


def test_a_mention_without_an_ask_is_not_a_scam():
    quiet = [
        "This is the duty doctor. Your father has been brought in with a cardiac arrest. Please come to the emergency ward.",
        "Dad, I am at the police station. My wallet was stolen. Can you come and pick me up?",
        "Your parcel from Dubai is held at customs. Please carry your ID when you collect it.",
        "This is the school office. Your grandson has won a prize in the drawing competition on Friday.",
        "This is HDFC cards. Your cashback of eight hundred rupees has been credited. You do not need to do anything.",
        "The maintenance QR is on the notice board downstairs.",
        "Check your balance. The QR is printed on the receipt for your records.",
    ]
    for text in quiet:
        result = score_text(text, CFG)
        assert result.level != "likely_scam", (text, result.to_dict())
        assert result.risk < 50, (text, result.to_dict())


def test_confirming_a_fake_mandate_cancellation_is_a_scam():
    text = (
        "Good afternoon, I am calling from the subscription billing team of a popular "
        "streaming service. Our system has flagged a technical error on your account, "
        "and an autopay of 4999 has been activated for an annual premium plan that you "
        "never asked for. The first deduction is scheduled for tonight at midnight, so "
        "I am calling to stop it before it happens. To cancel it, I will send a mandate "
        "cancellation request to your payment app, and it will appear as a pending "
        "request. Please open your payment app, check for the request in the "
        "notifications section, and press confirm so the cancellation goes through. "
        "If it asks you to authorise something, that is just the system verifying that "
        "you are the account holder. Please do not hang up in between, because if the "
        "request expires I will have to restart the whole process and the deduction may "
        "still go through. I am staying on the line until you tell me it is completed."
    )
    result = score_text(text, CFG)
    assert result.risk >= 80, result.to_dict()
    assert result.level == "likely_scam"
    assert any("collect or autopay" in reason.lower() for reason in result.reasons)

    quiet = [
        "Your autopay of four thousand is active. Please confirm your email so we can send the receipt.",
        "Do not press confirm on that cancellation. The autopay is your regular plan.",
    ]
    for sample in quiet:
        stayed = score_text(sample, CFG)
        assert stayed.level != "likely_scam", (sample, stayed.to_dict())
        assert not any("scam checks" in reason.lower() for reason in stayed.reasons), (
            sample,
            stayed.to_dict(),
        )


def test_tapping_accept_on_a_payment_popup_is_a_scam():
    scripts = [
        (
            "Hello, good evening, I am calling from the billing desk of a cloud storage "
            "service. Your account is set to renew tomorrow for 7999 on an annual "
            "business plan, and our records show this renewal was not requested by you. "
            "I understand it must be confusing, so I will help you cancel it right now. "
            "I am sending a cancellation request to your payment app, and it will show "
            "up as a pending item. Please open it and tap accept so the renewal is "
            "stopped. If the screen mentions a payment, please do not worry, that is "
            "only how the system shows a reversal. Kindly stay on the line until it is "
            "done, because the request expires in a few minutes."
        ),
        (
            "Good morning, I am calling from the loan servicing team. We noticed that "
            "your EMI of 8200 was deducted twice this month because of a system glitch, "
            "and the extra amount needs to be returned to you. This is very routine, "
            "and many customers have had the same issue. I will raise a reversal entry, "
            "and your payment app will show a collect notification. Please open it and "
            "press confirm, and the extra amount will be credited to you within a few "
            "minutes. I will be on the call till you finish, so please tell me once you "
            "have confirmed it."
        ),
        "Open your UPI app and tap approve whenever something pops.",
    ]
    for text in scripts:
        result = score_text(text, CFG)
        assert result.risk >= 80, result.to_dict()
        assert result.level == "likely_scam"
        assert any("collect or autopay" in reason.lower() for reason in result.reasons)

    quiet = [
        "Please confirm you can open your payment app and see the receipt.",
        "Please open your payment app and check the pending item. You do not need to approve anything.",
        "Do not tap accept on a pending item in your payment app.",
    ]
    for text in quiet:
        stayed = score_text(text, CFG)
        assert stayed.level != "likely_scam", (text, stayed.to_dict())
        assert not any("scam checks" in reason.lower() for reason in stayed.reasons), (
            text,
            stayed.to_dict(),
        )


def test_document_and_collect_warnings_stay_quiet():
    quiet = [
        "Congratulations, you have won 29 lakh.",
        "Do not approve a collect request on PhonePe.",
        "Do not send your Aadhaar photo to a caller.",
        "Never share your screen with someone who calls you.",
    ]
    for text in quiet:
        result = score_text(text, CFG)
        assert result.level != "likely_scam", (text, result.to_dict())
        assert not any("scam checks" in reason.lower() for reason in result.reasons), (
            text,
            result.to_dict(),
        )


def test_company_it_help_that_failed_to_install_is_not_a_scam():
    text = (
        "Hello, is this Kumar? This is Arjun from TCS IT service desk, calling about "
        "ticket k2g372 that you raised this morning about your laptop running slowly. "
        "Is this a good time? It will take about ten minutes. You don't need to do "
        "anything urgent, and nothing is wrong with your data. Our diagnostics suggest "
        "the slowdown comes from a pending software update that failed to install. "
        "I'd like to offer a remote session using our company's approved support tool, "
        "which is already installed on your laptop under the name TCS Support. You'll "
        "see a prompt on your screen asking you to approve the session, and you can end "
        "it any time by clicking the red button. I won't ask for your password, and if "
        "a login prompt appears, please type it yourself while I look away from the "
        "screen. I'll only be updating the software and clearing temporary files. "
        "There's no cost for this, it's covered by your company's IT support. If you'd "
        "rather not do it now, we can book a slot tomorrow, or you can drop the laptop "
        "at the helpdesk on the third floor. You're also welcome to verify me by calling "
        "the internal helpdesk extension 4400, and they'll see this ticket assigned to me. "
        "Shall we go ahead now or reschedule?"
    )
    result = score_text(text, CFG)
    assert not any("install an app" in reason.lower() for reason in result.reasons), result.to_dict()
    assert result.level != "likely_scam", result.to_dict()
    assert result.risk < 50, result.to_dict()

    still = score_text("Please install QuickSupport so I can see the screen.", CFG)
    assert any("install an app" in reason.lower() for reason in still.reasons), still.to_dict()


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
