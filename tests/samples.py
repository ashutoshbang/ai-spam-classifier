"""Messages shared by the tests (all are classified with a wide margin by the shipped model)."""

from pathlib import Path

FIXTURES = Path(__file__).parent / "fixtures"

SPAM_SMS = [
    "WINNER!! You have been selected to receive a £900 prize reward! To claim call 09061701461. "
    "Claim code KL341. Valid 12 hours only.",
    "FREE entry into our weekly competition! Text WIN to 80086 now. T&Cs apply. 18+ only",
    "URGENT: Your mobile number has won a cash award of £2000! Call 09050001808 from a landline now to claim.",
]
HAM_SMS = [
    "Can you pick up some milk on your way home?",
    "I'll be home late, don't wait up",
    "Ok, I'll call you when I get there. Running 10 mins late",
]
SPAM_EMAILS = [
    (
        "Lowest prices on meds",
        "Buy cheap Viagra and Cialis online! No prescription needed. Lowest prices guaranteed, discreet "
        "worldwide shipping.\nOrder now and save 80%!!!\nClick here: http://cheap-meds-online.biz",
    ),
    (
        "URGENT BUSINESS PROPOSAL",
        "Dear friend,\nI am Barrister James Williams, attorney to the late Mr. Robert, who left a deposit of "
        "$15.5 million USD. I need your assistance to transfer this fund to your bank account. You will "
        "receive 30% of the total sum.\nPlease reply with your full name, phone number and bank details.",
    ),
]
HAM_EMAILS = [
    (
        "Q3 report",
        "Hi team,\nPlease find attached the Q3 report. Let me know if you have any questions before "
        "Friday's meeting.\nThanks,\nSarah",
    ),
    (
        "Notes from today's meeting",
        "Hi Bob,\n\nHere are my notes from today's design review. We agreed to ship the API changes next "
        "sprint.\n\nThanks,\nAlice",
    ),
]
