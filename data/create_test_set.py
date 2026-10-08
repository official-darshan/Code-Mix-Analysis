from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "unseen_test_set.csv"

CASES = [
    # Payment
    ("I paid for the cart but the confirmation screen never appeared.", "English", "Payment Issue", "Negative", "Unresolved", "No"),
    ("My bank account was charged, but the order still says unpaid.", "English", "Payment Issue", "Negative", "Unresolved", "No"),
    ("The transaction went through and everything is confirmed now, thanks.", "English", "Payment Issue", "Positive", "Resolved", "No"),
    ("Mera payment successful hua tha lekin order place nahi hua.", "Hinglish", "Payment Issue", "Negative", "Unresolved", "No"),
    ("Payment methods kaun kaun se available hain?", "Hinglish", "General Query", "Neutral", "Unresolved", "No"),

    # Refund
    ("I have been waiting for the reversed amount since last week.", "English", "Refund Request", "Negative", "Unresolved", "No"),
    ("Can you process the money back for my cancelled purchase?", "English", "Refund Request", "Neutral", "Unresolved", "No"),
    ("Thanks, the reversed amount has finally reached my bank.", "English", "Refund Request", "Positive", "Resolved", "No"),
    ("Cancelled order ka refund abhi tak account mein nahi aaya.", "Hinglish", "Refund Request", "Negative", "Unresolved", "No"),
    ("Refund policy kya hai for a returned item?", "Hinglish", "General Query", "Neutral", "Unresolved", "No"),
    ("I have chased the refund three times and nobody has fixed it, please escalate this case.", "English", "Refund Request", "Negative", "Unresolved", "Yes"),

    # Cancellation
    ("Please stop this purchase before it gets shipped.", "English", "Order Cancellation", "Neutral", "Unresolved", "No"),
    ("The cancellation went through successfully, thank you.", "English", "Order Cancellation", "Positive", "Resolved", "No"),
    ("Mera order dispatch hone se pehle cancel karna hai.", "Hindi", "Order Cancellation", "Neutral", "Unresolved", "No"),
    ("I asked to cancel twice but the order is still active. Escalate this.", "English", "Order Cancellation", "Negative", "Unresolved", "Yes"),

    # Tracking / delay
    ("Can you show me the current location of the parcel?", "English", "Order Tracking", "Neutral", "Unresolved", "No"),
    ("Tracking link send kar do, mujhe parcel ka status dekhna hai.", "Hinglish", "Order Tracking", "Neutral", "Unresolved", "No"),
    ("I can see the shipment status now, thanks.", "English", "Order Tracking", "Positive", "Resolved", "No"),
    ("Yesterday was the promised delivery date and nothing arrived.", "English", "Delivery Delay", "Negative", "Unresolved", "No"),
    ("The delayed package finally arrived today, thanks.", "English", "Delivery Delay", "Positive", "Resolved", "No"),
    ("What is the usual delivery window for Mumbai orders?", "English", "General Query", "Neutral", "Unresolved", "No"),
    ("Meri delivery expected date se do din late hai.", "Hindi", "Delivery Delay", "Negative", "Unresolved", "No"),

    # Login
    ("The password is correct, but the account still rejects my login.", "English", "Login Problem", "Negative", "Unresolved", "No"),
    ("Password reset ka OTP hi nahi aa raha.", "Hinglish", "Login Problem", "Negative", "Unresolved", "No"),
    ("I can access the account again. Thank you.", "English", "Login Problem", "Positive", "Resolved", "No"),
    ("I have been locked out for days and already contacted support several times. Please escalate.", "English", "Login Problem", "Negative", "Unresolved", "Yes"),

    # Technical
    ("The checkout screen freezes every time I press continue.", "English", "Technical Problem", "Negative", "Unresolved", "No"),
    ("App baar baar crash ho rahi hai jab payment page open karta hoon.", "Hinglish", "Technical Problem", "Negative", "Unresolved", "No"),
    ("The app is working again after the update, thanks.", "English", "Technical Problem", "Positive", "Resolved", "No"),
    ("The website is completely fine now.", "English", "Technical Problem", "Positive", "Resolved", "No"),

    # Product complaint / feedback
    ("The box contained a different item from what I ordered.", "English", "Product Complaint", "Negative", "Unresolved", "No"),
    ("Product khula toh screen cracked mili.", "Hinglish", "Product Complaint", "Negative", "Unresolved", "No"),
    ("The replacement arrived and solved the product problem.", "English", "Product Complaint", "Positive", "Resolved", "No"),
    ("The item is exactly as described and works perfectly.", "English", "Product Feedback", "Positive", "Resolved", "No"),
    ("Mujhe product bahut pasand aaya aur quality excellent hai.", "Hindi", "Product Feedback", "Positive", "Resolved", "No"),
    ("The product arrived in excellent condition and I am very happy with it.", "English", "Product Feedback", "Positive", "Resolved", "No"),

    # General query
    ("Where can I see the different shipping options before checkout?", "English", "General Query", "Neutral", "Unresolved", "No"),
    ("Mumbai mein delivery available hai kya?", "Hindi", "General Query", "Neutral", "Unresolved", "No"),
    ("Can you explain how your return process works?", "English", "General Query", "Neutral", "Unresolved", "No"),
    ("I just wanted some information about the service.", "English", "General Query", "Neutral", "Unresolved", "No"),
    ("Your support team answered my question, thank you.", "English", "General Query", "Positive", "Resolved", "No"),
]


def make_agent_response(resolution: str, escalation: str) -> str:
    if escalation == "Yes":
        return "We have escalated the case to the next support level for urgent review."
    if resolution == "Resolved":
        return "The request has been resolved successfully. Thank you for contacting support."
    return "We are still reviewing the request and will provide an update shortly."


def main():
    columns = [
        "customer_message", "language", "intent", "sentiment",
        "resolution", "escalation"
    ]
    df = pd.DataFrame(CASES, columns=columns)
    df["agent_response"] = [
        make_agent_response(r[4], r[5]) for r in CASES
    ]
    df["priority"] = df.apply(
        lambda row: "High" if row["escalation"] == "Yes"
        else "Medium" if row["sentiment"] == "Negative" or row["resolution"] == "Unresolved"
        else "Low",
        axis=1,
    )
    df.to_csv(OUTPUT, index=False)
    print(f"Unseen test set created: {OUTPUT}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
