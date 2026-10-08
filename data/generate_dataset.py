from pathlib import Path
import random
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "support_conversations.csv"
TARGET_ROWS = 50000
SEED = 42
random.seed(SEED)

# -----------------------------------------------------------------------------
# The dataset is intentionally scenario-driven.
# Each message is built from an intent-specific problem + a logically
# consistent outcome/sentiment suffix. This avoids the contradictory labels
# that caused the earlier model to learn wrong relationships.
# -----------------------------------------------------------------------------

INTENTS = {
    "Payment Issue": {
        "English": [
            "My payment failed while placing the order",
            "The payment was deducted but my order was not confirmed",
            "My payment status is showing incomplete",
            "The transaction was declined during checkout",
            "My card was charged but I did not get an order confirmation",
            "The payment is stuck as pending",
            "I cannot complete the payment for my order",
            "The payment went through but the order is still pending",
            "The transaction keeps failing when I try to pay",
            "The amount was deducted but the order was not created",
            "I am having a problem with payment at checkout",
            "My payment has not been updated on the order"
        ],
        "Hindi": [
            "Mera payment fail ho gaya",
            "Mera payment deduct ho gaya lekin order confirm nahi hua",
            "Payment ka status incomplete dikh raha hai",
            "Checkout par transaction decline ho gaya",
            "Card se payment hua lekin order confirmation nahi aaya",
            "Payment pending dikh raha hai",
            "Mera payment complete nahi ho raha",
            "Payment ho gaya lekin order abhi pending hai",
            "Payment karte waqt baar baar fail ho raha hai",
            "Paisa deduct hua lekin order create nahi hua",
            "Checkout par payment ki problem aa rahi hai",
            "Payment ka status update nahi ho raha"
        ],
        "Hinglish": [
            "Mera payment deduct ho gaya but order confirm nahi hua",
            "Payment successful hai but order is still pending",
            "Maine payment complete kiya but status incomplete hai",
            "My card se payment fail ho gaya",
            "Payment ho gaya but confirmation nahi aaya",
            "Payment pending show ho raha hai",
            "I cannot complete mera payment",
            "Amount deduct hua but order create nahi hua",
            "Checkout par payment issue aa raha hai",
            "Mera transaction fail ho gaya while paying",
            "Payment ka status update nahi ho raha",
            "Order confirm nahi hua after payment"
        ],
        "contexts": [
            "during checkout", "while placing the order", "in the app",
            "on the website", "for my latest order", "right now",
            "after I submitted the payment", "on the payment screen",
            "when I tried to pay", "for this transaction"
        ]
    },
    "Refund Request": {
        "English": [
            "My refund has not arrived yet",
            "I want a refund for my order",
            "When will I receive my refund",
            "The refund is still pending",
            "My money has not been refunded",
            "Please process my refund",
            "I have not received the refund amount",
            "My refund is delayed",
            "I need a refund for this purchase",
            "The refund status has not changed",
            "Can you check my refund status",
            "The refund amount is still missing"
        ],
        "Hindi": [
            "Mera refund abhi tak nahi aaya",
            "Mujhe order ka refund chahiye",
            "Refund mujhe kab milega",
            "Mera refund pending hai",
            "Mera paisa refund nahi hua",
            "Mera refund process kar do",
            "Refund amount mujhe nahi mila",
            "Mera refund late hai",
            "Is purchase ka refund chahiye",
            "Refund ka status update nahi hua",
            "Mera refund status check karo",
            "Refund amount abhi bhi nahi mila"
        ],
        "Hinglish": [
            "Mera refund abhi tak nahi aaya",
            "I want mera refund for this order",
            "Refund kab milega for my order",
            "Refund abhi bhi pending hai",
            "Mera money refund nahi hua",
            "Please mera refund process karo",
            "Refund amount receive nahi hua",
            "Mera refund delayed hai",
            "Mujhe is purchase ka refund chahiye",
            "Refund ka status change nahi hua",
            "Can you check mera refund status",
            "Refund amount abhi bhi missing hai"
        ],
        "contexts": [
            "for my recent order", "after the cancellation", "for the returned item",
            "on my last purchase", "for this transaction", "in the app",
            "on the website", "after the payment reversal", "for order support",
            "right now"
        ]
    },
    "Order Cancellation": {
        "English": [
            "I want to cancel my order",
            "Please cancel my order",
            "Can I cancel the order now",
            "I need to cancel this purchase",
            "How can I cancel my order",
            "I no longer want this order",
            "Please help me cancel the order",
            "Is order cancellation possible",
            "I want to stop this order",
            "Cancel my order before dispatch",
            "I need to withdraw my order",
            "Please stop the shipment for my order"
        ],
        "Hindi": [
            "Mera order cancel karna hai",
            "Please mera order cancel karo",
            "Kya main order abhi cancel kar sakta hoon",
            "Mujhe ye purchase cancel karni hai",
            "Order cancel kaise karu",
            "Mujhe ye order ab nahi chahiye",
            "Mera order cancel karne mein help karo",
            "Order cancellation possible hai kya",
            "Mera order rokna hai",
            "Dispatch se pehle order cancel karna hai",
            "Mujhe order withdraw karna hai",
            "Order ki shipment stop karni hai"
        ],
        "Hinglish": [
            "Mera order cancel karna hai",
            "Please mera order cancel karo",
            "Can I order cancel kar sakta hoon",
            "I need to cancel mera purchase",
            "Order cancel kaise karu",
            "Mujhe ye order nahi chahiye anymore",
            "Please help me mera order cancel karne mein",
            "Order cancellation possible hai kya",
            "Mera order stop karna hai",
            "Dispatch se pehle order cancel karna hai",
            "I want to cancel ye order",
            "Shipment stop karni hai"
        ],
        "contexts": [
            "before dispatch", "for my latest purchase", "from the app",
            "on the website", "right now", "before it is shipped",
            "for this order", "as soon as possible", "today", "for this purchase"
        ]
    },
    "Order Tracking": {
        "English": [
            "Where is my order",
            "Can you tell me the status of my order",
            "I need the tracking link",
            "Please provide my order tracking update",
            "Where can I track my package",
            "My order tracking is not updating",
            "I want to know where my package is",
            "Can you check my order status",
            "What is the current order status",
            "Please give me the tracking details",
            "How can I track my order",
            "Can you share the current location of my parcel"
        ],
        "Hindi": [
            "Mera order kaha hai",
            "Mujhe order ka status batao",
            "Mujhe tracking link chahiye",
            "Order ka tracking update do",
            "Mera parcel kaha track kar sakta hoon",
            "Order tracking update nahi ho raha",
            "Mera package kaha hai",
            "Order ka status check karo",
            "Current order status kya hai",
            "Tracking details chahiye",
            "Mera order track kaise karu",
            "Mere parcel ki current location batao"
        ],
        "Hinglish": [
            "Mera order kaha hai can you check",
            "Order ka status bata do please",
            "Mujhe order tracking link chahiye",
            "Can you give me mera tracking update",
            "Mera parcel kaha hai right now",
            "Order tracking update nahi aa raha",
            "Can you check mera order status",
            "Mujhe current tracking details chahiye",
            "I need mera order tracking",
            "Mera order track kaise karu",
            "Please check mera parcel status",
            "Mere package ki location batao"
        ],
        "contexts": [
            "for my recent order", "from the app", "on the website",
            "right now", "for this package", "today", "for my shipment",
            "after the order was placed", "for my parcel", "at the moment"
        ]
    },
    "Delivery Delay": {
        "English": [
            "My delivery is delayed",
            "Why is my package late",
            "My order has not arrived on time",
            "The delivery is taking too long",
            "My package is overdue",
            "The promised delivery date has passed",
            "My parcel is late",
            "Please help with the delayed delivery",
            "The courier has not delivered my order",
            "My delivery is still pending",
            "My package has not arrived yet",
            "The order is past the expected delivery date"
        ],
        "Hindi": [
            "Mera order late ho gaya hai",
            "Mera package late hai",
            "Mera order time par nahi aaya",
            "Delivery bahut late ho rahi hai",
            "Mera package overdue hai",
            "Promised delivery date nikal gayi hai",
            "Mera parcel late hai",
            "Delayed delivery mein help chahiye",
            "Courier ne mera order deliver nahi kiya",
            "Meri delivery abhi bhi pending hai",
            "Mera package abhi tak nahi aaya",
            "Order expected date ke baad bhi nahi aaya"
        ],
        "Hinglish": [
            "Mera order late ho gaya hai",
            "Why is mera package late",
            "My order time par nahi aaya",
            "Delivery bahut late ho rahi hai",
            "Mera package overdue hai",
            "Expected delivery date nikal gayi hai",
            "Mera parcel abhi bhi late hai",
            "Please delayed delivery mein help karo",
            "Courier ne order deliver nahi kiya yet",
            "My delivery abhi pending hai",
            "Mera package abhi tak nahi aaya",
            "Order expected date ke baad nahi mila"
        ],
        "contexts": [
            "for my latest order", "for my package", "after waiting several days",
            "from the last shipment", "right now", "for this delivery",
            "after the promised date", "from the courier", "today", "for my parcel"
        ]
    },
    "Login Problem": {
        "English": [
            "I cannot log in to my account",
            "I forgot my password",
            "My login is not working",
            "I cannot access my account",
            "The password reset email is not arriving",
            "I am unable to sign in",
            "My account login is failing",
            "I have a problem logging in",
            "Login keeps showing an error",
            "I need help with my password",
            "My sign in is not working",
            "I am locked out of my account"
        ],
        "Hindi": [
            "Mera login nahi ho raha",
            "Main password bhool gaya hoon",
            "Account login nahi ho raha",
            "Mujhe account access nahi mil raha",
            "Password reset email nahi aa raha",
            "Main sign in nahi kar pa raha",
            "Account mein login fail ho raha",
            "Login mein problem hai",
            "Login par error aa raha hai",
            "Mera password kaam nahi kar raha",
            "Sign in nahi ho raha",
            "Main account se lock ho gaya hoon"
        ],
        "Hinglish": [
            "Mera login nahi ho raha on the app",
            "I forgot mera password",
            "Account login nahi ho raha",
            "I cannot access mera account",
            "Password reset email nahi aa raha",
            "Sign in nahi ho raha",
            "My account login fail ho raha hai",
            "Login mein problem aa rahi hai",
            "Login error aa raha hai",
            "Mera password is not working",
            "I cannot sign in mera account",
            "Account lock ho gaya hai"
        ],
        "contexts": [
            "on the app", "on the website", "right now", "from my phone",
            "after the password change", "when I try to sign in", "today",
            "from my account", "on the login page", "while accessing the app"
        ]
    },
    "Technical Problem": {
        "English": [
            "The app is crashing when I open it",
            "I am getting an error on the website",
            "The checkout page is not working",
            "The application keeps freezing",
            "The website is showing an error",
            "The app is not opening",
            "I cannot complete checkout because of an error",
            "The website is very slow",
            "The screen gets stuck during checkout",
            "There is a technical problem",
            "The app keeps closing unexpectedly",
            "The page is not responding"
        ],
        "Hindi": [
            "App open nahi ho rahi",
            "Website par error aa raha hai",
            "Checkout page kaam nahi kar raha",
            "App baar baar crash ho rahi hai",
            "Website error dikha rahi hai",
            "App open nahi ho rahi hai",
            "Error ki wajah se checkout complete nahi ho raha",
            "Website bahut slow hai",
            "Checkout par screen stuck ho rahi hai",
            "Technical problem aa rahi hai",
            "App achanak close ho ja rahi hai",
            "Page respond nahi kar raha"
        ],
        "Hinglish": [
            "App open nahi ho rahi on my phone",
            "Website par error aa raha hai",
            "Checkout page hang ho raha hai",
            "The app baar baar crash ho rahi hai",
            "Website error show kar rahi hai",
            "App open nahi ho rahi properly",
            "Checkout complete nahi ho raha because of an error",
            "Website bahut slow ho gayi hai",
            "Screen checkout par stuck ho rahi hai",
            "Technical issue aa raha hai",
            "App suddenly close ho rahi hai",
            "Page respond nahi kar raha"
        ],
        "contexts": [
            "while placing an order", "during checkout", "after opening the app",
            "on the website", "from my phone", "right now", "today",
            "when I try to pay", "on the checkout screen", "after the latest update"
        ]
    },
    "Product Complaint": {
        "English": [
            "The product I received is damaged",
            "I received the wrong product",
            "The product quality is poor",
            "The item was broken when I received it",
            "My product has a defect",
            "The product is not as described",
            "I received a damaged item",
            "The item is missing a part",
            "The product is defective",
            "I am unhappy with the product quality",
            "The item I received is different from what I ordered",
            "The product arrived with a defect"
        ],
        "Hindi": [
            "Mujhe damaged product mila",
            "Mujhe wrong product mila",
            "Product ki quality poor hai",
            "Item broken tha jab mujhe mila",
            "Product mein defect hai",
            "Product description jaisa nahi hai",
            "Mujhe damaged item mila",
            "Item ka ek part missing hai",
            "Mujhe defective product mila",
            "Product ki quality achhi nahi hai",
            "Mujhe order se alag item mila",
            "Product mein defect tha"
        ],
        "Hinglish": [
            "Mujhe damaged product mila",
            "I received wrong product",
            "Product ki quality poor hai",
            "Item broken tha when I received it",
            "Mera product defective hai",
            "Product as described nahi hai",
            "I received damaged item",
            "Product ka part missing hai",
            "Mujhe defective product mila",
            "Product quality bilkul achhi nahi hai",
            "Mera item order se different hai",
            "Product mein defect hai"
        ],
        "contexts": [
            "after opening the package", "from my latest order", "when it arrived",
            "for this purchase", "after delivery", "from the courier",
            "in the package", "for my recent order", "today", "after unboxing"
        ]
    },
    "Product Feedback": {
        "English": [
            "The product works perfectly and I am happy with it",
            "The item is exactly as described and looks great",
            "I am very satisfied with the product quality",
            "The product arrived in excellent condition",
            "I really like the product I received",
            "The item quality is better than expected",
            "Everything about the product is good",
            "The product is working perfectly",
            "I am happy with my purchase",
            "The item matched the description perfectly",
            "The product is excellent",
            "I had a good experience with the product"
        ],
        "Hindi": [
            "Product bahut achha hai aur mujhe pasand aaya",
            "Item bilkul description jaisa hai",
            "Mujhe product ki quality bahut achhi lagi",
            "Product excellent condition mein aaya",
            "Mujhe product bahut pasand aaya",
            "Item ki quality expected se better hai",
            "Product mein sab kuch achha hai",
            "Product perfectly kaam kar raha hai",
            "Main apne purchase se khush hoon",
            "Item description ke bilkul according hai",
            "Product excellent hai",
            "Mera product ka experience achha raha"
        ],
        "Hinglish": [
            "The product works perfectly and mujhe bahut pasand aaya",
            "Item exactly as described hai and looks great",
            "I am very happy with product quality",
            "Product excellent condition mein aaya",
            "Mujhe product really pasand aaya",
            "Item quality expected se better hai",
            "Product mein everything achha hai",
            "Mera product perfectly work kar raha hai",
            "I am happy with mera purchase",
            "Item description ke according hai",
            "Product really excellent hai",
            "Mera product experience bahut good raha"
        ],
        "contexts": [
            "from my latest order", "after delivery", "for this purchase",
            "after opening the package", "for my recent order", "today",
            "after using it", "from the new order", "for this item", "after receiving it"
        ]
    },
    "General Query": {
        "English": [
            "Can you tell me how long standard delivery usually takes",
            "What are your delivery charges",
            "What payment methods do you accept",
            "What is your return policy",
            "How can I contact customer support",
            "Do you provide delivery to Mumbai",
            "How can I change my account details",
            "What are your service hours",
            "Can you explain the refund process",
            "Where can I find your support information",
            "What are the available delivery options",
            "Can you tell me more about your service"
        ],
        "Hindi": [
            "Standard delivery mein kitna time lagta hai",
            "Delivery charges kitne hain",
            "Aap kaunse payment methods accept karte ho",
            "Aapki return policy kya hai",
            "Customer support se kaise contact karu",
            "Kya aap Mumbai mein delivery karte ho",
            "Account details kaise change karu",
            "Aapki service timings kya hain",
            "Refund process kya hai",
            "Support information kaha milegi",
            "Delivery options kaunse available hain",
            "Aapki service ke baare mein batao"
        ],
        "Hinglish": [
            "Standard delivery mein kitna time lagta hai",
            "Delivery charges kitne hain",
            "Which payment methods aap accept karte ho",
            "Aapki return policy kya hai",
            "Customer support se kaise contact karu",
            "Do you deliver Mumbai mein",
            "Account details kaise change karu",
            "Service timings kya hain",
            "Refund process kya hai",
            "Support information kaha milegi",
            "Delivery options kaunse hain",
            "Can you explain aapki service"
        ],
        "contexts": [
            "before placing an order", "for a new customer", "for this service",
            "from the website", "before purchase", "for my account",
            "right now", "for Mumbai", "for a future order", "in general"
        ]
    }
}

PREFIXES = {
    "English": ["", "Please", "Hi,", "Hello,"] ,
    "Hindi": ["", "Please", "Hi,", "Hello,"] ,
    "Hinglish": ["", "Please", "Hi,", "Hello,"]
}

OUTCOMES = {
    "resolved_positive": {
        "sentiment": "Positive", "resolution": "Resolved", "escalation": "No",
        "suffix": {
            "Payment Issue": ["Thanks, the payment issue is fixed and my order is confirmed.", "Thank you, the payment went through successfully now."],
            "Refund Request": ["Thanks, I received the refund successfully.", "Thank you, the refund has reached my account now."],
            "Order Cancellation": ["Thanks, the order was cancelled successfully.", "Thank you, the cancellation is complete now."],
            "Order Tracking": ["Thanks, I found the tracking details and can follow the order now.", "Thank you, I can see the order status now."],
            "Delivery Delay": ["Thanks, the delayed order has now been delivered.", "Thank you, the delivery is complete now."],
            "Login Problem": ["Thanks, I can log in successfully now.", "Thank you, my account access is working again."],
            "Technical Problem": ["Thanks, the app is working normally now.", "Thank you, the technical issue is fixed now."],
            "Product Complaint": ["Thanks, the replacement fixed the product issue.", "Thank you, the product issue is resolved now."],
            "Product Feedback": ["Thanks, I am very happy with the product.", "Thank you, the product experience has been great."],
            "General Query": ["Thanks, that answered my question.", "Thank you, I understand the process now."]
        }
    },
    "resolved_negative": {
        "sentiment": "Negative", "resolution": "Resolved", "escalation": "No",
        "suffix": {
            intent: ["The issue is fixed now, but the experience was frustrating.", "It is resolved, although the support experience was disappointing."]
            for intent in INTENTS
        }
    },
    "unresolved_neutral": {
        "sentiment": "Neutral", "resolution": "Unresolved", "escalation": "No",
        "suffix": {
            intent: ["It is still pending. Please provide an update.", "I am still waiting. Can you check the status for me?"]
            for intent in INTENTS
        }
    },
    "unresolved_negative": {
        "sentiment": "Negative", "resolution": "Unresolved", "escalation": "No",
        "suffix": {
            intent: ["It is still not resolved and this is very frustrating.", "I am still waiting and I am disappointed with the experience.", "The problem continues and I need help as soon as possible."]
            for intent in INTENTS
        }
    },
    "escalation": {
        "sentiment": "Negative", "resolution": "Unresolved", "escalation": "Yes",
        "suffix": {
            intent: ["I already contacted support several times and nobody resolved it. Please escalate this complaint.",
                     "I have followed up multiple times and need a supervisor to review this urgently.",
                     "This is still unresolved after repeated contact. Please escalate the case immediately."]
            for intent in INTENTS
        }
    }
}

# A few state-aware phrases make the language less repetitive without changing labels.

OUTCOME_DETAILS = {
    "resolved_positive": [
        "Everything is fine now.",
        "I appreciate the quick help.",
        "Thanks for resolving it.",
        "This works for me now.",
        "I am satisfied with the resolution.",
        "The process is clear now.",
        "I am happy with the result.",
        "Thank you for sorting this out.",
        "That solved my concern.",
        "I can use the service normally now.",
        "I am pleased with the support.",
        "The result is exactly what I needed.",
        "Thanks for the assistance.",
        "The request is complete now.",
        "I appreciate the resolution.",
        "This is working properly now.",
        "I am satisfied with the outcome.",
        "The issue is no longer a problem.",
        "Everything looks good now.",
        "Thank you for the successful resolution.",
    ],
    "resolved_negative": [
        "I expected a smoother experience.",
        "The issue took too long to fix.",
        "I am not happy with the support experience.",
        "It should have been handled faster.",
        "The problem is fixed, but the delay was frustrating.",
        "The resolution is complete now.",
    ],
    "unresolved_neutral": [
        "Please check the current status.",
        "I would like an update.",
        "Can you look into this?",
        "Please tell me what happens next.",
        "I need the latest update.",
        "Please advise on the next step.",
    ],
    "unresolved_negative": [
        "Please help me as soon as possible.",
        "I am disappointed with the delay.",
        "This is becoming frustrating.",
        "I expected this to be fixed sooner.",
        "I need this resolved quickly.",
        "Please take this issue seriously.",
    ],
    "escalation": [
        "Please treat this as urgent.",
        "I need a senior team member to review it.",
        "Please raise this to the next support level.",
        "I need this case handled urgently.",
        "Please make sure the issue is formally reviewed.",
        "I would like a supervisor to take ownership of this case.",
    ],
}

ALLOWED_OUTCOMES = {intent: list(OUTCOMES.keys()) for intent in INTENTS}
ALLOWED_OUTCOMES["Product Feedback"] = ["resolved_positive"]
ALLOWED_OUTCOMES["General Query"] = ["resolved_positive", "unresolved_neutral"]


def make_agent_response(intent: str, outcome_key: str) -> str:
    if outcome_key == "resolved_positive":
        return f"Your {intent.lower()} request has been resolved. Thank you for contacting support."
    if outcome_key == "resolved_negative":
        return f"The {intent.lower()} issue has been resolved. We apologize for the inconvenience."
    if outcome_key == "unresolved_neutral":
        return "We are still reviewing the request and will provide an update shortly."
    if outcome_key == "unresolved_negative":
        return "We are continuing to investigate the issue and will work to resolve it as soon as possible."
    return "The case has been escalated to the next support level for urgent review."


TIME_PHRASES = [
    "today", "for several days", "for a while", "since yesterday",
    "for the last few days", "recently", "at the moment", "right now"
]

CHAT_REPLACEMENTS = [
    ("please", "pls"),
    ("because", "coz"),
    ("cannot", "cant"),
    ("already", "alrdy"),
]


def add_surface_variation(text: str, index: int) -> str:
    text = text.strip()
    if index % 17 == 0:
        text = text.replace("please", "pls").replace("Please", "Pls")
    if index % 23 == 0:
        text = text.replace("cannot", "cant").replace("Cannot", "Cant")
    if index % 31 == 0:
        text += "!"
    if index % 47 == 0:
        text = text.replace(" and ", " & ", 1)
    return text


def choose_language(i: int) -> str:
    # 40% English, 20% Hindi, 40% Hinglish.
    r = i % 10
    if r < 4:
        return "English"
    if r < 6:
        return "Hindi"
    return "Hinglish"


def build_message(intent: str, language: str, base: str, context: str, prefix: str, outcome_key: str, i: int) -> str:
    message = base

    # Add the context only when it does not make the sentence awkward.
    if context and i % 3 != 0:
        message = f"{message} {context}"

    if prefix:
        message = f"{prefix} {message}"

    suffix = random.choice(OUTCOMES[outcome_key]["suffix"][intent])
    detail = random.choice(OUTCOME_DETAILS[outcome_key])
    message = f"{message}. {suffix} {detail}"

    # Add an occasional natural timing phrase for unresolved cases.
    if outcome_key in {"unresolved_neutral", "unresolved_negative", "escalation"} and i % 5 == 0:
        message = f"{message} {random.choice(TIME_PHRASES)}."

    return add_surface_variation(message, i)


def make_rows() -> list[dict]:
    rows = []
    seen = set()
    conversation_id = 100001

    intent_names = list(INTENTS.keys())
    base_per_intent = TARGET_ROWS // len(intent_names)
    remainder = TARGET_ROWS % len(intent_names)

    for intent_index, intent in enumerate(intent_names):
        target = base_per_intent + (1 if intent_index < remainder else 0)
        generated = 0
        attempts = 0

        while generated < target:
            attempts += 1
            if attempts > target * 50:
                raise RuntimeError(f"Could not generate enough unique rows for {intent}.")

            language = choose_language(generated + intent_index * 19)
            bank = INTENTS[intent][language]
            base = random.choice(bank)
            context = random.choice(INTENTS[intent]["contexts"])
            prefix = random.choice(PREFIXES[language])
            outcome_key = random.choice(ALLOWED_OUTCOMES[intent])

            message = build_message(
                intent, language, base, context, prefix, outcome_key,
                generated + intent_index * 10000
            )

            key = message.lower().strip()
            if key in seen:
                continue

            seen.add(key)

            outcome = OUTCOMES[outcome_key]
            sentiment = outcome["sentiment"]
            resolution = outcome["resolution"]
            escalation = outcome["escalation"]

            if escalation == "Yes":
                priority = "High"
            elif sentiment == "Negative" or resolution == "Unresolved":
                priority = "Medium"
            else:
                priority = "Low"

            rows.append({
                "conversation_id": conversation_id,
                "customer_message": message,
                "agent_response": make_agent_response(intent, outcome_key),
                "language": language,
                "intent": intent,
                "sentiment": sentiment,
                "resolution": resolution,
                "escalation": escalation,
                "priority": priority,
            })

            conversation_id += 1
            generated += 1

    random.shuffle(rows)
    return rows


def main() -> None:
    rows = make_rows()
    df = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)

    print("\nDataset generated successfully")
    print("=" * 60)
    print(f"Rows     : {len(df)}")
    print(f"Columns  : {len(df.columns)}")
    print(f"Output   : {OUTPUT}")
    print("\nIntent distribution:")
    print(df["intent"].value_counts().sort_index())
    print("\nLanguage distribution:")
    print(df["language"].value_counts())
    print("\nSentiment distribution:")
    print(df["sentiment"].value_counts())
    print("\nResolution distribution:")
    print(df["resolution"].value_counts())
    print("\nEscalation distribution:")
    print(df["escalation"].value_counts())


if __name__ == "__main__":
    main()
