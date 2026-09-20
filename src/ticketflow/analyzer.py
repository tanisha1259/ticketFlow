import dspy

import json
from dataclasses import asdict, dataclass

import time

# -----------------------------
# Ollama + Qwen3 configuration
# -----------------------------

lm = dspy.LM(
    "ollama_chat/qwen3:4b",
    api_base="http://localhost:11434",
    api_key="ollama",
    model_type="chat",
    temperature=0,
    think=False,
)

dspy.configure(lm=lm)


# -----------------------------
# Ticket Classification
# -----------------------------

class TicketClassification(dspy.Signature):
    """
    Classify a customer support ticket.

    Categories:
    - Billing: payments, charges, refunds, invoices
    - Technical: bugs, crashes, errors, broken functionality
    - Account: profile, login, email, account settings
    - Subscription: subscribing, cancelling, changing subscription
    - General: questions or requests that do not clearly belong to the other four categories
    """

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.OutputField(
        desc=(
            "Output ONLY one category name: "
            "Billing, Technical, Account, Subscription, or General.\n\n"

            "Billing: charges, payments, invoices, refunds, duplicate "
            "charges, or other payment-related issues.\n"

            "Technical: application bugs, crashes, freezes, errors, "
            "broken features, dark mode problems, upload failures, or "
            "the application not functioning correctly.\n"

            "Account: login, profile details, account email, password, "
            "account settings, or unauthorized account changes.\n"

            "Subscription: subscribing, cancelling, changing plans, "
            "subscription status, or a paid subscription not working "
            "when the account itself is otherwise accessible.\n"

            "General: questions or requests that do not clearly belong "
            "to Billing, Technical, Account, or Subscription. Examples "
            "include support hours, company information, contacting a "
            "human agent, or other general questions.\n\n"

            "Choose General when none of the four specialized categories "
            "clearly applies. Do not force a ticket into a specialized "
            "category."
        )
    )

    priority: str = dspy.OutputField(
        desc=(
            "Choose exactly one: Low, Medium, or High.\n\n"

            "High: unauthorized account access, suspected account takeover, "
            "financial loss, duplicate or incorrect charges, security "
            "incidents, major service outages, or a payment/subscription "
            "failure that prevents the customer from receiving a paid service.\n"

            "Medium: login problems, application errors, crashes, freezes, "
            "or broken functionality that prevents normal use but is not "
            "a major security, financial, or service-loss incident.\n"

            "Low: informational questions, invoices/statements, settings "
            "changes, updating profile information, changing subscription "
            "plans, cancelling a subscription, or other non-urgent requests.\n\n"

            "If the customer reports a complete failure of a paid service "
            "or an important application function, prefer High when the "
            "failure prevents normal use."
        )
    )

    sentiment: str = dspy.OutputField(
        desc=(
            "Output ONLY one sentiment: Positive, Neutral, or Negative.\n\n"

            "Negative: the customer reports a problem, failure, error, "
            "loss, unauthorized action, frustration, or inability to use "
            "a service.\n"

            "Neutral: the customer is asking an informational question "
            "or requesting a normal change without reporting a problem.\n"

            "Positive: the customer expresses satisfaction, praise, "
            "thanks, or a clearly positive experience."
        )
    )

class TicketClassifier(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.Predict(TicketClassification)

    def forward(self, ticket):
        return self.classify(ticket=ticket)


# -----------------------------
# Ticket Response
# -----------------------------

class TicketResponseSignature(dspy.Signature):
    """Generate a concise support response plan."""

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.InputField(
        desc="The ticket category."
    )

    priority: str = dspy.InputField(
        desc="The ticket priority."
    )

    summary: str = dspy.OutputField(
        desc="One short sentence summarizing the customer's request."
    )

    recommended_action: str = dspy.OutputField(
        desc="One short sentence describing the next action for support."
    )


class TicketResponder(dspy.Module):
    def __init__(self):
        super().__init__()
        self.respond = dspy.Predict(TicketResponseSignature)

    def forward(self, ticket, category, priority):
        return self.respond(
            ticket=ticket,
            category=category,
            priority=priority,
        )
        summary, recommended_action = validate_response(response)
        total_time = time.perf_counter() - total_start

VALID_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "subscription",
    "general",
}

VALID_PRIORITIES = {
    "low",
    "medium",
    "high",
}

VALID_SENTIMENTS = {
    "positive",
    "neutral",
    "negative",
}


def normalize_value(value: str) -> str:
    """Clean common formatting artifacts from model output."""
    return value.strip().replace("#", "").replace("]", "").replace("[", "").strip()


def validate_classification(classification):
    """Validate and normalize classifier output."""

    category = normalize_value(classification.category)
    priority = normalize_value(classification.priority)
    sentiment = normalize_value(classification.sentiment)

    if category.lower() not in VALID_CATEGORIES:
        raise ValueError(f"Invalid category returned by model: {category}")

    if priority.lower() not in VALID_PRIORITIES:
        raise ValueError(f"Invalid priority returned by model: {priority}")

    if sentiment.lower() not in VALID_SENTIMENTS:
        raise ValueError(f"Invalid sentiment returned by model: {sentiment}")

    return category, priority, sentiment

def validate_response(response):
    """Validate responder output and reject placeholder text."""

    summary = response.summary.strip()
    recommended_action = response.recommended_action.strip()

    invalid_values = {
        "",
        "[summary text]",
        "[recommended action text]",
        "{summary}",
        "{recommended_action}",
    }

    if summary.lower() in {value.lower() for value in invalid_values}:
        raise ValueError("Model returned an invalid summary.")

    if recommended_action.lower() in {
        value.lower() for value in invalid_values
    }:
        raise ValueError("Model returned an invalid recommended action.")

    return summary, recommended_action

# -----------------------------
# Structured application result
# -----------------------------

@dataclass
class TicketResult:
    ticket: str
    category: str
    priority: str
    sentiment: str
    summary: str
    recommended_action: str

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

# -----------------------------
# Complete TicketFlow pipeline
# -----------------------------

class TicketAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()

        self.classifier = TicketClassifier()
        self.responder = TicketResponder()

    def forward(self, ticket):
        total_start = time.perf_counter()
        classification_start = time.perf_counter()
        classification = self.classifier(ticket)
        classification_time = time.perf_counter() - classification_start

        category, priority, sentiment = validate_classification(
            classification
        )

        response_start = time.perf_counter()

        response = self.responder(
            ticket=ticket,
            category=category,
            priority=priority,
        )

        summary, recommended_action = validate_response(response)
        response_time = time.perf_counter() - response_start
        total_time = time.perf_counter() - total_start

        print(
            f"\n[Timing] Classification: {classification_time:.2f}s | "
            f"Response: {response_time:.2f}s | "
            f"Total: {total_time:.2f}s"
        )

        return TicketResult(
            ticket=ticket,
            category=category,
            priority=priority,
            sentiment=sentiment,
            summary=summary,
            recommended_action=recommended_action,
        )

analyzer = TicketAnalyzer()


def analyze_ticket(ticket: str):
    return analyzer(ticket=ticket)