import dspy


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
    """

    ticket: str = dspy.InputField(
        desc="The customer's support request."
    )

    category: str = dspy.OutputField(
        desc=(
            "Output ONLY one category name: "
            "Billing, Technical, Account, or Subscription.\n\n"
            "Billing: charges, payments, invoices, refunds, duplicate charges.\n"
            "Technical: application bugs, crashes, freezes, errors, "
            "broken features, settings/features such as dark mode, "
            "or the application not functioning correctly.\n"
            "Account: login, profile details, account email, password, "
            "account settings, or unauthorized account changes.\n"
            "Subscription: subscribing, cancelling, changing plans, "
            "subscription status, or a paid subscription not working "
            "when the account itself is otherwise accessible."
        )
    )

    priority: str = dspy.OutputField(
        desc=(
            "Choose exactly one: Low, Medium, or High.\n\n"
            "High: unauthorized access, suspected account takeover, "
            "financial loss, duplicate/incorrect charges, security "
            "incidents, major service outages, or a subscription/payment "
            "failure causing significant service loss.\n"
            "Medium: login problems, application errors, or broken "
            "functionality that prevents normal use but is not a major "
            "security or financial incident.\n"
            "Low: informational questions, invoices/statements, settings "
            "changes, updating profile information, changing subscription "
            "plans, cancelling a subscription, or other non-urgent requests."
        )   
    )

    sentiment: str = dspy.OutputField(
        desc=(
            "Output ONLY one sentiment: "
            "Positive, Neutral, or Negative."
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
    """Generate a useful response plan for a support ticket."""

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
        desc="A concise summary of the customer's issue."
    )

    recommended_action: str = dspy.OutputField(
        desc="The recommended next action for the support team."
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


VALID_CATEGORIES = {
    "billing",
    "technical",
    "account",
    "subscription",
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

# -----------------------------
# Complete TicketFlow pipeline
# -----------------------------

class TicketAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()

        self.classifier = TicketClassifier()
        self.responder = TicketResponder()

    def forward(self, ticket):
        classification = self.classifier(ticket)

        category, priority, sentiment = validate_classification(
            classification
        )

        response = self.responder(
            ticket=ticket,
            category=category,
            priority=priority,
        )

        return dspy.Prediction(
            category=category,
            priority=priority,
            sentiment=sentiment,
            summary=response.summary,
            recommended_action=response.recommended_action,
        )

analyzer = TicketAnalyzer()


def analyze_ticket(ticket: str):
    return analyzer(ticket=ticket)