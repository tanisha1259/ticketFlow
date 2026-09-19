import dspy

from analyzer import TicketClassifier


# Smaller development set for fast iteration
examples = [
    dspy.Example(
        ticket="I was charged twice for my subscription.",
        category="Billing",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Can I get a copy of my invoice for last month?",
        category="Billing",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="The application crashes every time I try to upload a file.",
        category="Technical",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Is there a way to enable dark mode in the application?",
        category="Technical",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="I cannot log into my account even though my password is correct.",
        category="Account",
        priority="Medium",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="How can I change the email address associated with my account?",
        category="Account",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="I'd like to cancel my subscription.",
        category="Subscription",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="My subscription was cancelled unexpectedly.",
        category="Subscription",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),
]


def ticket_metric(example, prediction, trace=None):
    category_correct = (
        example.category.lower().strip()
        == prediction.category.lower().strip()
    )

    priority_correct = (
        example.priority.lower().strip()
        == prediction.priority.lower().strip()
    )

    sentiment_correct = (
        example.sentiment.lower().strip()
        == prediction.sentiment.lower().strip()
    )

    return (
        category_correct
        and priority_correct
        and sentiment_correct
    )


def evaluate():
    classifier = TicketClassifier()

    print("\n========== CLASSIFIER EVALUATION ==========\n")

    passed = 0

    for i, example in enumerate(examples, start=1):
        prediction = classifier(ticket=example.ticket)

        category_ok = (
            example.category.lower().strip()
            == prediction.category.lower().strip()
        )

        priority_ok = (
            example.priority.lower().strip()
            == prediction.priority.lower().strip()
        )

        sentiment_ok = (
            example.sentiment.lower().strip()
            == prediction.sentiment.lower().strip()
        )

        case_passed = category_ok and priority_ok and sentiment_ok

        if case_passed:
            passed += 1

        print(f"Case {i}: {'PASS' if case_passed else 'FAIL'}")
        print(f"Ticket: {example.ticket}")
        print(
            f"Expected: "
            f"{example.category} | "
            f"{example.priority} | "
            f"{example.sentiment}"
        )
        print(
            f"Predicted: "
            f"{prediction.category} | "
            f"{prediction.priority} | "
            f"{prediction.sentiment}"
        )
        print()

    print("-------------------------------------------")
    print(f"Score: {passed}/{len(examples)}")
    print(f"Accuracy: {passed / len(examples) * 100:.1f}%")
    print("===========================================")


if __name__ == "__main__":
    evaluate()