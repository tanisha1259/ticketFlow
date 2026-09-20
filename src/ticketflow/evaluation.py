import dspy

from .analyzer import TicketClassifier
from .dataset import test_examples


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

    for i, example in enumerate(test_examples, start=1):
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
    print(f"Score: {passed}/{len(test_examples)}")
    print(f"Accuracy: {passed / len(test_examples) * 100:.1f}%")
    print("===========================================")


if __name__ == "__main__":
    evaluate()