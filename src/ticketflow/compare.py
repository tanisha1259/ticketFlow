import dspy

from analyzer import TicketClassifier
from evaluation import examples, ticket_metric


def evaluate_classifier(name, classifier):
    passed = 0

    print(f"\n========== {name} ==========\n")

    for i, example in enumerate(examples, start=1):
        prediction = classifier(ticket=example.ticket)

        score = ticket_metric(example, prediction)

        if score:
            passed += 1

        print(
            f"Case {i}: "
            f"{'PASS' if score else 'FAIL'} | "
            f"Expected: {example.category} | "
            f"{example.priority} | "
            f"{example.sentiment} | "
            f"Predicted: {prediction.category} | "
            f"{prediction.priority} | "
            f"{prediction.sentiment}"
        )

    accuracy = passed / len(examples) * 100

    print(f"\nScore: {passed}/{len(examples)}")
    print(f"Accuracy: {accuracy:.1f}%")

    return accuracy


def main():
    original = TicketClassifier()

    optimized = TicketClassifier()
    optimized.load("optimized_classifier.json")

    original_score = evaluate_classifier(
        "ORIGINAL CLASSIFIER",
        original,
    )

    optimized_score = evaluate_classifier(
        "OPTIMIZED CLASSIFIER",
        optimized,
    )

    print("\n========== COMPARISON ==========")
    print(f"Original:  {original_score:.1f}%")
    print(f"Optimized: {optimized_score:.1f}%")
    print("================================")


if __name__ == "__main__":
    main()