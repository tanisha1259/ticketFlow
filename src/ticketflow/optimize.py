import dspy

from analyzer import TicketClassifier
from dataset import development_examples
from evaluation import ticket_metric


def main():
    classifier = TicketClassifier()

    optimizer = dspy.BootstrapFewShot(
        metric=ticket_metric,
        max_bootstrapped_demos=2,
        max_labeled_demos=4,
    )

    optimized_classifier = optimizer.compile(
        classifier,
        trainset=development_examples,
    )

    optimized_classifier.save("optimized_classifier.json")

    print("\n========== OPTIMIZATION COMPLETE ==========\n")
    print("Optimized classifier saved to:")
    print("optimized_classifier.json")


if __name__ == "__main__":
    main()