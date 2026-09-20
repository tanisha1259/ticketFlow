import dspy


# Examples used during development / DSPy optimization
development_examples = [
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

    dspy.Example(
        ticket="What are your customer support hours?",
        category="General",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Can I speak with a human support agent?",
        category="General",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),
]


# Completely unseen examples for final testing
test_examples = [
    dspy.Example(
        ticket="The same payment was taken from my card twice.",
        category="Billing",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Where can I download my previous billing statement?",
        category="Billing",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="The app freezes whenever I try to upload a document.",
        category="Technical",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="How do I turn on dark mode?",
        category="Technical",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Someone appears to have changed my account email without my permission.",
        category="Account",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="I'd like to update the email linked to my profile.",
        category="Account",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="How can I switch to a different subscription plan?",
        category="Subscription",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="My paid plan stopped working even though I haven't cancelled it.",
        category="Subscription",
        priority="High",
        sentiment="Negative",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="Where can I find information about your company?",
        category="General",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="How do I contact someone from your support team?",
        category="General",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),

    dspy.Example(
        ticket="What time is your customer service available?",
        category="General",
        priority="Low",
        sentiment="Neutral",
    ).with_inputs("ticket"),
]