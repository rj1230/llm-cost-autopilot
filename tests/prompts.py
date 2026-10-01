BASELINE_PROMPTS: list[str] = [
    # Tier 1 - simple
    "Extract the email address from this text: 'Contact Raj at raj418060@gmail.com for details.'",
    "What is the capital of France? Answer in one word.",
    "Reformat this name from 'Rajput, Raj' to 'Raj Rajput'.",
    # Tier 2 - moderate
    (
        "Summarize the following in two sentences: 'The company reported record "
        "quarterly revenue driven by strong cloud services growth, though "
        "operating margins narrowed due to increased infrastructure spending.'"
    ),
    (
        "Classify the sentiment of this review as positive, negative, or neutral:"
        "'The product works fine but shipping took way too long.'"
    ),
    "List three pros and three cons of using microservices architecture.",
    (
        "Given this JSON schema, generate one valid example object: "
        '{"name": "string", "age": "integer", "is_active": "boolean"}'
    ),
    # Tier 3 - complex
    (
        "A train leaves City A at 60 km/h heading to City B, 300km away. "
        "Another train leaves City B at the same time heading to City A at 90 "
        "km/h. How far from City A do they meet, and how long does it take? "
        "Show your reasoning step by step."
    ),
    (
        "Write a short, original four-line poem about a programmer debugging "
        "code late at night. Do not reuse lines from any existing published poem."
    ),
    (
        "Compare and contrast REST and GraphQL API design for a high-traffic "
        "e-commerce backend, and recommend one with justification."
    ),
]
