from src.agent import get_recommendations

# Test cases
test_queries = [
    "I drive 20 km daily using a petrol car. How can I reduce my CO₂ emissions?",  # Happy path
    "",  # Empty query (should handle gracefully, but since no UI, just run)
    "Weather today?",  # Irrelevant query
    "I eat meat every day and fly frequently. Comprehensive plan to reduce emissions over a year.",  # Complex query
    "What is the CO2 emission for cooking rice?"  # Specific activity query
]

print("Testing Agent Responses:\n")

for i, query in enumerate(test_queries, 1):
    if query.strip():  # Skip empty for now
        try:
            response = get_recommendations(query)
            print(f"Test {i} - Query: {query}")
            print(f"Response: {response}\n")
        except Exception as e:
            print(f"Test {i} - Query: {query} - Error: {e}\n")
    else:
        print("Test 2 - Empty query: Handled in UI, not testable here.\n")
