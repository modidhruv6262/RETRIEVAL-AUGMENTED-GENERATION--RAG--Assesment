def validate_issue_type(issue_type):
    allowed_types = ['late delivery', 'missing item', 'wrong item']
    if issue_type.lower() not in allowed_types:
        raise ValueError(f"Invalid issue type '{issue_type}'. Allowed types are: {', '.join(allowed_types)}")

def build_structured_prompt(customer_name, order_id, issue_type):
    validate_issue_type(issue_type)
    system_prompt = (
        "You are a professional and helpful food delivery customer support agent. "
        "Assist customers courteously and resolve their order issues accurately. "
        "Your response must strictly not exceed 80 words."
    )
    user_template = (
        f"Customer Name: {customer_name}\n"
        f"Order ID: {order_id}\n"
        f"Reported Issue: {issue_type}\n"
        "Please provide an immediate resolution according to company policy."
    )
    return system_prompt, user_template

test_cases = [
    {"customer_name": "Aarav Sharma", "order_id": "ORD-94812", "issue_type": "late delivery"},
    {"customer_name": "Diya Patel", "order_id": "ORD-44129", "issue_type": "missing item"},
    {"customer_name": "Rohan Verma", "order_id": "ORD-11082", "issue_type": "cold food"}
]

for idx, case in enumerate(test_cases, start=1):
    print(f"=== TEST CASE {idx} ===")
    try:
        sys_prompt, user_prompt = build_structured_prompt(
            case["customer_name"],
            case["order_id"],
            case["issue_type"]
        )
        print("[SYSTEM PROMPT]")
        print(sys_prompt)
        print("\n[USER MESSAGE]")
        print(user_prompt)
        print()
    except ValueError as e:
        print(f"[VALIDATION ERROR]: {e}\n")
