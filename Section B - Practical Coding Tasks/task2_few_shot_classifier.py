few_shot_examples = [
    {
        "input": "My order was supposed to arrive at 8:00 PM, but it is already 9:15 PM and the driver is not answering.",
        "output": "Late Delivery"
    },
    {
        "input": "I ordered a classic chicken burger, but you delivered a paneer tikka sandwich instead.",
        "output": "Wrong Item"
    },
    {
        "input": "The sealed parcel only had garlic bread and coke; the large farmhouse pizza was completely omitted.",
        "output": "Missing Item"
    },
    {
        "input": "The soup was leaked all over the container, totally cold, and tasted stale.",
        "output": "Poor Quality"
    }
]

def add_example(text, label):
    valid_labels = ["Late Delivery", "Wrong Item", "Missing Item", "Poor Quality"]
    if label not in valid_labels:
        raise ValueError(f"Label must be one of: {valid_labels}")
    few_shot_examples.append({"input": text, "output": label})

def build_few_shot_prompt(complaint_text):
    prompt_lines = [
        "You are an automated complaint classification assistant for a food delivery platform.",
        "Classify the customer complaint into exactly one of these categories: 'Late Delivery', 'Wrong Item', 'Missing Item', or 'Poor Quality'.",
        ""
    ]
    for ex in few_shot_examples:
        prompt_lines.append(f"Input: {ex['input']}")
        prompt_lines.append(f"Output: {ex['output']}")
        prompt_lines.append("")
    prompt_lines.append(f"Input: {complaint_text}")
    prompt_lines.append("Output:")
    return "\n".join(prompt_lines)

complaint_1 = "The delivery agent took an extra 50 minutes and my lunch was delayed past my office break."
complaint_2 = "There was no chocolate brownie inside my combo box even though I paid for it."

print("==================== INITIAL FEW-SHOT PROMPT (4 EXAMPLES) ====================")
prompt_1 = build_few_shot_prompt(complaint_1)
print(prompt_1)
print("\n" + "=" * 70 + "\n")

new_text = "I received spicy mutton biryani when I explicitly placed an order for vegetarian veg pulao."
new_label = "Wrong Item"
add_example(new_text, new_label)

print("==================== UPDATED FEW-SHOT PROMPT (5 EXAMPLES AFTER add_example) ====================")
prompt_2 = build_few_shot_prompt(complaint_2)
print(prompt_2)
