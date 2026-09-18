## Import the necessary modules
import json
from unittest import result
## Import the function from the module parse_data
from parse_data import load_items, get_unclaimed_items, save_result
import ollama

## Build your prompt based on the description the user provides 
## and the items that are available in the lost-and-found database.
## The model must follow the rules listed in the README file
## The function should return the system prompt and the user prompt.
## You may need to use json.dumps() to convert the available_items list into a JSON string.

def build_prompt(description, available_items):
    system_prompt = (
        "You are a campus lost-and-found assistant. "
        "Your job is to find POSSIBLE matches, not exact matches. "
        "Use semantic reasoning: "
        "'bag' matches 'backpack', "
        "'bottle' matches 'water bottle', "
        "'charger' matches 'laptop charger'. "
        "IMPORTANT: The ITEM TYPE must be semantically compatible first. "
        "Color and location are secondary hints, not primary criteria. "
        "Do NOT match items whose type is unrelated, even if the color matches. "
        "(for example, a 'bag' should NOT match a 'charger' or a 'bottle'). "
        "If the item type OR color OR location partially matches, INCLUDE it. "
        "Be inclusive, not strict. "
        "You must return ONLY JSON, no explanation, no markdown, exactly this structure:\n"
        '{"matches": ["ITEM_ID"], "confidence": "LOW"}\n'
        "Rules:\n"
        '- "matches" is a list of item IDs that could be the lost item.\n'
        '- "confidence" is exactly one of: LOW, MEDIUM, HIGH.\n'
        '- If truly nothing matches, return {"matches": [], "confidence": "LOW"}.\n'
        "\n"
        "Examples:\n"
        'User: "I lost a black bag"\n'
        'Items: [{"id": "F101", "item": "backpack", "color": "black"}, '
        '{"id": "F104", "item": "laptop charger", "color": "black"}]\n'
        'Answer: {"matches": ["F101"], "confidence": "HIGH"}\n'
        "\n"
        'User: "I lost a black charger"\n'
        'Items: [{"id": "F101", "item": "backpack", "color": "black"}, '
        '{"id": "F104", "item": "laptop charger", "color": "black"}]\n'
        'Answer: {"matches": ["F104"], "confidence": "HIGH"}\n'
        "\n"
        'User: "I lost a pink umbrella"\n'
        'Items: [{"id": "F101", "item": "backpack", "color": "black"}]\n'
        'Answer: {"matches": [], "confidence": "LOW"}\n'
    )

    user_prompt = (
        f"User lost: {description}\n\n"
        f"Available items:\n{json.dumps(available_items, indent=4)}\n\n"
        "Return the JSON now."
    )

    return system_prompt, user_prompt
    
## Logic to ask Qwen for all the possible matches based on the system prompt and user prompt.
## The function should return the response from Qwen.
def ask_qwen(system_prompt, user_prompt):
    response = ollama.chat(
        model="qwen3:8b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": 0.1,
            "top_p": 0.9,
        },
    )
    return response["message"]["content"]


## Logic to parse the response from Qwen and return the result. 
## You may need to use json.loads() to convert the response string into a suitable Python data structure.
def parse_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    
## Logic to validate the result returned by Qwen.
## It should check if the result is a dictionary, contains the keys "matches" and "confidence", and that the values are of the correct type.
## If everything is correct, then it should check if the item IDs in the "matches" list are valid IDs .
def validate_result(result, available_items):
    if not isinstance(result, dict):
        return False
    if "matches" not in result or "confidence" not in result:
        return False
    if not isinstance(result["matches"], list):
        return False
    if not isinstance(result["confidence"], str):
        return False
    if result["confidence"] not in ("LOW", "MEDIUM", "HIGH"):
        return False

    valid_ids = []
    for item in available_items:
        valid_ids.append(item["id"])
    for match in result["matches"]:
        if match not in valid_ids:
            return False
    return True

## Logic to display the matches found by Qwen in a user-friendly format.
## It should look something like this:
""" 
CAMPUS LOST-AND-FOUND ASSISTANT
==================================================

Describe the item you lost: I lost a black bag somewhere

Searching for possible matches...

MATCH RESULT
--------------------------------------------------
Confidence: MEDIUM

Possible matches:

ID: F101
Item: backpack
Color: black
Location: Library 2nd floor
Date found: 2026-09-15

Result saved to output/match_result.json
 """
## If no matches are found, it should display a message indicating that no matches were found, along with the empty list
def display_matches(result, available_items):
    print("\nMATCH RESULT")
    print("--------------------------------------------------")
    print(f"Confidence: {result['confidence']}")
    print()

    if not result["matches"]:
        print("No matches were found.")
        return

    print("Possible matches:")
    print()
    items_by_id = {}
    for item in available_items:
        items_by_id[item["id"]] = item

    for match_id in result["matches"]:
        if match_id in items_by_id:
            item = items_by_id[match_id]
            print(f"ID: {item['id']}")
            print(f"Item: {item['item']}")
            print(f"Color: {item['color']}")
            print(f"Location: {item['location']}")
            print(f"Date found: {item['date']}")
            print()

## Control center for the entire program.
def main():
    print("CAMPUS LOST-AND-FOUND ASSISTANT")
    print("==================================================")
    print()

    description = input("Describe the item you lost: ")
    print()
    print("Searching for possible matches...")

    items = load_items("found_items.json")
    available_items = get_unclaimed_items(items)

    system_prompt, user_prompt = build_prompt(description, available_items)
    response_text = ask_qwen(system_prompt, user_prompt)
    result = parse_response(response_text)

    if result is None:
        print("Error: The model did not return valid JSON.")
        return

    if not validate_result(result, available_items):
        print("Error: The model returned an invalid result.")
        return

    display_matches(result, available_items)

    output_filename = "output/match_result.json"
    save_result(result, output_filename)
    print(f"Result saved to {output_filename}")

if __name__ == "__main__":
    main()