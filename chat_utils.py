import json
def load_income_type_tips():
    with open("knowledge/income_type_tips.json", "r") as f:
        return json.load(f)