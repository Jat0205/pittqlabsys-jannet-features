import json


data = {
    "experiment_name": "CONFOCAL",
    "DATE": "10-07-2025",
    "Finished": False,
    "saples": ["nanodiamond_01", "nanodiamond_02"]
}
# Serialize to a JSON string
json_string = json.dumps(data, indent=4)  # indent for pretty printing
print(json_string)

# Serialize to a file
with open("data.json", "w") as f:
    json.dump(data, f, indent=4)
json_string = '{"experiment_name": "ODMR", "DATE": "10-10-2025", "Finished": true, "saples": ["nanodiamond_09", "nanodiamond_10"]}'

# Deserialize from a JSON string
python_obj = json.loads(json_string)
print(python_obj["experiment_name"])

# Deserialize from a file
with open("data.json", "r") as f:
    loaded_data = json.load(f)
print(loaded_data["saples"])



