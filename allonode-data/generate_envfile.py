import json

# Load the config file
with open("allonode-data/config.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Extract values from the config
wallet = config["wallet"]
address = wallet["address"]
worker = config["worker"][0]  # Assuming the first worker in the list

# Remove the 'address' field from the config JSON before writing it to the env_file
del wallet["address"]

# Create the output for the env_file
output = f"""ALLORA_OFFCHAIN_NODE_CONFIG_JSON='{json.dumps(config, separators=(',', ':'))}'
ALLORA_OFFCHAIN_ACCOUNT_ADDRESS={address}
NAME={wallet['addressKeyName']}
ENV_LOADED=true
"""

# Save the output to env_file
with open("allonode-data/.env", "w", encoding="utf-8") as env_file:
    env_file.write(output)

print("env file created successfully.")
