import json

with open("novelish_metrics.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        print("".join(cell["source"]))
