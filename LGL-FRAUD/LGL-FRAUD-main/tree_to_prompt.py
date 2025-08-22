import json
import pprint
import os





def convert_tree_to_prompt(tree):
    lines = []

    lines.append("Graph Information:")

    # --- Feature Section ---
    lines.append("Feature:")
    lines.append(f"  Center-node:")
    lines.append(f"    Node {tree['center_id']} → {tree['Feature']['center']}")

    for rel, neighbors in tree['Feature']['1hop'].items():
        if neighbors:
            lines.append(f"  1st-hop ({rel}):")
            for nbr, label in neighbors.items():
                lines.append(f"    Node {nbr} → {label}")

    # --- Label Section ---
    lines.append("\nLabel:")
    for rel, labeled_neighbors in tree['Label'].items():
        if labeled_neighbors:
            lines.append(f"  1st-hop ({rel}):")
            for nbr, true_label in labeled_neighbors.items():
                lines.append(f"    Node {nbr} → Label: {true_label}")

    # --- Question ---
    lines.append("\nQuestion: Is the center node fraudulent?")
    lines.append("Answer:")

    return "\n".join(lines)



with open("syntax_trees.json", "r") as f:
    syntax_trees = json.load(f)
    


print("Generating all prompts...")
all_prompts = [convert_tree_to_prompt(tree) for tree in syntax_trees]

output_path = "all_prompts.txt"

with open(output_path, "w", encoding="utf-8") as f:
    for i, prompt in enumerate(all_prompts):
        f.write(f"### Prompt {i + 1} ###\n")
        f.write(prompt)
        f.write("\n\n" + "="*60 + "\n\n")  # separator between prompts

