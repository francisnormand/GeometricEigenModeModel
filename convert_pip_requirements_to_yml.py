# convert_requirements_to_yml.py
import sys
import ast
import os
import importlib.metadata

folder = os.getcwd()  # folder containing your Python scripts
imports = set()

for root, dirs, files in os.walk(folder):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            with open(path, "r") as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for n in node.names:
                            imports.add(n.name.split(".")[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module.split(".")[0])

# for pkg in sorted(imports):
#     print(pkg)




installed = {pkg.metadata['Name'].lower(): pkg.version 
             for pkg in importlib.metadata.distributions()}

# Keep only packages actually imported
minimal_packages = {}
for pkg in imports:
    name = pkg.lower()
    if name in installed:
        minimal_packages[name] = installed[name]

print(minimal_packages)

with open("environment.yml", "w") as f:
    f.write("name: geom_eigen_model\n")
    f.write("channels:\n  - defaults\n  - conda-forge\n")
    f.write("dependencies:\n  - python=3.11\n  - pip\n")
    f.write("  - pip:\n")
    for pkg, ver in minimal_packages.items():
        f.write(f"      - {pkg}=={ver}\n")