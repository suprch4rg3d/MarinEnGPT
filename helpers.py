import os
import sys
import tomllib
from pprint import pp

def dd(obj):
    pp(obj)
    sys.exit()

def d(obj):
    print(obj)

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_dependency_version(dep_name, file_path="pyproject.toml") -> str | None:
    """
    Returns the version string of a given dependency from pyproject.toml.
    Supports Poetry and PEP 621 standard format.
    """
    try:
        with open(file_path, "rb") as f:
            data = tomllib.load(f)

        # Poetry-style dependencies
        if "tool" in data and "poetry" in data["tool"]:
            deps = data["tool"]["poetry"].get("dependencies", {})
            return deps.get(dep_name)

        # PEP 621-style dependencies
        if "project" in data:
            for entry in data["project"].get("dependencies", []):
                if entry.startswith(dep_name + " "):
                    return entry.split(" ", 1)[1].strip()
                elif entry == dep_name:
                    return None  # No version specified

        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


# ANSI escape codes for custom RGB foreground colors (truecolor terminals)
def rgb(r, g, b):
    return f"\033[38;2;{r};{g};{b}m"
