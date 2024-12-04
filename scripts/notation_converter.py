"""
notation_converter.py
~~~~~~~~~~~~~~~~~~~~~
Converts LER notation containing UTF-16 symbols to strict ASCII for the LER MLIR compiler.
"""

import sys


def main():
    if len(sys.argv) < 3:
        print("Usage: <file to convert> <out file>")
        exit(1)
    filename = sys.argv[1]
    converted_file = sys.argv[2];
    old_ler = None
    new_ler = None

    with open(filename, "r") as ler_file:
        old_ler = ler_file.read()

    # Do replacements
    new_ler = old_ler.replace("∫", "|")
    new_ler = new_ler.replace("Σ", "^S")
    new_ler = new_ler.replace("Γ", "^R")
    new_ler = new_ler.replace("Π", "^P")
    new_ler = new_ler.replace("Ψ", "^W")

    with open(converted_file, "w") as new_ler_file:
        new_ler_file.write(new_ler)


if __name__ == "__main__":
    main()
