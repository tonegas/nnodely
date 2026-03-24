#!/usr/bin/env python3
"""
Esegue tutti gli esempi nnodely tranne example7_connect.
"""
import subprocess
import sys

EXAMPLES = [
    "example.py",
    "example2.py",
    "example3.py",
    "example4.py",
    "example5_custom_layer.py",
    "example6_flatten.py",
    "example_arithmetic.py",
    "example_parameter.py",
]


def main():
    failed = []
    for name in EXAMPLES:
        print(f"\n{'='*60}")
        print(f"Esecuzione: {name}")
        print("=" * 60)
        result = subprocess.run(
            [sys.executable, name],
            capture_output=False,
        )
        if result.returncode != 0:
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"FALLITI: {', '.join(failed)}")
        sys.exit(1)
    print("Tutti gli esempi completati con successo")
    print("=" * 60)


if __name__ == "__main__":
    main()
