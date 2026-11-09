#!/usr/bin/env python3
"""
Prepare notebooks for submission. Works on COPIES only; the originals are never changed.

    python3 clean_notebooks.py SOURCE_DIR TARGET_DIR            # report only
    python3 clean_notebooks.py SOURCE_DIR TARGET_DIR --apply    # write cleaned copies

What --apply does in each copied notebook:
  * removes cells whose output is an error (traceback)            -> failed attempts
  * removes empty cells
  * removes output lines containing Earth Engine verification codes / tokens
  * removes verbose library warnings (FutureWarning / pip notices) from outputs
It does NOT change any code or any normal output, so code and results stay consistent.
"""
import json, os, re, shutil, sys

SECRET = re.compile(r"(verification code|4/1A[0-9A-Za-z_\-]{10,}|ya29\.[0-9A-Za-z_\-]+|refresh_token|client_secret)",
                    re.IGNORECASE)
NOISE = re.compile(r"(FutureWarning|warnings\.warn\(|\[notice\]|Post-Quantum Cryptography)")


def cell_has_error(cell):
    return any(o.get("output_type") == "error" for o in cell.get("outputs", []))


def clean_text(lines):
    removed = 0
    if isinstance(lines, str):
        lines = lines.splitlines(keepends=True)
    keep = []
    for ln in lines:
        if SECRET.search(ln) or NOISE.search(ln):
            removed += 1
        else:
            keep.append(ln)
    return keep, removed


def process(path, out_path, apply):
    nb = json.load(open(path, encoding="utf-8"))
    report = {"error_cells": 0, "empty_cells": 0, "secret_or_noise_lines": 0, "print_lines": 0, "cells": 0}
    new_cells = []
    for c in nb.get("cells", []):
        src = "".join(c.get("source", []))
        report["cells"] += 1
        if c.get("cell_type") == "code":
            report["print_lines"] += sum(1 for ln in src.splitlines() if ln.strip().startswith("print("))
        if not src.strip():
            report["empty_cells"] += 1
            continue
        if c.get("cell_type") == "code" and cell_has_error(c):
            report["error_cells"] += 1
            continue
        if c.get("cell_type") == "code":
            outs = []
            for o in c.get("outputs", []):
                if o.get("name") == "stderr" and NOISE.search("".join(o.get("text", ""))) \
                        and not SECRET.search("".join(o.get("text", ""))):
                    txt, r = clean_text(o.get("text", ""))
                    report["secret_or_noise_lines"] += r
                    if not "".join(txt).strip():
                        continue
                    o["text"] = txt
                elif "text" in o:
                    o["text"], r = clean_text(o["text"])
                    report["secret_or_noise_lines"] += r
                    if not "".join(o["text"]).strip():
                        continue
                outs.append(o)
            c["outputs"] = outs
        new_cells.append(c)
    if apply:
        nb["cells"] = new_cells
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(nb, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return report


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, dst, apply = sys.argv[1], sys.argv[2], "--apply" in sys.argv
    files = sorted(f for f in os.listdir(src) if f.endswith(".ipynb") and not f.startswith("Untitled"))
    print(f"{'notebook':45s} cells  errors  empty  secret/noise  print-lines")
    for f in files:
        r = process(os.path.join(src, f), os.path.join(dst, f), apply)
        print(f"{f:45s} {r['cells']:5d} {r['error_cells']:7d} {r['empty_cells']:6d} "
              f"{r['secret_or_noise_lines']:13d} {r['print_lines']:12d}")
    print("\n" + (f"Cleaned copies written to: {dst}" if apply else "Report only - run again with --apply to write cleaned copies."))


if __name__ == "__main__":
    main()
