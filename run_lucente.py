import subprocess
import re
import numpy as np
from pathlib import Path

def run_cmd(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n\nSTDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}"
        )
    return p.stdout.strip()

def parse_probabilities(stdout_text):
    # floats like 0.123, -1.2e-3, 3E+02
    floats = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?", stdout_text)
    if len(floats) < 3:
        raise ValueError(f"Could not find >=3 numbers in output:\n{stdout_text}")
    last3 = list(map(float, floats[-3:]))
    return last3  # [P_e, P_mu, P_tau] (most likely)

def main():
    # --- User choices (match your PyTorch run) ---
    energies = np.linspace(5.0, 16.0, 256)  # MeV
    eta = 1.2                               # radians
    depth_m = 2000.0                        # meters (IMPORTANT: peanuts expects meters) :contentReference[oaicite:3]{index=3}

    # Incoming state:
    # Use -m for incoherent mixture in mass basis: w1,w2,w3 (real weights)
    # Example (replace by your solar/MSW weights or the ones used in your test):
    w1, w2, w3 = 0.30, 0.60, 0.10
    state_mass = f"{w1},{w2},{w3}"

    # Optional: explicit oscillation parameters (if you want to force EXACT same values)
    # Order: th12 th13 th23 delta dm21 dm3l
    # If omitted, peanuts uses defaults (but for fair comparison, pass them).
    th12 = 0.58
    th13 = 0.15
    th23 = 0.85
    delta = 0.0
    dm21 = 7.5e-5
    dm3l = 2.5e-3

    # Mode selection (analytical matches peanuts intended fast mode) :contentReference[oaicite:4]{index=4}
    mode_flag = "--analytical"

    # Output file
    out_path = Path("lucente_pee_eta1p2.txt")

    rows = []
    for E in energies:
        cmd = [
            "python", "run_prob_earth.py",
            mode_flag,
            "-m", state_mass,
            f"{E:.10g}", f"{eta:.10g}", f"{depth_m:.10g}",
            f"{th12:.10g}", f"{th13:.10g}", f"{th23:.10g}", f"{delta:.10g}",
            f"{dm21:.10g}", f"{dm3l:.10g}",
        ]
        stdout = run_cmd(cmd)
        P_e, P_mu, P_tau = parse_probabilities(stdout)
        rows.append((E, P_e))

    arr = np.array(rows, dtype=float)
    np.savetxt(out_path, arr, header="E(MeV) Pee", comments="")
    print(f"Saved {out_path.resolve()}")

if __name__ == "__main__":
    main()
main()