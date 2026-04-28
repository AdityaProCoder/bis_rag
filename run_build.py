import subprocess
import sys

result = subprocess.run(
    [sys.executable, r"D:\Project\bis_rag\build_index.py"],
    capture_output=True,
    text=True,
    timeout=900
)
with open(r"D:\Project\bis_rag\build_out.txt", "w") as f:
    f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nRETURN CODE: {result.returncode}")
print(f"Return code: {result.returncode}")
print(f"Stdout length: {len(result.stdout)}")
print(f"Stderr length: {len(result.stderr)}")
