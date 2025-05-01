import re
import subprocess

# r = subprocess.run(
#     ["ninja", "LLVMAMDGPUCodeGen"],
#     cwd="/home/mlevental/dev_projects/llvm-project/cmake-build-debug",
#     capture_output=True,
# )
#
# print(r.stdout.decode())
# print(r.stderr.decode())

f = open("/home/mlevental/.config/JetBrains/CLion2025.1/scratches/scratch_98.txt").read()
# print(f)

errors = re.findall(r"^(.*?):(\d+):(\d+): error: (.*?)$", f, flags=re.MULTILINE)
for e in errors:
    print(e[3])

# redefs = re.findall(r"^(.*?):(\d+):(\d+): error: redefinition of '(\w+)'$", f, flags=re.MULTILINE)
# for r in redefs:
#     print(r)