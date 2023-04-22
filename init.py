from pathlib import Path

# print current files in directory, and size of file
print("Current files in directory:")
for file in Path(".").glob("*"):
    print(file, file.stat().st_size)

assert Path("ggml-model-f16.bin").exists()
