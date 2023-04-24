from pathlib import Path
from predict import model_path

# print current files in directory, and size of file
print("Current files in directory:")
for file in Path(".").glob("*"):
    print(file, file.stat().st_size)

assert Path(model_path).exists()
