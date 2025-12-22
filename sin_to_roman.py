import sys
import os
from aksharamukha import transliterate

def romanize(text, scheme="ISO"):
    # source script name: "Sinhala"
    # target romanization: "ISO" / "IAST" / "ITRANS" / "Roman (Colloquial)" etc.
    return transliterate.process("Sinhala", scheme, text)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sin_to_roman.py input.txt [output.txt] [scheme]")
        print('Example: python sin_to_roman.py in.txt out.txt "Roman (Colloquial)"')
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print("Input file not found:", input_path)
        sys.exit(1)

    output_path = sys.argv[2] if len(sys.argv) >= 3 else (
        os.path.join("roman_txt", os.path.basename(os.path.splitext(input_path)[0]) + "_romanized.txt")
    )
    scheme = sys.argv[3] if len(sys.argv) >= 4 else "ISO"
    
    # Create roman_txt folder if it doesn't exist
    os.makedirs("roman_txt", exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    romanized = romanize(content, scheme=scheme)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(romanized)

    print("Done!")
    print("Scheme:", scheme)
    print("Output:", output_path)

if __name__ == "__main__":
    main()
