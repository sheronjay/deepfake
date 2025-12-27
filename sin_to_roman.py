import sys
import os
from pathlib import Path
from aksharamukha import transliterate

def romanize(input_path, scheme="ISO"):
    # source script name: "Sinhala"
    # target romanization: "ISO" / "IAST" / "ITRANS" / "Roman (Colloquial)" etc.
    
    # Create roman_txt folder if it doesn't exist
    roman_folder = Path("roman_txt")
    roman_folder.mkdir(exist_ok=True)
    
    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Transliterate
    romanized_text = transliterate.process("Sinhala", scheme, content)
    
    # Generate output filename
    output_filename = Path(input_path).stem + "_romanized.txt"
    output_path = roman_folder / output_filename
    
    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(romanized_text)
    
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python sin_to_roman.py input.txt [scheme]")
        print('Example: python sin_to_roman.py in.txt "Roman (Colloquial)"')
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print("Input file not found:", input_path)
        sys.exit(1)

    scheme = sys.argv[2] if len(sys.argv) >= 3 else "ISO"

    output_path = romanize(input_path, scheme)

    print("Done!")
    print("Scheme:", scheme)
    print("Output:", output_path)
    
    return output_path

if __name__ == "__main__":
    main()
