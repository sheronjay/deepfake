import sys
import os
from pathlib import Path
from aksharamukha import transliterate
import json

def romanize(input_path, scheme="ISO"):
    # source script name: "Sinhala"
    # target romanization: "ISO" / "IAST" / "ITRANS" / "Roman (Colloquial)" etc.
    
    # Read input file
    with open(input_path, "r", encoding="utf-8") as f:
        segments = json.load(f)
    

    for segment in segments:
        segment['roman'] = transliterate.process("Sinhala", scheme, segment['text'])
    
    # Save to file
    with open(input_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return input_file

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
