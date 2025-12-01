import sys
from pathlib import Path
import textwrap
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

TGT_LANG = "sin_Sinh"
SRC_LANG = "eng_Latn"

translated_sentences = []

def load_translator(model_name: str, src_lang: str, tgt_lang: str):

    print(f"Loading model {model_name}...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("translation", model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang, max_length=512)

def split_to_sentences(text: str, max_length: int = 300):
    # Split by full stops and clean up whitespace
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # If any sentence is still too long, split it further
    final_sentences = []
    for sentence in sentences:
        if len(sentence) > max_length:
            # Use textwrap for sentences that are too long
            wrapped = textwrap.wrap(sentence, width=max_length, break_long_words=False, replace_whitespace=False)
            final_sentences.extend(wrapped)
        else:
            final_sentences.append(sentence)
    
    return final_sentences

def translate_file(input_file: Path, output_file: Path):

    print(f"[INFO] Reading: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    sentences = split_to_sentences(text, max_length=300)

    translator = load_translator(MODEL_NAME, SRC_LANG, TGT_LANG)

    print(f"[INFO] Translating and writing to: {output_file}")
    for sentence in sentences:
        translated = translator(sentence, max_length=512)
        translated_sentences.append(translated[0]['translation_text'])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(translated_sentences))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sin_to_en.py <input_text_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = input_file.stem + "_translated.txt"

    if not input_file.exists():
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)

    translate_file(input_file, output_file)