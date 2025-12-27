import sys
import json
from pathlib import Path
import textwrap
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/nllb-200-distilled-600M"

TGT_LANG = "sin_Sinh"
SRC_LANG = "eng_Latn"

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

def translate_file(input_file: Path):

    print(f"[INFO] Reading: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    
    translator = load_translator(MODEL_NAME, SRC_LANG, TGT_LANG)
    
    print(f"[INFO] Translating {len(segments)} segments")
    translated_segments = []
    
    for i, segment in enumerate(segments):
        print(f"[INFO] Translating segment {i+1}/{len(segments)}")
        original_text = segment['text']
        translated = translator(original_text, max_length=512)
        
        # Create new segment with translated text
        translated_segment = {
            "start": segment['start'],
            "end": segment['end'],
            "text": translated[0]['translation_text']
        }
        translated_segments.append(translated_segment)
    
    # Ensure translated_txt folder exists
    translated_txt = Path("translated_txt")
    translated_txt.mkdir(exist_ok=True)
    
    output_file = translated_txt / f"{input_file.stem}_translated.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(translated_segments, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Saved translated JSON to: {output_file}")
    return output_file
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sin_to_en.py <input_text_file>")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    

    if not input_file.exists():
        print(f"Error: The file {input_file} does not exist.")
        sys.exit(1)

    translated_path = translate_file(input_file)