from TTS.api import TTS

tts = TTS(
    model_path="tts_model/checkpoint_80000.pth",
    config_path="tts_model/config.json",
    gpu=False
)

chars = tts.synthesizer.tts_config.characters

print("Supported characters:")
print(chars)

print("\nAs a set:")
print(set(chars))
