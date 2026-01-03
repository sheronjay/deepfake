import os
import sys
import logging
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from infer.modules.vc.modules import VC

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_voice_folder(
    input_folder_path,
    model_name="mahindasiri_thero_3.pth",
    output_folder_name="voice_converted_sinhala_audio_segments",
    metadata_json_path=None,
    index_file=None,
    f0_up_key=0,
    f0_method="rmvpe",
    index_rate=0.75,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
    protect=0.33,
    output_format="wav",
):
    """
    Convert all audio files in a folder using RVC voice conversion.
    Creates a new subfolder inside the input folder and saves converted files there.
    
    Args:
        input_folder_path: Path to folder containing audio files to convert
        model_name: Name of the RVC model file (in assets/weights/)
        output_folder_name: Name of the subfolder to create for converted files
        metadata_json_path: Path to metadata JSON file to update with converted audio paths
        f0_up_key: Pitch shift in semitones (0 = no change, +12 = one octave up)
        f0_method: Pitch detection method ('harvest', 'pm', 'crepe', 'rmvpe')
        index_rate: Feature retrieval ratio (0-1)
        filter_radius: Median filtering for pitch (0-7, 3 is typical)
        resample_sr: Output sample rate (0 = use model's default)
        rms_mix_rate: Volume envelope mix ratio (0-1)
        protect: Protect voiceless consonants (0-0.5)
        output_format: Output format ('wav', 'flac', 'mp3', etc.)
    
    Returns:
        str: Path to the created output folder containing converted audio files
    """
    # Create output folder inside the input folder so converted files live next to inputs
    input_folder = Path(input_folder_path)
    output_folder_path = input_folder / output_folder_name
    output_folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output folder created: {output_folder_path}")
    
    # Initialize config
    config = Config()
    
    # Set environment variables for model paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ["weight_root"] = os.path.join(project_root, "assets", "weights")
    os.environ["index_root"] = os.path.join(project_root, "assets", "indices")
    os.environ["rmvpe_root"] = os.path.join(project_root, "assets", "rmvpe")
    
    # Initialize VC instance
    vc = VC(config)
    
    # Load the model
    logger.info(f"Loading model: {model_name}")
    vc.get_vc(model_name)

    # Resolve index file path if provided
    file_index = ""
    if index_file:
        # If a relative filename was provided, look under the assets/indices folder
        possible = Path(index_file)
        if not possible.exists():
            possible = Path(project_root) / "assets" / "indices" / index_file
        if possible.exists():
            file_index = str(possible)
        else:
            logger.warning(f"Index file not found: {index_file}")
    
    # Perform batch voice conversion
    logger.info(f"Converting audio files from: {input_folder_path}")
    for result in vc.vc_multi(
        sid=0,
        dir_path=input_folder_path,
        opt_root=str(output_folder_path),
        paths=[],
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        file_index=file_index,
        file_index2="",
        index_rate=index_rate,
        filter_radius=filter_radius,
        resample_sr=resample_sr,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
        format1=output_format,
    ):
        logger.info(result)
    
    logger.info(f"Conversion complete! Files saved in: {output_folder_path}")
    
    # Update metadata JSON file with converted audio paths
    if metadata_json_path and os.path.exists(metadata_json_path):
        logger.info(f"Updating metadata file: {metadata_json_path}")
        try:
            with open(metadata_json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Update each segment with converted audio path
            for segment in metadata:
                if 'audio' in segment:
                    # Get original filename
                    original_audio = Path(segment['audio'])
                    original_filename = original_audio.name
                    
                    # Construct converted audio path
                    converted_audio_path = output_folder_path / f"{original_filename}.{output_format}"
                    
                    # Add converted_audio field to metadata
                    segment['converted_audio'] = str(converted_audio_path)
            
            # Write updated metadata back to file
            with open(metadata_json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Metadata file updated successfully")
        except Exception as e:
            logger.error(f"Failed to update metadata file: {e}")
    
    return output_folder_path


if __name__ == "__main__":
    # Parse command-line arguments BEFORE importing Config
    if len(sys.argv) >= 3:
        metadata_json = sys.argv[1]
        model_name = sys.argv[2]
        index_file_name = sys.argv[3] if len(sys.argv) >= 4 else "assets/indices/added_IVF1281_Flat_nprobe_1_mahindasiri_thero_4_v1.index"
        
        # Remove our custom args so Config's argparse doesn't see them
        sys.argv = [sys.argv[0]]
    else:
        print("Usage: python convert_voice.py <metadata_json> <model_name> [<index_file_name>]")
        sys.exit(1)
    
    # Load metadata to extract input folder
    with open(metadata_json, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Extract the input folder from the first audio file path in metadata
    if metadata and len(metadata) > 0 and 'audio' in metadata[0]:
        first_audio_path = Path(metadata[0]['audio'])
        input_folder = str(first_audio_path.parent)
    else:
        logger.error("No audio paths found in metadata file")
        sys.exit(1)
    
    output_folder_name = "voice_converted_sinhala_audio_segments"
    
    logger.info("Starting voice conversion...")
    logger.info(f"Input folder: {input_folder}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output folder: {output_folder_name}")
    logger.info(f"Metadata JSON: {metadata_json}")
    
    # Convert all audio files in the folder
    output_path = convert_voice_folder(
        input_folder_path=input_folder,
        model_name=model_name,
        output_folder_name=output_folder_name,
        metadata_json_path=metadata_json,
        index_file=index_file_name,
        f0_up_key=0,  # No pitch change
        f0_method="rmvpe",  # or 'harvest', 'pm', 'crepe'
        index_rate=0.75,
        protect=0.33,
        output_format="wav",
    )
    
    print(f"\n✓ Conversion complete!")
    print(f"✓ Converted files saved in: {output_path}")
