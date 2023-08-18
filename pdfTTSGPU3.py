

# Using parallel threads,
# - Progress order correct
# - Audio mp3 order correct
# - Includes text processing to rectify misspoken words, etc.
# - Includes text summarization
# - Includes argpars

import argparse
import requests
from lxml import etree
import re

import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

import torch
from datasets import load_dataset
import numpy as np
import io
from pydub.playback import play
import subprocess
import time
from nltk.tokenize import sent_tokenize
from io import BytesIO
from pydub import AudioSegment
import os
import nltk
import inflect

from datetime import datetime
speedup_factor = 1.28

def initialize():
    print("INIT")
    nltk.download('punkt')


from tempfile import NamedTemporaryFile
from concurrent.futures import ThreadPoolExecutor, as_completed

def get_max_batch_size(memory_requirement_per_sample, safety_factor=0.9):
    total_memory = torch.cuda.get_device_properties(0).total_memory
    free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    available_memory = safety_factor * free_memory
    max_batch_size = int(available_memory / memory_requirement_per_sample)
    return max_batch_size

def start_grobid():
    # grobid_path = "/Users/leland/grobid "  # Replace this with the actual path
    command = "./gradlew run"
    subprocess.Popen(command, shell=True)

def is_grobid_running():
    try:
        response = requests.get("http://localhost:8070/api")
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def wait_for_grobid():
    url = "http://localhost:8070/api/version"  # This endpoint should respond when GROBID is ready
    max_attempts = 30
    delay_seconds = 10

    for attempt in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("[✓] GROBID is ready.")
                return
        except requests.ConnectionError:
            pass

        print(f"[!] GROBID is not ready, waiting for {delay_seconds} seconds before retrying...[{attempt}/{max_attempts}]")
        time.sleep(delay_seconds)

    print("[!] GROBID did not become ready in time. Please check your GROBID installation.")


def process_pdf(pdf_path, outfile_path):
    # Send the PDF file to GROBID for processing
    print('[✓] Converting PDF to .txt & parsing data using GORBID')
    with open(pdf_path, 'rb') as pdf_file:
        response = requests.post('http://localhost:8070/api/processFulltextDocument', files={'input': pdf_file})
        if response.status_code != 200:
            print("An error occurred while processing the PDF.")
            return

        # Parse the XML response
        parsed_xml = response.text
        root = etree.fromstring(bytes(parsed_xml, encoding='utf-8'))

        # Regular expression to match readable characters and exclude specific unwanted character
        pattern = re.compile(r'[^\x00-\x1F\x7F-\x9F■]+')

        # Open the output file for writing
        with open(outfile_path, 'w', encoding='utf-8') as outfile:
            # Iterate through the elements and write titles and paragraphs in order
            for elem in root.xpath('//tei:body//tei:head|//tei:body//tei:p', namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}):
                text = ' '.join(elem.xpath('.//text()'))
                # Remove any non-readable characters
                readable_text = ''.join(pattern.findall(text))
                # Check if the element is a head (possibly chapter title) or paragraph
                if elem.tag.endswith('head'):
                    # Add formatting for a chapter title
                    outfile.write(f'\n\n[Section] {readable_text}\n\n')
                else:
                    # Add formatting for a paragraph
                    outfile.write(readable_text + '\n')
    print("[✓] .txt file generated. Moving to text-editing step")

    processed_txt_file_script = args.vscript #name of output file for audio script

    # If summarization is requested, summarize the content and write to a new file
    if args.summarize:
        print("[!] Summarizing text file. Please wait")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn').to(device)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        with open(outfile_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            summarized_content = ""
            paragraph = ""
            paragraphs_to_summarize = []

            for line in lines:
                if not line.startswith("[Section]"):  # Exclude lines starting with "[Section]"
                    paragraph += line
                    if line.strip() == "":
                        if paragraph.strip():  # Check if paragraph is not empty
                            paragraphs_to_summarize.append(paragraph)
                        paragraph = ""  # Reset paragraph

            total_paragraphs = len(paragraphs_to_summarize)
            completed_paragraphs = 0

            def parallel_summarize_with_index(args):
                index, text = args
                summary = summarize_text(text, model, tokenizer, device)  # Include device in the call
                return index, summary

            with ThreadPoolExecutor() as executor:
                for index, summary in executor.map(parallel_summarize_with_index, enumerate(paragraphs_to_summarize)):
                    if summary.strip() != "CNN.com will feature iReporter photos in a weekly Travel Snapshots gallery. Please submit your best shots for next week. Visit CNN.com/Travel next Wednesday for a new gallery of snapshots.":
                        summarized_content += summary + '\n\n'
                    completed_paragraphs += 1
                    progress = completed_paragraphs / total_paragraphs * 100
                    print(f"-> Summarization progress: {progress:.2f}% ({completed_paragraphs}/{total_paragraphs})",
                          flush=True)

            if paragraph.strip() != "":
                summarized_content += summarize_text(paragraph, model, tokenizer,
                                                     device) + '\n'  # Include device in the call

            # summarized_output_file = "summarized_output.txt"  # Change to your desired path

            summarized_output_txt = f"{base_name}_smrzd_output.txt"

            with open(summarized_output_txt, 'w', encoding='utf-8') as summary_file:
                summary_file.write(summarized_content)
            print(f"[✓] Text successfully summarized and written to {summarized_output_txt}")

    if args.audio_summarize:
        # Preprocess the text and create the altOutput.txt file
        print(f"[✓] Using {summarized_output_txt} for audio script")
        preprocess_text_file(summarized_output_txt, processed_txt_file_script)

    else:
        print(f"[✓] Using {outfile_path} for audio script")
        preprocess_text_file(outfile_path, processed_txt_file_script)


    # audio_summarize
    # Read the processed text file
    with open(processed_txt_file_script, 'r', encoding='utf-8') as file:
        content = file.read()

    # Read the processed text file and count the total number of sections
    section_total = 0

    with open(processed_txt_file_script, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("[Section]"):
                section_total += 1

    # Process the sections and speak the text
    section_content = ""
    section_index = 0

    with open(processed_txt_file_script, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("[Section]"):
                # When we reach a new section, speak the previous one (if any)
                if section_content.strip():
                    print(f"[✓] Speaking Section {section_index}/{section_total}")
                    speak_text(section_content, section_index)  # Adjust the function signature if needed
                    section_content = ""
                    section_index += 1
            # Accumulate the content for the current section
            section_content += line

    # Speak the last section if there is any content left
    if section_content.strip():
        print(f"[✓] Speaking Section {section_index}/{section_total}")
        speak_text(section_content, section_index)

    # combine_sections(section_total)


def summarize_text(text, model, tokenizer, device):
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True).to(device)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=100, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def convert_numbers_to_words(text):
    p = inflect.engine()
    words = text.split()
    for i in range(len(words)):
        if words[i].isdigit():
            words[i] = p.number_to_words(words[i])
    return " ".join(words)


def preprocess_text(text):
    pronunciation_dict = {
        'viruses': 'virus\'es',
        'GPU': 'Gee Pee You',
        'CPU': 'Cee Pee You',
        'PC': 'Pee Cee',
        'API': 'A Pee Eye',
        'VPU': 'Vee Pee You',
        'v/s': 'Versus',
        'GPS': 'Gee Pee Ess',
        'AES': 'A Ee Ess',
        'DES': 'De Ee Ess',
        '2D': 'Two Dee',
        '3D': 'Three Dee',
        '4D': 'Four Dee',
        'OS': 'Oh Ess',
        'AMD': 'Ayy Em Dee',
        'Nvidia': "Envidia",
        '+': 'plus',
        '=': 'equals',
        '&': 'and',

    }

    def process_word(word):

        p = inflect.engine()

        # # Remove non-speakable characters (only keep alphanumeric and some punctuation)
        # word = re.sub(r'[^\w\s\'-]', '', word)

        # Remove '(' and ')' brackets
        word = word.replace("(", "").replace(")", "")

        # Handle words ending with 's' separately
        if word.endswith('s') and word[:-1] in pronunciation_dict:
            return pronunciation_dict[word[:-1]] + 's'

        # Check the pronunciation dictionary
        if word in pronunciation_dict:
            return pronunciation_dict[word]

        # Convert numbers to words
        if word.isdigit():
            return p.number_to_words(word)

        # Remove hyphens
        if '-' in word:
            word = word.replace("-", " ")

        return word

    # Tokenize text using regex to separate words, whitespace, punctuation, and newlines
    words = re.findall(r'\b\w+\b|\s+|[^\w\s]|\\n', text)
    processed_words = [process_word(word) for word in words]
    processed_text = ''.join(processed_words)

    return processed_text


def preprocess_text_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    processed_content = ''
    skip_next_lines = False
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if '[Section]' in line:
            # Check if next lines are empty
            empty_lines_count = 0
            j = i + 1
            while j < len(lines) and lines[j].strip() == '':
                empty_lines_count += 1
                j += 1

            # If the next line after the section is not empty, or if there's more than one empty line,
            # process the section
            # Run the words through the processing stage to fix pronunciations
            if empty_lines_count <= 1:
                processed_content += preprocess_text(line) + '\n'
            else:
                # Skip next empty lines
                skip_next_lines = True
                i += empty_lines_count

        elif not skip_next_lines:
            processed_content += preprocess_text(line) + '\n'

        else:
            skip_next_lines = False

        i += 1

    # Write the processed content to a new file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(processed_content)


def save_progress(progress_file, progress):
    with open(progress_file, 'w') as file:
        file.write(str(progress))


# def combine_sections(total_sections):
#     print("[!] Combining audio files")
#     audioSegmenet = AudioSegment.empty()
#     temp_audio_path = "temp_audio"
#
#     for section_index in range(total_sections):
#         section_audio_path = os.path.join(temp_audio_path,
#                                           f"temp_audio_{section_index}.mp3")  # Assuming files are in WAV format
#         section_audio = AudioSegment.from_wav(section_audio_path)  # Change to appropriate method if a different format
#         audioSegmenet += section_audio
#
#     # Save the combined audio to the final MP3 file
#     final_output_mp3 = f"{base_name}_final_output.mp3"
#     audioSegmenet.export(final_output_mp3, format="mp3")
#     print(f"[✓] Final combined audio saved to {final_output_mp3}")


def process_chunk(chunks, processor, model, vocoder, speaker_embeddings, device):
    sounds = []
    for chunk in chunks:
        inputs = processor(text=chunk, return_tensors="pt").to(device)
        with torch.no_grad():  # Deactivate gradients for inference
            speech = model.generate_speech(inputs["input_ids"].to(device), speaker_embeddings.to(device), vocoder=vocoder)
        speech = speech.cpu().numpy()  # Move the tensor to CPU before converting to numpy
        speech = np.squeeze(speech)
        with NamedTemporaryFile(suffix=".wav") as tmp_wav:
            sf.write(tmp_wav.name, speech, 16000)
            sound = AudioSegment.from_wav(tmp_wav.name)
        sounds.append(sound)
    return sounds


def compile_audio(range_str=None, clean_up=False):
    temp_audio_path = "temp_audio/"
    compiled_audio_path = "compiled_audio/"
    progress_file_path = "audioCompProgress.txt"

    # Ensure the output folder exists
    os.makedirs(compiled_audio_path, exist_ok=True)

    # Determine the final output file name
    if range_str is not None:
        final_output_mp3 = os.path.join(compiled_audio_path, f"compiledAudio_{range_str}.mp3")
    else:
        final_output_mp3 = os.path.join(compiled_audio_path, "compiledAudio.mp3")

    # If resuming, open the existing final file
    if os.path.exists(final_output_mp3):
        combined_audio = AudioSegment.from_mp3(final_output_mp3)
    else:
        combined_audio = AudioSegment.empty()

    # Define the range to compile if provided
    start_index = 0
    end_index = None
    if range_str:
        start_index, end_index = map(int, range_str.split("-"))
        end_index += 1  # Inclusive of the end range

    # Check if there's a progress file to resume
    if os.path.exists(progress_file_path):
        with open(progress_file_path, 'r') as progress_file:
            start_index = int(progress_file.readline().strip())

    try:
        audio_files = sorted(os.listdir(temp_audio_path))[start_index:end_index]
        total_files = len(audio_files)

        for index, audio_file in enumerate(audio_files, start=start_index):
            audio_path = os.path.join(temp_audio_path, audio_file)
            audio_segment = AudioSegment.from_mp3(audio_path)

            combined_audio += audio_segment

            # Print the progress
            progress_percent = (index + 1) / total_files * 100
            print(f"-> Compilation progress: {progress_percent:.2f}% ({index + 1}/{total_files})")

            # Save the current progress
            with open(progress_file_path, 'w') as progress_file:
                progress_file.write(str(index + 1))

            # Clean up individual audio files if flag is set
            if clean_up:
                os.remove(audio_path)

        # Export the final combined audio
        combined_audio.export(final_output_mp3, format="mp3")
        print(f"[✓] Final combined audio saved to [{final_output_mp3}]")

        # Delete the progress file if successful
        os.remove(progress_file_path)

    except Exception as e:
        print(f"[X] An error occurred during compilation: {str(e)}. Saving progress so far.")
        combined_audio.export(final_output_mp3, format="mp3")


def speak_text(text, section_index):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

    temp_audio_path = "temp_audio"
    os.makedirs(temp_audio_path, exist_ok=True)


    # sections = text.split('[Section]')
    max_chunk_length = 512  # Maximum length of tokens
    total_chunks = sum(
        (len(processor.tokenizer(sentence).input_ids) // max_chunk_length + 1) for sentence in sent_tokenize(text))

    BATCH_SIZE = 64

    print("[✓] Starting TTS generation...")
    audioSegmenet = AudioSegment.empty()
    batch_chunks = []
    progress_counter = 0

    sentences = sent_tokenize(text)
    for sentence in sentences:
        words = sentence.split()
        chunk = ""
        for word in words:
            temp_chunk = chunk + " " + word
            if len(processor.tokenizer(temp_chunk).input_ids) <= max_chunk_length:
                chunk = temp_chunk
            else:
                batch_chunks.append(chunk.strip())
                chunk = word

            if len(batch_chunks) == BATCH_SIZE:
                results = process_chunk(batch_chunks, processor, model, vocoder, speaker_embeddings, device)
                for result in results:
                    audioSegmenet += result
                    progress_counter += 1
                    progress = progress_counter / total_chunks * 100
                    print(f"-> Progress: {progress:.2f}% ({progress_counter}/{total_chunks})", flush=True)
                batch_chunks = []
        if chunk:
            batch_chunks.append(chunk.strip())

    # Process remaining chunks
    if batch_chunks:
        results = process_chunk(batch_chunks, processor, model, vocoder, speaker_embeddings, device)
        for result in results:
            audioSegmenet += result

    # Apply the speed-up
    audioSegmenet = audioSegmenet.speedup(playback_speed=speedup_factor)

    # Save the audio for this section to a separate file
    # output_mp3 = f"{base_name}_section_{section_index}.mp3"
    output_mp3 = os.path.join(temp_audio_path, f"{base_name}_section_{section_index}.mp3")
    audioSegmenet.export(output_mp3, format="mp3")
    # temp_audio_file = os.path.join(temp_audio_path, f"temp_audio_{section_index}.wav")

    print(f"[✓] TTS generation for section {section_index} completed!")


if __name__ == "__main__":

    # ./gradlew run
    # python3 pdfTTS8.py --input test.pdf --output outfile.txt --summarize
    parser = argparse.ArgumentParser(description="Text processing and TTS synthesis.")
    parser.add_argument('--input', required=True, help="Path to the input text file.")
    parser.add_argument('--output', default="outfile.txt", help="Path to the output text file.")
    parser.add_argument('--vscript', default="Vscript.txt", help="Path to the audio script text file.")
    parser.add_argument('--summarize', action="store_true", help="Enable text summarization.")
    parser.add_argument('--audio_summarize', action="store_true", help="Enable Audio summarization.")
    parser.add_argument('--start_gorbid', action="store_true", help="Auto start GORBID.")
    parser.add_argument('--overwrite', action="store_true", help="Ignore any past progress for TTS")
    parser.add_argument('--clean_up', action='store_true', help='Clean up original files after compilation')

    args = parser.parse_args()
    initialize()

    # Check if GORBID is already running
    if is_grobid_running():
        print("[✓] GORBID is already running")
    else:
        # Start GORBID if specified
        if args.start_gorbid:
            print("[✓] Starting GORBID")
            start_grobid()
        else:
            print("[!] You may need to manually start GORBID: ./gradlew run")

    # Wait for GORBID to be ready before continuing
    wait_for_grobid()

    # pdf_path = args.input
    input_file_path = args.input  # Assuming args.input contains the input file path
    base_name, _ = os.path.splitext(os.path.basename(input_file_path))
    base_name = base_name.replace(' ', '_')

    outfile_txt = f"{base_name}_outfile.txt"
    # summarized_output_txt = f"{base_name}_smrzd_output.txt"
    # output_mp3 = f"{base_name}_output.mp3"

    print("[!] File being converted: " + input_file_path)

    # Process the PDF
    process_pdf(input_file_path, outfile_txt)

    try:
        compile_audio(None, args.clean_up)

    except Exception as save_e:
        print(f"Failed to compile audio samples: {save_e}")

