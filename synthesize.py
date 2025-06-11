import re
import argparse
from string import punctuation

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

import openai
import os

# --- SECURITY NOTE: The OpenAI API key should NOT be hardcoded in public code! ---
# Please set your OpenAI API key as an environment variable (recommended) or load it from a config file.
# Example (in your shell):
#   export OPENAI_API_KEY='sk-...'
# Then in Python:
#   openai.api_key = os.getenv('OPENAI_API_KEY')
# DO NOT commit your real API key to public repositories!

# Remove the hardcoded API key for security
# openai.api_key = "sk-..."  # <-- REMOVE THIS LINE
openai.api_key = os.getenv('OPENAI_API_KEY')  # Read from environment variable
client = openai.OpenAI(api_key=openai.api_key)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

    
def gpt(text, emotion_type="surprise"):
    # Use OpenAI GPT-4o to automatically detect up to three emotion-related keywords in the input text.
    # This enables fine-grained, context-aware emotional control for TTS synthesis.
    # The detected keywords are used for local pitch/energy control and smoothing.
    messages = [
        {
            "role": "system",
            "content": f"You are a {emotion_type} keyword detection assistant. Identify up to three words that express {emotion_type} or related elements in the sentence. If less than three, fill with 'None'. Only output the words, one per line, no extra explanation."
        },
        {
            "role": "user",
            "content": f"Sentence: '{text}'\nList up to three {emotion_type} words:"
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=30,
            temperature=0.1,
        )
        if response.choices:
            response_text = response.choices[0].message.content
            keywords = [re.sub(r"^[0-9]+[.、]?\\s*", "", line).strip() for line in response_text.splitlines()]
            keywords = [k for k in keywords if k and k.lower() != "none"]
            print(keywords)
            return keywords
        else:
            return []
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return []


def preprocess_english(text, preprocess_config, emotion_type="surprise"):
    keys = gpt(text, emotion_type)
    print(keys)
    # Example: keys=["just", "more", "boring"]

    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    key_indices = []  # List to store indices of keywords found in text
    current_phoneme_index = 0  # Track the index of phonemes in the phoneme sequence

    for w in words:
        word_phones = []
        if w.lower() in lexicon:
            word_phones = lexicon[w.lower()]
        else:
            word_phones = list(filter(lambda p: p != " ", g2p(w)))

        phones += word_phones

        # Find indices for keywords
        if w.lower() in [key.lower() for key in keys]:
            start_index = current_phoneme_index
            end_index = current_phoneme_index + len(word_phones) - 1
            # Correction: Avoid the keyword interval exactly covering the last phoneme
            if end_index >= len(phones) - 2:
                end_index = len(phones) - 3
                if end_index < start_index:
                    continue  # Skip if the interval is too short
            key_indices.append((start_index, end_index))

        current_phoneme_index += len(word_phones)  # Increment the phoneme index by the length of the current word's phonemes

    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    print("Keyword Indices: {}".format(key_indices))  # Output the indices of keywords

    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence), key_indices


def preprocess_mandarin(text, preprocess_config, keys=None, emotion_type="surprise"):
    # Mandarin text preprocessing, supporting generation of key_indices for keyword intervals.
    # keys: List of keywords (pinyin or Chinese characters) for marking keyword intervals.
    # emotion_type: Emotion type (e.g., surprise, sarcasm, etc.) for automatic keyword detection.
    # Returns: (phoneme sequence, key_indices)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    # If no keys are provided, use GPT to automatically detect keywords
    if keys is None or len(keys) == 0:
        keys = gpt(text, emotion_type)

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    key_indices = []
    current_phoneme_index = 0
    for idx, p in enumerate(pinyins):
        if p in lexicon:
            word_phones = lexicon[p]
        else:
            word_phones = ["sp"]
        phones += word_phones
        # Keyword interval marking (supports matching by pinyin or original character)
        if keys is not None and (p in keys or text[idx] in keys):
            start_index = current_phoneme_index
            end_index = current_phoneme_index + len(word_phones) - 1
            key_indices.append((start_index, end_index))
        current_phoneme_index += len(word_phones)

    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    print("Keyword Indices: {}".format(key_indices))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence), key_indices


def synthesize(model, step, configs, vocoder, batchs, control_values, key_indices):
    # The synthesize function now accepts key_indices, which are passed to the model
    # for local pitch/energy/duration control and smoothing at emotion keyword intervals.
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                key_indices=key_indices
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "--keys",
        nargs='+',  # This allows the argument to accept multiple values
        default=[],
        help="List of keywords to apply pitch control"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--emotion_level",
        type=str,
        choices=["mild", "moderate", "strong"],
        default=None,
        help="Set emotion strength for keywords (mild/moderate/strong)"
    )
    parser.add_argument(
        "--emotion_type",
        type=str,
        default="surprise",
        help="Type of emotion for keyword detection (e.g., surprise, sarcasm, anger, joy)"
    )
    args = parser.parse_args()

    EMOTION_LEVEL_MAP = {
        "mild": {"pitch_control": 1.2, "energy_control": 1.2},
        "moderate": {"pitch_control": 1.5, "energy_control": 1.5},
        "strong": {"pitch_control": 1.8, "energy_control": 2.0},
    }

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        speakers = np.array([args.speaker_id])
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            sequence, key_indices = preprocess_english(args.text, preprocess_config, args.emotion_type)
            texts = np.array([sequence])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            sequence, key_indices = preprocess_mandarin(args.text, preprocess_config, args.keys, args.emotion_type)
            texts = np.array([sequence])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    if args.emotion_level is not None:
        pitch_control = EMOTION_LEVEL_MAP[args.emotion_level]["pitch_control"]
        energy_control = EMOTION_LEVEL_MAP[args.emotion_level]["energy_control"]
    else:
        pitch_control = args.pitch_control
        energy_control = args.energy_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, (pitch_control, energy_control, args.duration_control), key_indices)
    
