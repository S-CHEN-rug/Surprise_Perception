This repository extends the original FastSpeech2 with fine-grained, interval-based control for pitch and energy, emotion-driven synthesis, and a series of improvements for more natural and expressive TTS.

## Key Improvements Over Original FastSpeech2

### 1. Emotion/Keyword Interval Control
- **Support for emotion or keyword intervals**: The model can now accept `key_indices` marking intervals (e.g., emotional keywords) in the input text.
- **Local pitch/energy control**: Instead of only global scaling, pitch and energy can be dynamically controlled within specified intervals, enabling expressive, context-aware prosody.

### 2. Dynamic and Smooth Prosody Modification
- **Dynamic scaling curves**: Within each interval, pitch/energy can be modulated using smooth curves (e.g., cosine window), rather than abrupt or static changes.
- **Buffer zone smoothing**: At the boundaries of each interval, a buffer zone is used for weighted interpolation, reducing clicks and unnatural transitions.
- **Global mel-spectrogram smoothing**: 1D convolutional smoothing is applied to the mel-spectrogram before vocoder inference for further quality improvement.

### 3. Emotion Detection and Parameterization
- **GPT-based keyword detection**: The system can automatically detect up to three emotion-related keywords in the input text using OpenAI GPT-4o, supporting both English and Mandarin.
- **Emotion type and strength**: New command-line arguments (`--emotion_type`, `--emotion_level`) allow users to specify the type (e.g., surprise, sarcasm, anger, joy) and strength (mild, moderate, strong) of emotion, which are mapped to pitch/energy control values.
- **Manual keyword override**: Users can also manually specify keywords for interval control via the `--keys` argument.

### 4. Enhanced Command-Line Interface
- **New arguments**:
  - `--emotion_type`: Set the emotion type for keyword detection and prosody control.
  - `--emotion_level`: Set the strength of emotion (affecting pitch/energy scaling).
  - `--keys`: Manually specify keywords for interval-based control.
- **Backward compatibility**: All new features are optional; the system can still run in standard FastSpeech2 mode.

### 5. Code Structure and Documentation
- **Extensive English comments**: All new features and changes are clearly documented in the code for easy understanding and further development.
- **Security best practices**: Sensitive information such as API keys is no longer hardcoded; users are instructed to set their own keys via environment variables.

## Affected Files and Main Changes

- **synthesize.py**
  - Adds GPT-based emotion keyword detection and interval extraction.
  - Supports new CLI arguments for emotion type, level, and manual keywords.
  - Passes `key_indices` and emotion parameters through the synthesis pipeline.

- **tools.py**
  - Adds `smooth_mel_spectrogram` for global mel smoothing.
  - Adds `apply_transition_smooth` for buffer zone smoothing at interval boundaries.
  - Modifies `synth_samples` to support and visualize interval-based control.

- **modules.py**
  - `VarianceAdaptor` supports `key_indices` for local, dynamic pitch/energy control.
  - Implements smooth dynamic scaling and buffer smoothing for each interval.

- **fastspeech2.py**
  - The model's `forward` method now accepts and propagates `key_indices` for interval-based control.

## Usage Notes
- **API Key**: For GPT-based keyword detection, set your OpenAI API key as an environment variable (`OPENAI_API_KEY`). Do not hardcode your key in the code.
- **Compatibility**: All new features are backward compatible. If you do not use interval control or emotion arguments, the model behaves as standard FastSpeech2.

## Example Command
```bash
python synthesize.py \
  --restore_step 90000 \
  --mode single \
  --text "I just want something more exciting, not boring." \
  --preprocess_config config/preprocess.yaml \
  --model_config config/model.yaml \
  --train_config config/train.yaml \
  --emotion_type surprise \
  --emotion_level strong
```

## Reference

Some of the improvements and ideas in this repository were inspired by or referenced from the work at [Weihaohaoao/Synthesis-sarcastic-voice](https://github.com/Weihaohaoao/Synthesis-sarcastic-voice/tree/main).

Please check out their repository for related approaches and further inspiration.
---
For more details, see the code comments in each file or contact the maintainer.

# FastSpeech 2 - PyTorch Implementation

This is a PyTorch implementation of Microsoft's text-to-speech system [**FastSpeech 2: Fast and High-Quality End-to-End Text to Speech**](https://arxiv.org/abs/2006.04558v1). 
This project is based on [xcmyz's implementation](https://github.com/xcmyz/FastSpeech) of FastSpeech. Feel free to use/modify the code.

There are several versions of FastSpeech 2.
This implementation is more similar to [version 1](https://arxiv.org/abs/2006.04558v1), which uses F0 values as the pitch features.
On the other hand, pitch spectrograms extracted by continuous wavelet transform are used as the pitch features in the [later versions](https://arxiv.org/abs/2006.04558).

![](./img/model.png)

# Updates
- 2021/7/8: Release the checkpoint and audio samples of a multi-speaker English TTS model trained on LibriTTS
- 2021/2/26: Support English and Mandarin TTS
- 2021/2/26: Support multi-speaker TTS (AISHELL-3 and LibriTTS)
- 2021/2/26: Support MelGAN and HiFi-GAN vocoder

# Audio Samples
Audio samples generated by this implementation can be found [here](https://ming024.github.io/FastSpeech2/). 

# Quickstart

## Dependencies
You can install the Python dependencies with
```
pip3 install -r requirements.txt
```

## Inference

You have to download the [pretrained models](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them in ``output/ckpt/LJSpeech/``,  ``output/ckpt/AISHELL3``, or ``output/ckpt/LibriTTS/``.

For English single-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

For Mandarin multi-speaker TTS, try
```
python3 synthesize.py --text "大家好" --speaker_id SPEAKER_ID --restore_step 600000 --mode single -p config/AISHELL3/preprocess.yaml -m config/AISHELL3/model.yaml -t config/AISHELL3/train.yaml
```

For English multi-speaker TTS, run
```
python3 synthesize.py --text "YOUR_DESIRED_TEXT"  --speaker_id SPEAKER_ID --restore_step 800000 --mode single -p config/LibriTTS/preprocess.yaml -m config/LibriTTS/model.yaml -t config/LibriTTS/train.yaml
```

The generated utterances will be put in ``output/result/``.

Here is an example of synthesized mel-spectrogram of the sentence "Printing, in the only sense with which we are at present concerned, differs from most if not from all the arts and crafts represented in the Exhibition", with the English single-speaker TTS model.  
![](./img/synthesized_melspectrogram.png)

## Batch Inference
Batch inference is also supported, try

```
python3 synthesize.py --source preprocessed_data/LJSpeech/val.txt --restore_step 900000 --mode batch -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
to synthesize all utterances in ``preprocessed_data/LJSpeech/val.txt``

## Controllability
The pitch/volume/speaking rate of the synthesized utterances can be controlled by specifying the desired pitch/energy/duration ratios.
For example, one can increase the speaking rate by 20 % and decrease the volume by 20 % by

```
python3 synthesize.py --text "YOUR_DESIRED_TEXT" --restore_step 900000 --mode single -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml --duration_control 0.8 --energy_control 0.8
```

# Training

## Datasets

The supported datasets are

- [LJSpeech](https://keithito.com/LJ-Speech-Dataset/): a single-speaker English dataset consists of 13100 short audio clips of a female speaker reading passages from 7 non-fiction books, approximately 24 hours in total.
- [AISHELL-3](http://www.aishelltech.com/aishell_3): a Mandarin TTS dataset with 218 male and female speakers, roughly 85 hours in total.
- [LibriTTS](https://research.google/tools/datasets/libri-tts/): a multi-speaker English dataset containing 585 hours of speech by 2456 speakers.

We take LJSpeech as an example hereafter.

## Preprocessing
 
First, run 
```
python3 prepare_align.py config/LJSpeech/preprocess.yaml
```
for some preparations.

As described in the paper, [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/) (MFA) is used to obtain the alignments between the utterances and the phoneme sequences.
Alignments of the supported datasets are provided [here](https://drive.google.com/drive/folders/1DBRkALpPd6FL9gjHMmMEdHODmkgNIIK4?usp=sharing).
You have to unzip the files in ``preprocessed_data/LJSpeech/TextGrid/``.

After that, run the preprocessing script by
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

Alternately, you can align the corpus by yourself. 
Download the official MFA package and run
```
./montreal-forced-aligner/bin/mfa_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt english preprocessed_data/LJSpeech
```
or
```
./montreal-forced-aligner/bin/mfa_train_and_align raw_data/LJSpeech/ lexicon/librispeech-lexicon.txt preprocessed_data/LJSpeech
```

to align the corpus and then run the preprocessing script.
```
python3 preprocess.py config/LJSpeech/preprocess.yaml
```

## Training

Train your model with
```
python3 train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

The model takes less than 10k steps (less than 1 hour on my GTX1080Ti GPU) of training to generate audio samples with acceptable quality, which is much more efficient than the autoregressive models such as Tacotron2.

# TensorBoard

Use
```
tensorboard --logdir output/log/LJSpeech
```

to serve TensorBoard on your localhost.
The loss curves, synthesized mel-spectrograms, and audios are shown.

![](./img/tensorboard_loss.png)
![](./img/tensorboard_spec.png)
![](./img/tensorboard_audio.png)

# Implementation Issues

- Following [xcmyz's implementation](https://github.com/xcmyz/FastSpeech), I use an additional Tacotron-2-styled Post-Net after the decoder, which is not used in the original FastSpeech 2.
- Gradient clipping is used in the training.
- In my experience, using phoneme-level pitch and energy prediction instead of frame-level prediction results in much better prosody, and normalizing the pitch and energy features also helps. Please refer to ``config/README.md`` for more details.

Please inform me if you find any mistakes in this repo, or any useful tips to train the FastSpeech 2 model.

# References
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://arxiv.org/abs/2006.04558), Y. Ren, *et al*.
- [xcmyz's FastSpeech implementation](https://github.com/xcmyz/FastSpeech)
- [TensorSpeech's FastSpeech 2 implementation](https://github.com/TensorSpeech/TensorflowTTS)
- [rishikksh20's FastSpeech 2 implementation](https://github.com/rishikksh20/FastSpeech2)

# Citation
```
@INPROCEEDINGS{chien2021investigating,
  author={Chien, Chung-Ming and Lin, Jheng-Hao and Huang, Chien-yu and Hsu, Po-chun and Lee, Hung-yi},
  booktitle={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Investigating on Incorporating Pretrained and Learnable Speaker Representations for Multi-Speaker Multi-Style Text-to-Speech}, 
  year={2021},
  volume={},
  number={},
  pages={8588-8592},
  doi={10.1109/ICASSP39728.2021.9413880}}
```
