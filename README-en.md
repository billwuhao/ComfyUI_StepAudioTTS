[中文](README.md) | English

# A Text To Speech node using Step-Audio-TTS in ComfyUI. Can speak, rap, sing, or clone voice.

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-02-21_05-34-25.png)

assets/2025-02-21_05-34-25.png

## Model Download

Download to the `ComfyUI\models\TTS` folder

### Huggingface
| Models   | Links   |
|-------|-------|
| Step-Audio-Tokenizer | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B) |

### Modelscope
| Models   | Links   |
|-------|-------|
| Step-Audio-Tokenizer | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-TTS-3B) |

Where_you_download_dir should have the following structure:
```
ComfyUI\models\TTS
├── Step-Audio-Tokenizer
├── Step-Audio-TTS-3B
```

### Welcome to contribute more voices

The audio file is named as `{Speaker}_prompt_.WAV`, For example, `明文_prompt.WAV`. I will add them to the code. Thus, there is no need for cloning.

The currently supported voices are in the `Step-Audio-speakers` folder. Welcome to PR more voices.


## Supports Chinese, English, Korean, Japanese, Sichuanese, Cantonese etc.

## Acknowledgements

Part of the code for this project comes from:
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!