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

### Voices Download

https://github.com/stepfun-ai/Step-Audio/tree/main/speakers 

Download the entire folder to the `ComfyUI\models\TTS` folder. And rename it as `Step-Audio-speakers`.

Where_you_download_dir should have the following structure:

```
ComfyUI\models\TTS
├── Step-Audio-Tokenizer
├── Step-Audio-Chat
├── Step-Audio-TTS-3B
```

## Supports Chinese, English, Korean, Japanese, Sichuanese, Cantonese etc.

## Acknowledgements

Part of the code for this project comes from:
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!