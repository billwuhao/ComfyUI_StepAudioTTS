[中文](README.md) | English

# A Text To Speech node using Step-Audio-TTS in ComfyUI. Can speak, rap, sing, or clone voice.



## 3. Model Download

Download to the `ComfyUI\models\TTS` folder

### 3.1 Huggingface
| Models   | Links   |
|-------|-------|
| Step-Audio-Tokenizer | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B) |

### 3.2 Modelscope
| Models   | Links   |
|-------|-------|
| Step-Audio-Tokenizer | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-TTS-3B) |

### Voices Download

https://github.com/stepfun-ai/Step-Audio/tree/main/speakers 

Download the entire folder to the `ComfyUI\models\TTS` folder. And rename it as `Step-Audio-speakers`.

Where_you_download_dir should have the following structure:
```
where_you_download_dir
├── Step-Audio-Tokenizer
├── Step-Audio-Chat
├── Step-Audio-TTS-3B
```

## Supports Chinese, English, Korean, Japanese, Sichuanese, Cantonese etc.

## 8. Acknowledgements

Part of the code for this project comes from:
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!