中文 | [English](README-en.md)

# Step-Audio-TTS 的 ComfyUI 节点, 文本转语音, 可说话, 唱歌, RAP, 或者克隆声音.

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-02-21_05-34-25.png)

## 模型下载

下载到 `ComfyUI\models\TTS` 文件夹中.

### Huggingface
| 模型   | 链接   |
|-------|-------|
| Step-Audio-Tokenizer | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [🤗huggingface](https://huggingface.co/stepfun-ai/Step-Audio-TTS-3B) |

### Modelscope
| 模型   | 链接   |
|-------|-------|
| Step-Audio-Tokenizer | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-Tokenizer) |
| Step-Audio-TTS-3B | [modelscope](https://modelscope.cn/models/stepfun-ai/Step-Audio-TTS-3B) |

### 声音下载

https://github.com/stepfun-ai/Step-Audio/tree/main/speakers 

整个文件夹下载到 `ComfyUI\models\TTS` 文件夹中. 并重命名为 `Step-Audio-speakers`.

应包含以下结构：

```
ComfyUI\models\TTS
├── Step-Audio-Tokenizer
├── Step-Audio-speakers
├── Step-Audio-TTS-3B
```

## 支持 中文, 英文, 韩语, 日语, 四川话, 粤语等

## 致谢

本项目的部分代码来自：
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

感谢以上所有开源项目对本项目开源做出的贡献！