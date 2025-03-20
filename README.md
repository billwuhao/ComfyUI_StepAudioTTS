[中文](README-CN.md) | English

# A Text To Speech node using Step-Audio-TTS in ComfyUI. Can speak, rap, sing, or clone voice.

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-02-21_05-34-25.png)

## Update

[2025-03-21] ⚒️: Completely refactored the code, added more tunable parameters, and `max_length` can be adjusted according to the text length.  Optional `unload_model` to choose whether to unload the model to accelerate inference speed.

[2025-03-07]⚒️: Custom speakers can be defined directly in `ComfyUI\models\TTS\Step-Audio-speakers\speakers_info.json` without the need for input in the node.

Move the `Step-Audio-speakers` folder from this repository to the `ComfyUI\models\TTS` folder. The structure is as follows:

```
ComfyUI\models\TTS
├── Step-Audio-Tokenizer
├── Step-Audio-speakers
├── Step-Audio-TTS-3B
```

You can then freely customize speakers under the `ComfyUI\models\TTS\Step-Audio-speakers` folder for use. Ensure that the speaker name configuration matches exactly:

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-30-51.png)

[2025-03-06]⚒️: New recording node `MW Audio Recorder` can be used to record audio with a microphone, and the progress bar displays the recording progress:

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-03-06_21-29-09.png)

| 参数名/Parameter     | 作用描述/Description                                                                 | 范围/Range                     | 注意事項/Notes                                                                 |
|---------------------|------------------------------------------------------------------------------------|--------------------------------|------------------------------------------------------------------------------|
| **trigger**         | 录音触发开关 - 设为True开始录音<br>Recording trigger - Set to True to start recording | Boolean (True/False)           | 需要从False切到True才能触发<br>Requires changing from False to True to activate          |
| **record_sec**      | 主录音时长（秒）<br>Main recording duration (seconds)                               | 1-60 (整数/integer)             | 实际时长<br>Actual duration            |
| **n_fft**           | FFT窗口大小（影响频率分辨率）<br>FFT window size (affects frequency resolution)      | 512,1024,...,4096 (512倍数/multiplies) | 值越大频率分辨率越高<br>Higher values give better frequency resolution                   |
| **sensitivity**     | 降噪灵敏度（值越高越激进）<br>Noise reduction sensitivity (higher=more aggressive)   | 0.5-3.0 (步长0.1/step 0.1)      | 1.2=标准办公室环境<br>1.2=standard office environment                                   |
| **smooth**          | 时频平滑系数（值越高越自然）<br>Time-frequency smoothing (higher=more natural)       | 1,3,5,7,9,11 (奇数/odd numbers) | 建议语音：5，音乐：7<br>Recommended: 5 for speech, 7 for music                          |
| **sample_rate**     | 采样率（影响音质与文件大小）<br>Sampling rate (affects quality & size)               | 16000/44100/48000 Hz           | 44100=CD音质<br>44100=CD quality                                                         |

[2025-03-02]⚒️: Add experimental `custom_mark`, surrounding with "()", for example `(温柔)(东北话)`, it may have an effect.

[2025-02-25]⚒️: Support custom speaker `custom_stpeaker`. 

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_StepAudioTTS.git
cd ComfyUI_StepAudioTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

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


## Supports Chinese, English, Korean, Japanese, Sichuanese, Cantonese etc.

## Acknowledgements

Part of the code for this project comes from:
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

Thank you to all the open-source projects for their contributions to this project!