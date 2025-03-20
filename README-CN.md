中文 | [English](README.md)

# Step-Audio-TTS 的 ComfyUI 节点, 文本转语音, 可说话, 唱歌, RAP, 或者克隆声音.

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-02-21_05-34-25.png)

## 更新

[2025-03-21]⚒️: 完全重构代码, 增加更多可调参数, 可根据文本长短调节 max_length. 可选 unload_model, 是否卸载模型, 加快推理速度.

[2025-03-07]⚒️: 自定义说话者直接在 `ComfyUI\models\TTS\Step-Audio-speakers\speakers_info.json` 中定义, 无需节点中输入. 

请将本仓库中 `Step-Audio-speakers` 文件夹移动到 `ComfyUI\models\TTS` 文件夹中, 结构如下:

```
ComfyUI\models\TTS
├── Step-Audio-Tokenizer
├── Step-Audio-speakers
├── Step-Audio-TTS-3B
```

然后就可在 `ComfyUI\models\TTS\Step-Audio-speakers` 文件夹下随意自定义说话者即可使用. 注意说话者名称配置一定要一致:

![](https://github.com/billwuhao/ComfyUI_SparkTTS/blob/master/images/2025-03-07_03-30-51.png)

[2025-03-06]⚒️: 新增录音节点 `MW Audio Recorder` 可用麦克风录制音频, 进度条显示录制进度:

![](https://github.com/billwuhao/ComfyUI_StepAudioTTS/blob/master/assets/2025-03-06_21-29-09.png)

| 参数名/Parameter     | 作用描述/Description                                                                 | 范围/Range                     | 注意事項/Notes                                                                 |
|---------------------|------------------------------------------------------------------------------------|--------------------------------|------------------------------------------------------------------------------|
| **trigger**         | 录音触发开关 - 设为True开始录音<br>Recording trigger - Set to True to start recording | Boolean (True/False)           | 需要从False切到True才能触发<br>Requires changing from False to True to activate          |
| **record_sec**      | 主录音时长（秒）<br>Main recording duration (seconds)                               | 1-60 (整数/integer)             | 实际时长<br>Actual duration            |
| **n_fft**           | FFT窗口大小（影响频率分辨率）<br>FFT window size (affects frequency resolution)      | 512,1024,...,4096 (512倍数/multiplies) | 值越大频率分辨率越高<br>Higher values give better frequency resolution                   |
| **sensitivity**     | 降噪灵敏度（值越高越激进）<br>Noise reduction sensitivity (higher=more aggressive)   | 0.5-3.0 (步长0.1/step 0.1)      | 1.2=标准办公室环境<br>1.2=standard office environment                                   |
| **smooth**          | 时频平滑系数（值越高越自然）<br>Time-frequency smoothing (higher=more natural)       | 1,3,5,7,9,11 (奇数/odd numbers) | 建议语音：5，音乐：7<br>Recommended: 5 for speech, 7 for music                          |
| **sample_rate**     | 采样率（影响音质与文件大小）<br>Sampling rate (affects quality & size)               | 16000/44100/48000 Hz           | 44100=CD音质<br>44100=CD quality                                                         |

[2025-03-02]⚒️: 增加实验性的 `custom_mark`, 用 "()" 包围例如 `(温柔)(东北话)`, 它可能会有效.

[2025-02-25]⚒️: 支持自定义说话者 `custom_speaker`. 

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_StepAudioTTS.git
cd ComfyUI_StepAudioTTS
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

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

## 支持 中文, 英文, 韩语, 日语, 四川话, 粤语等

## 致谢

本项目的部分代码来自：
* [Step-Audio](https://github.com/stepfun-ai/Step-Audio)
* [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
* [transformers](https://github.com/huggingface/transformers)
* [FunASR](https://github.com/modelscope/FunASR)

感谢以上所有开源项目对本项目开源做出的贡献！