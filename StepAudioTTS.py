import torchaudio
import folder_paths
import os
import hashlib
import re
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

from tokenizer import StepAudioTokenizer
from cosyvoice.cli.cosyvoice import CosyVoice 

node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(node_dir))
model_path = os.path.join(comfy_path, "models/TTS")
encoder_model_path = os.path.join(model_path, "Step-Audio-Tokenizer")
tts_model_path = os.path.join(model_path, "Step-Audio-TTS-3B")
speaker_path = os.path.join(model_path, "Step-Audio-speakers")

class RepetitionAwareLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        window_size = 10
        threshold = 0.1

        window = input_ids[:, -window_size:]
        if window.shape[1] < window_size:
            return scores

        last_tokens = window[:, -1].unsqueeze(-1)
        repeat_counts = (window == last_tokens).sum(dim=1)
        repeat_ratios = repeat_counts.float() / window_size

        mask = repeat_ratios > threshold
        scores[mask, last_tokens[mask].squeeze(-1)] = float("-inf")
        return scores


class StepAudioTTS:
    def __init__(
        self,
        model_path,
        encoder,
    ):
        self.model_path = model_path
        self._llm = None  # 初始化为 None，表示尚未加载
        self._autotokenizer = None
        self._common_cosy_model = None
        self._music_cosy_model = None

        self.encoder = encoder
        self.sys_prompt_dict = {
            "sys_prompt_for_rap": "作为一名优秀的RAP歌手, 卓越的声优演员，请用同样的音色，用RAP方式将文本内容大声说唱出来。",
            "sys_prompt_for_vocal": "作为一名优秀的歌唱家, 歌星, 卓越的声优演员，请用同样的音色，用哼唱的方式将文本内容大声唱出来。",
            "sys_prompt_for_spk": '作为一名卓越的声优演员或歌手，你的任务是根据文本中（）或()括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，以丰富细腻的情感和自然顺畅的语调，用同样的音色，来朗读或哼唱文本。\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n- "广东话"\n\n# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n- "快速1"\n- "快速2"\n\n请在朗读或哼唱时，使用[{}]的声音，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，如果没有()或（）括号，则根据文本语义内容恰到好处地演绎。',
        }


    @property
    def llm(self):
        if self._llm is None:
            self._llm = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        ) # 初始化模型
        return self._llm
    
    @property
    def autotokenizer(self):
        if self._autotokenizer is None:
            self._autotokenizer = AutoTokenizer.from_pretrained(
            self.model_path, 
            trust_remote_code=True
        ) # 初始化模型
        return self._autotokenizer

    @property
    def common_cosy_model(self):
        if self._common_cosy_model is None:
            self._common_cosy_model = CosyVoice(os.path.join(tts_model_path, "CosyVoice-300M-25Hz")) # 初始化模型
        return self._common_cosy_model
    
    @property
    def music_cosy_model(self):
        if self._music_cosy_model is None:
            self._music_cosy_model = CosyVoice(os.path.join(tts_model_path, "CosyVoice-300M-25Hz-Music"))
        return self._music_cosy_model

    def __call__(self, text: str, prompt_speaker: str, clone_dict: dict | None = None):
        if clone_dict:
            clone_prompt_code, clone_prompt_token, clone_prompt_token_len, clone_speech_feat, clone_speech_feat_len, clone_speech_embedding = (
                self.preprocess_prompt_wav(clone_dict['audio'])
            )
            prompt_speaker_info =  {
                "prompt_text": clone_dict['prompt_text'],
                "prompt_code": clone_prompt_code,
                "cosy_speech_feat": clone_speech_feat.to(torch.bfloat16),
                "cosy_speech_feat_len": clone_speech_feat_len,
                "cosy_speech_embedding": clone_speech_embedding.to(torch.bfloat16),
                "cosy_prompt_token": clone_prompt_token,
                "cosy_prompt_token_len": clone_prompt_token_len,
            }
            
            clone_speaker = clone_dict['speaker']
            # print(clone_speaker, " 内置文本: ", prompt_speaker_info["prompt_text"], end="\n\n")

        else:
            with open(f"{speaker_path}/speakers_info.json", "r") as f:
                speakers_info = json.load(f)

            for speaker_id, prompt_text in speakers_info.items():
                if speaker_id == prompt_speaker:
                    prompt_wav_path = f"{speaker_path}/{speaker_id}_prompt.wav"
                    waveform, sample_rate = torchaudio.load(prompt_wav_path)
                    audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                    prompt_code, prompt_token, prompt_token_len, speech_feat, speech_feat_len, speech_embedding = (
                        self.preprocess_prompt_wav(audio)
                    )
                    prompt_speaker_info = {
                        "prompt_text": prompt_text,
                        "prompt_code": prompt_code,
                        "cosy_speech_feat": speech_feat.to(torch.bfloat16),
                        "cosy_speech_feat_len": speech_feat_len,
                        "cosy_speech_embedding": speech_embedding.to(torch.bfloat16),
                        "cosy_prompt_token": prompt_token,
                        "cosy_prompt_token_len": prompt_token_len,
                    }
                    # print(prompt_speaker, " 内置文本: ", prompt_speaker_info["prompt_text"], end="\n\n")
                    break

                elif prompt_speaker not in speakers_info.keys():
                    raise ValueError("There is no such speaker") 

        instruction_name = self.detect_instruction_name(text)
        
        if "RAP" or "哼唱" in instruction_name:
            cosy_model = self.music_cosy_model
        else:
            cosy_model = self.common_cosy_model
        # print("指定文本: ", text, "的说话者是: ", prompt_speaker, end="\n\n")
        token_ids = self.tokenize(
            text,
            prompt_speaker_info["prompt_text"],
            prompt_speaker,
            prompt_speaker_info["prompt_code"],
        )
        output_ids = self.llm.generate(
            torch.tensor([token_ids]).to(torch.long).to("cuda"),
            max_length=8192,
            temperature=0.7,
            do_sample=True,
            logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
        )
        output_ids = output_ids[:, len(token_ids) : -1]  # skip eos token
        return (
            cosy_model.token_to_wav_offline(
                output_ids - 65536,
                prompt_speaker_info["cosy_speech_feat"].to(torch.bfloat16),
                prompt_speaker_info["cosy_speech_feat_len"],
                prompt_speaker_info["cosy_prompt_token"],
                prompt_speaker_info["cosy_prompt_token_len"],
                prompt_speaker_info["cosy_speech_embedding"].to(torch.bfloat16),
            ),
            22050,
        )

    def detect_instruction_name(self, text):
        instruction_names = []
        pattern = r"\(.*?\)|（.*?）"
        matches = re.findall(pattern, text) 
        if matches:
            instruction_names = [i.strip("() （）") for i in matches if i.strip("() （）")]
        return instruction_names

    def tokenize(
        self, text: str, prompt_text: str, prompt_speaker: str, prompt_code: list
    ):
        rap_or_vocal = self.detect_instruction_name(text)

        if len(rap_or_vocal) == 1 and rap_or_vocal[0] == "哼唱":
            prompt = self.sys_prompt_dict["sys_prompt_for_vocal"]
            # print("哼唱系统消息: ", prompt, end="\n\n")
        elif len(rap_or_vocal) == 1 and rap_or_vocal[0] == "RAP":
            prompt = self.sys_prompt_dict["sys_prompt_for_rap"]
            # print("RAP系统消息: ", prompt, end="\n\n")
        else:
            prompt = self.sys_prompt_dict["sys_prompt_for_spk"].format(prompt_speaker)
            # print("其他系统消息: ", prompt, end="\n\n")
        
        sys_tokens = self.autotokenizer.encode(f"system\n{prompt}")

        history = [1]
        history.extend([4] + sys_tokens + [3])

        _prefix_tokens = self.autotokenizer.encode("\n")
        prompt_token_encode = self.autotokenizer.encode("\n" + prompt_text)
        prompt_tokens = prompt_token_encode[len(_prefix_tokens) :]

        target_token_encode = self.autotokenizer.encode("\n" + text)
        target_tokens = target_token_encode[len(_prefix_tokens) :]

        qrole_toks = self.autotokenizer.encode("human\n")
        arole_toks = self.autotokenizer.encode("assistant\n")

        history.extend(
            [4]
            + qrole_toks
            + prompt_tokens
            + [3]
            + [4]
            + arole_toks
            + prompt_code
            + [3]
            + [4]
            + qrole_toks
            + target_tokens
            + [3]
            + [4]
            + arole_toks
        )
        return history

    def preprocess_prompt_wav(self, audio : str):
        prompt_wav = audio["waveform"].squeeze(0)
        prompt_wav_sr = audio["sample_rate"]

        if prompt_wav.shape[0] > 1:
            prompt_wav = prompt_wav.mean(dim=0, keepdim=True)  # 将多通道音频转换为单通道
        prompt_wav_16k = torchaudio.transforms.Resample(
            orig_freq=prompt_wav_sr, new_freq=16000
        )(prompt_wav)
        prompt_wav_22k = torchaudio.transforms.Resample(
            orig_freq=prompt_wav_sr, new_freq=22050
        )(prompt_wav)

        speech_feat, speech_feat_len = (
            self.common_cosy_model.frontend._extract_speech_feat(prompt_wav_22k)
        )
        speech_embedding = self.common_cosy_model.frontend._extract_spk_embedding(
            prompt_wav_16k
        )

        prompt_code, _, _ = self.encoder.wav2token(prompt_wav, prompt_wav_sr)
        prompt_token = torch.tensor([prompt_code], dtype=torch.long) - 65536
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.long)

        return (
            prompt_code,
            prompt_token,
            prompt_token_len,
            speech_feat,
            speech_feat_len,
            speech_embedding,
        )
    

encoder = StepAudioTokenizer(encoder_model_path)
tts_engine = StepAudioTTS(tts_model_path, encoder)

# 选项列表
emotion_options = ["高兴1", "高兴2", "生气1", "生气2", "悲伤1", "撒娇1", "None"]
language_options = ["中文", "英文", "韩语", "日语", "四川话", "粤语", "None"]
speed_options = ["慢速1", "慢速2", "快速1", "快速2", "None"]
speaker_options = ["婷婷", "婷婷RAP", "婷婷哼唱", "明文"]
express_options = ["RAP", "哼唱", "None"]

def gen_text(*args):
    formatted_args = []
    for arg in args:
        if arg != "None": 
            formatted_args.append(f"({arg})")
    return formatted_args

class StepAudioRun:
    @classmethod
    def INPUT_TYPES(s):
        
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "speaker": (speaker_options, {"default": "婷婷"}),
            },
            "optional": {
                "custom_speaker": ("STRING", {"default": "", "multiline": False}),
                "emotion": (emotion_options, {"default": "None"}),
                "language": (language_options, {"default": "None"}),
                "express": (express_options, {"default": "None"}),
                "speed": (speed_options, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = "MW-Step-Audio"

    def speak(self, text, speaker, custom_speaker, emotion, language, express, speed):
        if not custom_speaker.strip():
            speaker = custom_speaker
        conditions = gen_text(emotion, language, express, speed)
        # print(conditions, end="\n\n")
        texts = [i.strip() for i in text.split("\n+") if  i.strip()]
        audio_data = []
        for i in texts:
            text = "".join(conditions) + " "  + i
            # print(text, end="\n\n")
            output_audio, sr = tts_engine(text, speaker)
            audio_data.append(output_audio)

        audio_tensor = torch.cat(audio_data, dim=1).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": sr},)


class StepAudioClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "speaker_name": ("STRING", {"default": "Myvoice", "multiline": False}),
                "clone_text": ("STRING", {"default": "", "multiline": True, "tooltip": "The clone audio's text."}),
                "clone_audio": ("AUDIO", ),
            },
            "optional": {
                "emotion": (emotion_options, {"default": "None"}),
                "language": (language_options, {"default": "None"}),
                "express": (express_options, {"default": "None"}),
                "speed": (speed_options, {"default": "None"}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "MW-Step-Audio"

    def clone(self, text, clone_audio, clone_text, speaker_name, emotion, language, express, speed):
        conditions = gen_text(emotion, language, express, speed)
        # print(conditions, end="\n\n")
        if not speaker_name.strip():
            speaker_name = "同样"

        clone_text = "".join(conditions) + " "  + clone_text
        # print(clone_text, end="\n\n")
        clone_speaker = {
            "audio": clone_audio,
            "speaker": speaker_name,
            "prompt_text": clone_text
        }

        texts = [i.strip() for i in text.split("\n+") if  i.strip()]
        audio_data = []
        for i in texts:
            text = "".join(conditions) + " "  + i
            # print(text, end="\n\n")
            output_audio, sr = tts_engine(text, speaker_name, clone_speaker)
            audio_data.append(output_audio)

        audio_tensor = torch.cat(audio_data, dim=1).unsqueeze(0).float()
        return ({"waveform": audio_tensor, "sample_rate": sr},)
    

NODE_CLASS_MAPPINGS = {
    "StepAudioRun": StepAudioRun,
    "StepAudioClone": StepAudioClone,
}
