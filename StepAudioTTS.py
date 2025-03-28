import torchaudio
import gc
import os
import json
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
import sys
import io
import threading
import time
import onnxruntime
import whisper

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sa_utils import resample_audio, energy_norm_fn, trim_silence
from funasr_detach import AutoModel 
from cosyvoice.cli.cosyvoice import CosyVoice 


node_dir = os.path.dirname(os.path.abspath(__file__))
comfy_path = os.path.dirname(os.path.dirname(node_dir))
model_path = os.path.join(comfy_path, "models/TTS")
encoder_model_path = os.path.join(model_path, "Step-Audio-Tokenizer")
tts_model_path = os.path.join(model_path, "Step-Audio-TTS-3B")
speaker_path = os.path.join(model_path, "Step-Audio-speakers")

CACHED_MODELS = {
    "funasr_model": None,
    "ort_cosy_tokenizer": None,
    "llm": None,
    "autotokenizer": None,
    "common_cosy_model": None,
    "music_cosy_model": None,
}

def load_models(device, use_cache=True):
    kms_path = os.path.join(encoder_model_path, "linguistic_tokenizer.npy")
    kms = torch.tensor(np.load(kms_path))
    
    if use_cache and all(CACHED_MODELS.values()):
        return (
            CACHED_MODELS["funasr_model"],
            kms,
            CACHED_MODELS["ort_cosy_tokenizer"],
            CACHED_MODELS["llm"],
            CACHED_MODELS["autotokenizer"],
            CACHED_MODELS["common_cosy_model"],
            CACHED_MODELS["music_cosy_model"],
        )
    else:
        funasr_model_path = os.path.join(
            encoder_model_path,
            "dengcunqin/speech_paraformer-large_asr_nat-zh-cantonese-en-16k-vocab8501-online",
        )
        funasr_model = AutoModel(model=funasr_model_path, model_revision="master", device=device)

        cosy_tokenizer_path = os.path.join(encoder_model_path, "speech_tokenizer_v1.onnx")
        providers = ["CUDAExecutionProvider"]
        session_option = onnxruntime.SessionOptions()
        session_option.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session_option.intra_op_num_threads = 1
        ort_cosy_tokenizer = onnxruntime.InferenceSession(
            cosy_tokenizer_path, sess_options=session_option, providers=providers
        )

        llm = AutoModelForCausalLM.from_pretrained(
            tts_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        autotokenizer = AutoTokenizer.from_pretrained(
            tts_model_path,
            trust_remote_code=True
        )
        common_cosy_model = CosyVoice(os.path.join(tts_model_path, "CosyVoice-300M-25Hz"))
        music_cosy_model = CosyVoice(os.path.join(tts_model_path, "CosyVoice-300M-25Hz-Music"))
        
        CACHED_MODELS["funasr_model"] = funasr_model
        CACHED_MODELS["ort_cosy_tokenizer"] = ort_cosy_tokenizer
        CACHED_MODELS["llm"] = llm
        CACHED_MODELS["autotokenizer"] = autotokenizer
        CACHED_MODELS["common_cosy_model"] = common_cosy_model
        CACHED_MODELS["music_cosy_model"] = music_cosy_model
        del funasr_model
        del ort_cosy_tokenizer
        del llm
        del autotokenizer
        del common_cosy_model
        del music_cosy_model
        gc.collect()
        torch.cuda.empty_cache()
        
        return (
            CACHED_MODELS["funasr_model"],
            kms,
            CACHED_MODELS["ort_cosy_tokenizer"],
            CACHED_MODELS["llm"],
            CACHED_MODELS["autotokenizer"],
            CACHED_MODELS["common_cosy_model"],
            CACHED_MODELS["music_cosy_model"],
        )


def clear_cached_models():
    for key in CACHED_MODELS:
        CACHED_MODELS[key] = None
    gc.collect()
    torch.cuda.empty_cache()


class StepAudioTokenizer:
    def __init__(
        self,
        funasr_model,
        kms,
        ort_cosy_tokenizer,
        device
    ):
        self.funasr_model = funasr_model
        self.kms = kms
        self.ort_session = ort_cosy_tokenizer
        self.device = device

        self.chunk_size = [0, 4, 5]
        self.encoder_chunk_look_back = 4
        self.decoder_chunk_look_back = 1

        self.vq02_sessions = {}
        self.vq02_lock = threading.Lock()
        self.vq06_lock = threading.Lock()

    def cleanup(self):
        self.funasr_model = None
        self.kms = None
        self.ort_session = None
        gc.collect()
        torch.cuda.empty_cache()

    def __call__(self, audio, sr):
        _, vq02, vq06 = self.wav2token(audio, sr, False)
        text = self.merge_vq0206_to_token_str(vq02, vq06)
        return text

    def preprocess_wav(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = resample_audio(audio, sample_rate, 16000)
        if energy_norm:
            audio = energy_norm_fn(audio)

        if enable_trim:
            audio = audio.cpu().numpy().squeeze(0)
            audio = trim_silence(audio, 16000)
            audio = torch.from_numpy(audio)
            audio = audio.unsqueeze(0)
        return audio

    def wav2token(self, audio, sample_rate, enable_trim=True, energy_norm=True):
        audio = self.preprocess_wav(
            audio, sample_rate, enable_trim=enable_trim, energy_norm=energy_norm
        )

        vq02_ori = self.get_vq02_code(audio)
        vq02 = [int(x) + 65536 for x in vq02_ori]
        vq06_ori = self.get_vq06_code(audio)
        vq06 = [int(x) + 65536 + 1024 for x in vq06_ori]

        chunk = 1
        chunk_nums = min(len(vq06) // (3 * chunk), len(vq02) // (2 * chunk))
        speech_tokens = []
        for idx in range(chunk_nums):
            speech_tokens += vq02[idx * chunk * 2 : (idx + 1) * chunk * 2]
            speech_tokens += vq06[idx * chunk * 3 : (idx + 1) * chunk * 3]
        return speech_tokens, vq02_ori, vq06_ori

    def get_vq02_code(self, audio, session_id=None, is_final=True):
        _tmp_wav = io.BytesIO()
        torchaudio.save(_tmp_wav, audio, 16000, format="wav")
        _tmp_wav.seek(0)

        with self.vq02_lock:
            cache = {}
            if session_id in self.vq02_sessions:
                cache = self.vq02_sessions[session_id].get("cache", {})

            res, new_cache = self.funasr_model.infer_encoder(
                input=[_tmp_wav],
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=self.encoder_chunk_look_back,
                decoder_chunk_look_back=self.decoder_chunk_look_back,
                device=self.device,
                is_final=is_final,
                cache=cache,
            )
            c_list = []
            for j, res_ in enumerate(res):
                feat = res_["enc_out"]
                if len(feat) > 0:
                    c_list = self.dump_label([feat], self.kms)[0]

            if is_final:
                if session_id in self.vq02_sessions:
                    self.vq02_sessions.pop(session_id)
            else:
                if isinstance(session_id, str) and len(session_id) > 0:
                    self.vq02_sessions[session_id] = {
                        "cache": new_cache,
                        "update_time": time.time(),
                    }

            return c_list

    def get_vq06_code(self, audio):

        def split_audio(audio, chunk_duration=480000):
            start = 0
            chunks = []
            while start < len(audio):
                end = min(start + chunk_duration, len(audio))
                chunk = audio[start:end]
                if len(chunk) < 480:
                    pass
                else:
                    chunks.append(chunk)
                start = end
            return chunks

        with self.vq06_lock:
            audio = audio.squeeze(0)
            chunk_audios = split_audio(audio, chunk_duration=30 * 16000)  # 最大支持30s
            speech_tokens = []
            for chunk in chunk_audios:
                duration = round(chunk.shape[0] / 16000, 2)
                feat = whisper.log_mel_spectrogram(chunk, n_mels=128)
                feat = feat.unsqueeze(0)
                feat_len = np.array([feat.shape[2]], dtype=np.int32)
                chunk_token = (
                    self.ort_session.run(
                        None,
                        {
                            self.ort_session.get_inputs()[0]
                            .name: feat.detach()
                            .cpu()
                            .numpy(),
                            self.ort_session.get_inputs()[1].name: feat_len,
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                assert abs(len(chunk_token) - duration * 25) <= 2
                speech_tokens += chunk_token

            return speech_tokens

    def kmean_cluster(self, samples, means):
        dists = torch.cdist(samples, means)
        indices = dists.argmin(dim=1).cpu().numpy()
        return indices.tolist()

    def dump_label(self, samples, mean):
        dims = samples[0].shape[-1]
        x_lens = [x.shape[1] for x in samples]
        total_len = sum(x_lens)
        x_sel = torch.FloatTensor(1, total_len, dims)
        start_len = 0
        for sample in samples:
            sample_len = sample.shape[1]
            end_len = start_len + sample_len
            x_sel[:, start_len:end_len] = sample
            start_len = end_len
        dense_x = x_sel.squeeze(0)
        indices = self.kmean_cluster(dense_x, mean)
        indices_list = []
        start_len = 0
        for x_len in x_lens:
            end_len = start_len + end_len
            indices_list.append(indices[start_len:end_len])
        return indices_list

    def merge_vq0206_to_token_str(self, vq02, vq06):
        _vq06 = [1024 + x for x in vq06]
        result = []
        i = 0
        j = 0
        while i < len(vq02) - 1 and j < len(_vq06) - 2:
            sublist = vq02[i : i + 2] + _vq06[j : j + 3]
            result.extend(sublist)
            i += 2
            j += 3
        return "".join([f"<audio_{x}>" for x in result])


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
        encoder,
        llm,
        autotokenizer,
        cosy_model,
        device,
    ):
        self.llm = llm
        self.autotokenizer = autotokenizer
        self.cosy_model = cosy_model
        self.device = device
        self.encoder = encoder

    def cleanup(self):
        self.llm = None
        self.autotokenizer = None
        self.cosy_model = None
        self.encoder = None
        gc.collect()
        torch.cuda.empty_cache()


    def data_preprocess(self, prompt_speaker: str, clone_dict: dict | None = None):
        prompt_speaker_info =  {}
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

        else:
            encodings = ["utf-8", "gbk", "utf-8-sig"]  # utf-8-sig 处理带 BOM 的 UTF-8
            for encoding in encodings:
                try:
                    with open(f"{speaker_path}/speakers_info.json", "r", encoding=encoding) as f:
                        speakers_info = json.load(f)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(f"Failed to decode {speaker_path}/speakers_info.json with encodings {encodings}")

            if prompt_speaker not in speakers_info.keys():
                raise ValueError("There is no such speaker") 
                
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

        return prompt_speaker_info


    def tokenize_history(
        self, 
        text,
        marks: list, 
        prompt_text: str, 
        prompt_speaker: str, 
        prompt_code: list
    ):
        sys_prompt_dict = {
            "sys_prompt_for_rap": "请用 RAP 方式将文本内容大声说唱出来。[] 括号内标注了说唱者的名字, 请使用 [{}] 的声音, 大声说唱出其后面的文本内容: ",
            "sys_prompt_for_vocal": "请用哼唱的方式将文本内容大声唱出来。[] 括号内标注了唱歌者的名字, 请使用 [{}] 的声音, 大声唱出其后面的文本内容: ",
            "sys_prompt_for_spk": ("作为一名卓越的声优演员，你的任务是根据文本中 （） 或 () 括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，"
            '以丰富细腻的情感和自然顺畅的语调，来朗读文本。[] 括号内标注了朗读者的名字, 请使用 [{}] 的声音, 大声朗读出其后面的文本内容: '
            '\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n'
            '# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n\n'
            '# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n'
            '- "快速1"\n- "快速2"\n\n请在朗读时，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，'
            '如果没有 () 或 （） 括号，则根据文本语义内容恰到好处地演绎。'),
            "sys_prompt_for_clone": ("作为一名卓越的声优演员，你的任务是根据文本中 （） 或 () 括号内标注的情感、语种或方言、音乐哼唱、语音调整等标签，"
            '以丰富细腻的情感和自然顺畅的语调，来朗读文本。请使用历史会话的声音, 根据标签要求大声朗读出文本内容: '
            '\n# 情感标签涵盖了多种情绪状态，包括但不限于：\n- "高兴1"\n- "高兴2"\n- "生气1"\n- "生气2"\n- "悲伤1"\n- "撒娇1"\n\n'
            '# 语种或方言标签包含多种语言或方言，包括但不限于：\n- "中文"\n- "英文"\n- "韩语"\n- "日语"\n- "四川话"\n- "粤语"\n\n'
            '# 音乐哼唱标签包含多种类型歌曲哼唱，包括但不限于：\n- "RAP"\n- "哼唱"\n\n# 语音调整标签，包括但不限于：\n- "慢速1"\n- "慢速2"\n'
            '- "快速1"\n- "快速2"\n\n请在朗读时，根据这些情感标签的指示，调整你的情感、语气、语调和哼唱节奏，以确保文本的情感和意义得到准确而生动的传达，'
            '如果没有 () 或 （） 括号，则根据文本语义内容恰到好处地演绎。'),
        }

        if prompt_speaker == None:
            prompt = sys_prompt_dict["sys_prompt_for_clone"]
            # print("克隆系统消息: ", prompt, end="\n\n")
        else:
            if marks:
                if marks[0] == "(哼唱)":
                    prompt = sys_prompt_dict["sys_prompt_for_vocal"].format(prompt_speaker)
                    # print("哼唱系统消息: ", prompt, end="\n\n")
                elif marks[0] == "(RAP)":
                    prompt = sys_prompt_dict["sys_prompt_for_rap"].format(prompt_speaker)
                    # print("RAP系统消息: ", prompt, end="\n\n")
                else:
                    prompt = sys_prompt_dict["sys_prompt_for_spk"].format(prompt_speaker)
                    # print("其他系统消息: ", prompt, end="\n\n")
            else:
                prompt = sys_prompt_dict["sys_prompt_for_spk"].format(prompt_speaker)
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

    def preprocess_prompt_wav(self, audio):
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
            self.cosy_model.frontend._extract_speech_feat(prompt_wav_22k)
        )
        speech_embedding = self.cosy_model.frontend._extract_spk_embedding(
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

    with torch.no_grad():
        def generate(self, 
                    text: str, 
                    marks: list, 
                    prompt_speaker: str, 
                    clone_dict: dict | None = None, 
                    max_length: int = 8192, 
                    temperature: float = 0.7, 
                    do_sample: bool = True,
                    ):
            prompt_speaker_info = self.data_preprocess(
                prompt_speaker, clone_dict
            )

            token_ids = self.tokenize_history(
                text,
                marks,
                prompt_speaker_info["prompt_text"],
                prompt_speaker,
                prompt_speaker_info["prompt_code"],
            )

            output_ids = self.llm.generate(
                torch.tensor([token_ids]).to(torch.long).to(self.device),
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                logits_processor=LogitsProcessorList([RepetitionAwareLogitsProcessor()]),
            )
            output_ids = output_ids[:, len(token_ids) : -1]  # skip eos token
            return (
                self.cosy_model.token_to_wav_offline(
                    output_ids - 65536,
                    prompt_speaker_info["cosy_speech_feat"].to(torch.bfloat16),
                    prompt_speaker_info["cosy_speech_feat_len"],
                    prompt_speaker_info["cosy_prompt_token"],
                    prompt_speaker_info["cosy_prompt_token_len"],
                    prompt_speaker_info["cosy_speech_embedding"].to(torch.bfloat16),
                ),
                22050,
            )
        

# 选项列表
emotion_options = ["高兴1", "高兴2", "生气1", "生气2", "悲伤1", "撒娇1", "None"]
language_options = ["中文", "英文", "韩语", "日语", "四川话", "粤语", "None"]
speed_options = ["慢速1", "慢速2", "快速1", "快速2", "None"]
express_options = ["RAP", "哼唱", "None"]

with open(f"{speaker_path}/speakers_info.json", "r", encoding="utf-8") as f:
    speakers_info = json.load(f)
speaker_options = list(speakers_info.keys())    

def gen_tags(*args):
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
                "emotion": (emotion_options, {"default": "None"}),
                "language": (language_options, {"default": "None"}),
                "express": (express_options, {"default": "None"}),
                "speed": (speed_options, {"default": "None"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_length": ("INT", {"default": 8192, "min": 0}),
                "do_sample": ("BOOLEAN", {"default": True,}),
                "custom_mark": ("STRING", {"default": "", "multiline": False}),
                "unload_model": ("BOOLEAN", {"default": False,}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "speak"
    CATEGORY = "🎤MW/MW-Step-Audio"

    def speak(self, 
              text, 
              speaker, 
              emotion, 
              language, 
              express, 
              speed, 
              temperature=0.7, 
              max_length=8192, 
              do_sample=True, 
              custom_mark="",
              unload_model=False,
              ):
                
        device = "cuda"
        funasr_model, kms, ort_cosy_tokenizer, llm, autotokenizer, common_cosy_model, music_cosy_model = load_models(device, use_cache=True)

        encoder = StepAudioTokenizer(
            funasr_model,
            kms,
            ort_cosy_tokenizer,
            device,
        )
        custom_mark = custom_mark.strip() if custom_mark.strip() else None
        
        if express == "哼唱":
            marks = ["(哼唱)"]
        elif express == "RAP":
            marks = ["(RAP)"]
        else:
            marks = gen_tags(emotion, language, speed, custom_mark)

        if "(RAP)" in marks or "(哼唱)" in marks:
            cosy_model = music_cosy_model
        else:
            cosy_model = common_cosy_model

        tts_engine = StepAudioTTS(
            encoder,
            llm,
            autotokenizer,
            cosy_model,
            device,
        )
        
        text = "".join(marks) + f"[{speaker}]: " + text

        output_audio, sr = tts_engine.generate(
                                                text,
                                                marks,
                                                speaker,
                                                clone_dict=None,
                                                max_length=max_length,
                                                temperature=temperature,
                                                do_sample=do_sample,
                                                )

        audio_tensor = output_audio.unsqueeze(0).float()

        if unload_model:
            tts_engine.cleanup()
            encoder.cleanup()
            clear_cached_models()
            del funasr_model
            del kms
            del ort_cosy_tokenizer
            del llm
            del autotokenizer
            del common_cosy_model
            del music_cosy_model
            del cosy_model
            gc.collect()
            torch.cuda.empty_cache()
        
        return ({"waveform": audio_tensor, "sample_rate": sr},)


class StepAudioClone:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clone_audio": ("AUDIO", ),
                "clone_text": ("STRING", {"default": "", "multiline": True, "tooltip": "The clone audio's text."}),
                "text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "emotion": (emotion_options, {"default": "None"}),
                "language": (language_options, {"default": "None"}),
                "express": (express_options, {"default": "None"}),
                "speed": (speed_options, {"default": "None"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 1, "step": 0.1}),
                "max_length": ("INT", {"default": 8192, "min": 0}),
                "do_sample": ("BOOLEAN", {"default": True,}),
                "custom_mark": ("STRING", {"default": "", "multiline": False}),
                "unload_model": ("BOOLEAN", {"default": False,}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "clone"
    CATEGORY = "🎤MW/MW-Step-Audio"

    def clone(self, 
              text, 
              clone_audio,
              clone_text,
              emotion, 
              language, 
              express, 
              speed, 
              temperature=0.7, 
              max_length=8192, 
              do_sample=True, 
              custom_mark="",
              unload_model=False,
              ):
                
        device = "cuda"
        funasr_model, kms, ort_cosy_tokenizer, llm, autotokenizer, common_cosy_model, music_cosy_model = load_models(device, use_cache=True)

        encoder = StepAudioTokenizer(
            funasr_model,
            kms,
            ort_cosy_tokenizer,
            device,
        )
        custom_mark = custom_mark.strip() if custom_mark.strip() else None
        
        if express == "哼唱":
            marks = ["(哼唱)"]
        elif express == "RAP":
            marks = ["(RAP)"]
        else:
            marks = gen_tags(emotion, language, speed, custom_mark)

        if "(RAP)" in marks or "(哼唱)" in marks:
            cosy_model = music_cosy_model
        else:
            cosy_model = common_cosy_model

        tts_engine = StepAudioTTS(
            encoder,
            llm,
            autotokenizer,
            cosy_model,
            device,
        )

        text = "".join(marks) + f" {text}"
        clone_dict = {"prompt_text": clone_text, "audio": clone_audio}

        output_audio, sr = tts_engine.generate(
                                                text,
                                                marks,
                                                prompt_speaker=None,
                                                clone_dict=clone_dict,
                                                max_length=max_length,
                                                temperature=temperature,
                                                do_sample=do_sample,
                                                )

        audio_tensor = output_audio.unsqueeze(0).float()

        if unload_model:
            tts_engine.cleanup()
            encoder.cleanup()
            clear_cached_models()
            del funasr_model
            del kms
            del ort_cosy_tokenizer
            del llm
            del autotokenizer
            del common_cosy_model
            del music_cosy_model
            del cosy_model
            gc.collect()
            torch.cuda.empty_cache()
        
        return ({"waveform": audio_tensor, "sample_rate": sr},)
    

from MWAudioRecorder import AudioRecorder

NODE_CLASS_MAPPINGS = {
    "StepAudioRun": StepAudioRun,
    "StepAudioClone": StepAudioClone,
    "AudioRecorder": AudioRecorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StepAudioRun": "Step Audio Run",
    "StepAudioClone": "Step Audio Clone",
    "AudioRecorder": "MW Audio Recorder"
}