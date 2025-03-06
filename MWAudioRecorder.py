import numpy as np
import torch
import time
import librosa
import sounddevice as sd
from scipy import ndimage
from comfy.utils import ProgressBar

class AudioRecorder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 触发控制
                "trigger": ("BOOLEAN", {"default": False}),
                # 录音时长
                "record_sec": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 60,
                    "step": 1  # 整数秒递增
                }),
                "sample_rate": (["16000", "44100", "48000"], {  # 限定标准采样率
                    "default": "48000"
                }),
                "n_fft": ("INT", {  # 限定为2的幂次方
                    "default": 2048,
                    "min": 512,
                    "max": 4096,
                    "step": 512  # 只能选择512,1024,1536,2048,...4096
                }),
                "sensitivity": ("FLOAT", {  # 灵敏度精确控制
                    "default": 1.2,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.1  # 0.1步进
                }),
                "smooth": ("INT", {  # 确保为奇数
                    "default": 5,
                    "min": 1,
                    "max": 11,
                    "step": 2  # 生成1,3,5,7,9,11
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "record_and_clean"
    CATEGORY = "MW-Step-Audio"

    def _stft(self, y, n_fft):
        hop = n_fft // 4
        return librosa.stft(y, n_fft=n_fft, hop_length=hop, win_length=n_fft)

    def _istft(self, spec, n_fft):
        hop = n_fft // 4
        return librosa.istft(spec, hop_length=hop, win_length=n_fft)

    def _calc_noise_profile(self, noise_clip, n_fft):
        noise_spec = self._stft(noise_clip, n_fft)
        return {
            'mean': np.mean(np.abs(noise_spec), axis=1, keepdims=True),
            'std': np.std(np.abs(noise_spec), axis=1, keepdims=True)
        }

    def _spectral_gate(self, spec, noise_profile, sensitivity):
        threshold = noise_profile['mean'] + sensitivity * noise_profile['std']
        return np.where(np.abs(spec) > threshold, spec, 0)

    def _smooth_mask(self, mask, kernel_size):
        smoothed = ndimage.uniform_filter(mask, size=(kernel_size, kernel_size))
        return np.clip(smoothed * 1.2, 0, 1)  # 增强边缘保留

    def record_and_clean(self, trigger, record_sec, n_fft, sensitivity, smooth, sample_rate, seed):
        if not trigger:
            return (None,)

        sr = int(sample_rate)
        final_audio = None

        try:
            noise_clip = None
            # 主录音
            # print(f"开始主录音 {record_sec}秒...")
            main_rec = sd.rec(int(record_sec * sr), samplerate=sr, channels=1, dtype='float32')
            pb = ProgressBar(record_sec)
            for _ in range(record_sec * 2):
                time.sleep(0.5)
                pb.update(0.5)
            sd.wait()
            audio = main_rec.flatten()

            # 自动噪声检测
            if noise_clip is None: 
                # print("自动检测静默段作为噪声参考...")
                energy = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=n_fft//4)
                min_idx = np.argmin(energy)
                start = min_idx * (n_fft//4)
                noise_clip = audio[start:start + n_fft*2]

            # 降噪处理
            # print("进行频谱降噪...")
            noise_profile = self._calc_noise_profile(noise_clip, n_fft)
            spec = self._stft(audio, n_fft)
            
            # 多步骤处理
            mask = np.ones_like(spec)  # 初始掩膜
            for _ in range(2):  # 双重处理循环
                cleaned_spec = self._spectral_gate(spec, noise_profile, sensitivity)
                mask = np.where(np.abs(cleaned_spec) > 0, 1, 0)
                mask = self._smooth_mask(mask, smooth//2+1)
                spec = spec * mask

            # 相位恢复重建
            processed = self._istft(spec * mask, n_fft)
            
            # 动态增益归一化
            peak = np.max(np.abs(processed))
            processed = processed * (0.99 / peak) if peak > 0 else processed

            # 格式转换
            waveform = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0)
            final_audio = {"waveform": waveform, "sample_rate": sr}

        except Exception as e:
            print(f"Recording/processing failed: {str(e)}")
            raise

        return (final_audio,)

# 节点注册
NODE_CLASS_MAPPINGS = {
    "AudioRecorder": AudioRecorder
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioRecorder": "MW Audio Recorder"
}