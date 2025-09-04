import os
from loguru import logger
import torch
import numpy as np
import threading
import time
from torch.nn import functional as F
from contextlib import nullcontext
import uuid
from viettts.utils.common import fade_in_out_audio
import torch.nn.functional as F

# Force FP32 trên CPU để tránh lỗi Half precision
if not torch.cuda.is_available():
    torch.set_default_dtype(torch.float32)

def ensure_float32(tensor):
    """Đảm bảo tensor là FP32 trên CPU"""
    if tensor.dtype == torch.float16 and tensor.device.type == 'cpu':
        return tensor.float()
    return tensor
class TTSModel:
    def __init__(
        self,
        llm: torch.nn.Module,
        flow: torch.nn.Module,
        hift: torch.nn.Module
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.llm = llm
        self.flow = flow
        self.hift = hift
        self.token_min_hop_len = 2 * self.flow.input_frame_rate
        self.token_max_hop_len = 4 * self.flow.input_frame_rate
        self.token_overlap_len = 20
        # mel fade in out
        self.mel_overlap_len = int(self.token_overlap_len / self.flow.input_frame_rate * 22050 / 256)
        self.mel_window = np.hamming(2 * self.mel_overlap_len)
        # hift cache
        self.mel_cache_len = 20
        self.source_cache_len = int(self.mel_cache_len * 256)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        assert self.stream_scale_factor >= 1, 'stream_scale_factor should be greater than 1, change it according to your actual rtf'
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()
        self.lock = threading.Lock()
        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.hift_cache_dict = {}
         # Force tất cả models về FP32 trên CPU
        if not torch.cuda.is_available():
            if hasattr(self, 'llm') and self.llm is not None:
                self.llm = self.llm.float()
            if hasattr(self, 'flow') and self.flow is not None:
                self.flow = self.flow.float()
            if hasattr(self, 'hift') and self.hift is not None:
                self.hift = self.hift.float()

    def load(self, llm_model, flow_model, hift_model):
        """
        Smart model loading with automatic CPU/GPU detection and precision handling
        """
        # Determine device and precision based on availability
        if torch.cuda.is_available() and not os.environ.get('FORCE_CPU', False):
            self.device = torch.device('cuda')
            use_half_precision = True
            print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            use_half_precision = False
            print("🖥️  Using CPU (GPU not available or FORCE_CPU=True)")
        
        # Load LLM model
        print("Loading LLM model...")
        llm_checkpoint = torch.load(llm_model, map_location=self.device)
        
        # Convert half precision to float32 if loading on CPU
        if not use_half_precision and self.device.type == 'cpu':
            llm_checkpoint = self._convert_checkpoint_precision(llm_checkpoint)
        
        self.llm.load_state_dict(llm_checkpoint)
        self.llm.to(self.device).eval()
        
        # Apply precision: half() only on GPU, float() on CPU
        if use_half_precision:
            self.llm.half()
            print("✅ LLM loaded with FP16 precision on GPU")
        else:
            self.llm.float()
            print("✅ LLM loaded with FP32 precision on CPU")
        
        # Load Flow model
        print("Loading Flow model...")
        flow_checkpoint = torch.load(flow_model, map_location=self.device)
        
        if not use_half_precision and self.device.type == 'cpu':
            flow_checkpoint = self._convert_checkpoint_precision(flow_checkpoint)
        
        self.flow.load_state_dict(flow_checkpoint)
        self.flow.to(self.device).eval()
        
        if use_half_precision:
            self.flow.half()
            print("✅ Flow loaded with FP16 precision on GPU")
        else:
            self.flow.float()
            print("✅ Flow loaded with FP32 precision on CPU")
        
        # Load HiFT model  
        print("Loading HiFT model...")
        hift_checkpoint = torch.load(hift_model, map_location=self.device)
        
        if not use_half_precision and self.device.type == 'cpu':
            hift_checkpoint = self._convert_checkpoint_precision(hift_checkpoint)
            
        self.hift.load_state_dict(hift_checkpoint)
        self.hift.to(self.device).eval()
        
        if use_half_precision:
            self.hift.half()
            print("✅ HiFT loaded with FP16 precision on GPU")
        else:
            self.hift.float() 
            print("✅ HiFT loaded with FP32 precision on CPU")
        
        print(f"🎉 All models loaded successfully on {self.device}")
    def _convert_checkpoint_precision(self, checkpoint):
        """
        Convert model checkpoint from half precision to float32 for CPU compatibility
        """
        def convert_recursive(obj):
            if isinstance(obj, torch.Tensor):
                if obj.dtype == torch.float16:
                    return obj.float()
                return obj
            elif isinstance(obj, dict):
                return {k: convert_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(convert_recursive(item) for item in obj)
            return obj
        
        return convert_recursive(checkpoint)
    def load_jit(self, llm_text_encoder_model, llm_llm_model, flow_encoder_model):
        llm_text_encoder = torch.jit.load(llm_text_encoder_model, map_location=self.device)
        self.llm.text_encoder = llm_text_encoder
        llm_llm = torch.jit.load(llm_llm_model, map_location=self.device)
        self.llm.llm = llm_llm
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_onnx(self, flow_decoder_estimator_model):
        import onnxruntime
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        providers = ['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        del self.flow.decoder.estimator
        self.flow.decoder.estimator = onnxruntime.InferenceSession(flow_decoder_estimator_model, sess_options=option, providers=providers)

    def llm_job(self, text, prompt_text, llm_prompt_speech_token, llm_embedding, uuid):
        with self.llm_context:
            for i in self.llm.inference(
                text=text.to(self.device),
                text_len=torch.tensor([text.shape[1]], dtype=torch.int32).to(self.device),
                prompt_text=prompt_text.to(self.device),
                prompt_text_len=torch.tensor([prompt_text.shape[1]], dtype=torch.int32).to(self.device),
                prompt_speech_token=llm_prompt_speech_token.to(self.device),
                prompt_speech_token_len=torch.tensor([llm_prompt_speech_token.shape[1]], dtype=torch.int32).to(self.device),
                embedding=llm_embedding.to(self.device).half()
            ):
                self.tts_speech_token_dict[uuid].append(i)
        self.llm_end_dict[uuid] = True

    def token2wav(self, token, prompt_token, prompt_feat, embedding, uuid, finalize=False, speed=1.0):
        tts_mel = self.flow.inference(
            token=token.to(self.device),
            token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_token=prompt_token.to(self.device),
            prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
            prompt_feat=prompt_feat.to(self.device),
            prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
            embedding=embedding.to(self.device)
        )

        if self.hift_cache_dict[uuid] is not None:
            hift_cache_mel, hift_cache_source = self.hift_cache_dict[uuid]['mel'], self.hift_cache_dict[uuid]['source']
            tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
        else:
            hift_cache_source = torch.zeros(1, 1, 0)

        if finalize is False:
            self.mel_overlap_dict[uuid] = tts_mel[:, :, -self.mel_overlap_len:]
            tts_mel = tts_mel[:, :, :-self.mel_overlap_len]
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)
            self.hift_cache_dict[uuid] = {
                'mel': tts_mel[:, :, -self.mel_cache_len:],
                'source': tts_source[:, :, -self.source_cache_len:],
                'speech': tts_speech[:, -self.source_cache_len:]
            }
            tts_speech = tts_speech[:, :-self.source_cache_len]
        else:
            if speed != 1.0:
                assert self.hift_cache_dict[uuid] is None, 'speed change only support non-stream inference mode'
                tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
            tts_speech, tts_source = self.hift.inference(mel=tts_mel, cache_source=hift_cache_source)

        tts_speech = fade_in_out_audio(tts_speech)
        return tts_speech

    def tts(
        self,
        text: str,
        flow_embedding: torch.Tensor,
        llm_embedding: torch.Tensor=torch.zeros(0, 192),
        prompt_text: torch.Tensor=torch.zeros(1, 0, dtype=torch.int32),
        llm_prompt_speech_token: torch.Tensor=torch.zeros(1, 0, dtype=torch.int32),
        flow_prompt_speech_token: torch.Tensor=torch.zeros(1, 0, dtype=torch.int32),
        prompt_speech_feat: torch.Tensor=torch.zeros(1, 0, 80),
        stream: bool=False,
        speed: float=1.0,
        **kwargs
    ):
        # this_uuid is used to track variables related to this inference thread
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
            self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = None, None
        
        p = threading.Thread(target=self.llm_job, args=(text, prompt_text, llm_prompt_speech_token, llm_embedding, this_uuid))
        p.start()
        
        if stream:
            token_hop_len = self.token_min_hop_len
            while True:
                time.sleep(0.01)
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]).unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        uuid=this_uuid,
                        finalize=False
                    )
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                finalize=True
            )
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            p.join()
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                finalize=True,
                speed=speed
            )
            yield {'tts_speech': this_tts_speech.cpu()}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)

    def vc(
        self,
        source_speech_token: torch.Tensor,
        flow_prompt_speech_token: torch.Tensor,
        prompt_speech_feat: torch.Tensor,
        flow_embedding: torch.Tensor,
        stream: bool=False,
        speed: float=1.0,
        **kwargs
    ):
        this_uuid = str(uuid.uuid1())
        with self.lock:
            self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = source_speech_token.flatten().tolist(), True
            self.mel_overlap_dict[this_uuid], self.hift_cache_dict[this_uuid] = None, None

        if stream:
            token_hop_len = self.token_min_hop_len
            while True:
                if len(self.tts_speech_token_dict[this_uuid]) >= token_hop_len + self.token_overlap_len:
                    this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid][:token_hop_len + self.token_overlap_len]) \
                        .unsqueeze(dim=0)
                    this_tts_speech = self.token2wav(
                        token=this_tts_speech_token,
                        prompt_token=flow_prompt_speech_token,
                        prompt_feat=prompt_speech_feat,
                        embedding=flow_embedding,
                        uuid=this_uuid,
                        finalize=False
                    )
                    yield {'tts_speech': this_tts_speech.cpu()}
                    with self.lock:
                        self.tts_speech_token_dict[this_uuid] = self.tts_speech_token_dict[this_uuid][token_hop_len:]
                    # increase token_hop_len for better speech quality
                    token_hop_len = min(self.token_max_hop_len, int(token_hop_len * self.stream_scale_factor))
                if self.llm_end_dict[this_uuid] is True and len(self.tts_speech_token_dict[this_uuid]) < token_hop_len + self.token_overlap_len:
                    break

            # deal with remain tokens, make sure inference remain token len equals token_hop_len when cache_speech is not None
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid], dim=1).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                finalize=True
            )
            yield {'tts_speech': this_tts_speech.cpu()}
        else:
            # deal with all tokens
            this_tts_speech_token = torch.tensor(self.tts_speech_token_dict[this_uuid]).unsqueeze(dim=0)
            this_tts_speech = self.token2wav(
                token=this_tts_speech_token,
                prompt_token=flow_prompt_speech_token,
                prompt_feat=prompt_speech_feat,
                embedding=flow_embedding,
                uuid=this_uuid,
                finalize=True,
                speed=speed
            )
            yield {'tts_speech': this_tts_speech.cpu()}

        with self.lock:
            self.tts_speech_token_dict.pop(this_uuid)
            self.llm_end_dict.pop(this_uuid)
            self.mel_overlap_dict.pop(this_uuid)
            self.hift_cache_dict.pop(this_uuid)
