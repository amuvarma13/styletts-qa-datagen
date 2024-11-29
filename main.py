from scipy.io.wavfile import write
import msinference 
import random
import nltk
import librosa
from datasets import load_dataset, Audio

nltk.download('punkt_tab')
text = 'Hello world!'
voices_strings = ["f-us-1.wav", "f-us-2.wav", "f-us-3.wav", "f-us-4.wav", "m-us-1.wav", "m-us-2.wav", "m-us-3.wav", "m-us-4.wav"]
voices = [msinference.compute_style("voices/"+voice) for voice in voices_strings]
dsn = "amuvarma/qa-tospeech_2"
push_name = "amuvarma/qa_large_0_0_speechq"

ds = load_dataset(dsn, split='train')
ds = ds.shuffle(seed=42)
def add_audio(example):
    text = example['question']
    voice = random.choice(voices)
    wav = msinference.inference(
        text, 
        voice, 
        alpha=0.3, 
        beta=0.7, 
        diffusion_steps=7, 
        embedding_scale=1
    )
    
    wav_16k = librosa.resample(wav, orig_sr=24000, target_sr=16000)
    
    print(wav_16k.shape)
    
    return {
        'audio': {
            'array': wav_16k,
            'sampling_rate': 16000
        }
    }

ds = ds.map(add_audio, batched=False)
ds = ds.cast_column("audio", Audio(sampling_rate=16000))


ds.push_to_hub(push_name)
