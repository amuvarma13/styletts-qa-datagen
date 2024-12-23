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
dsn = "aamuvarma/conversation-elias-5-0-t248"
push_name = "amuvarma/conversation-elias-5-0-t248-convo-both-full"

ds = load_dataset(dsn, split='train')
ds = ds.shuffle(seed=42)
# ds = ds.select(range(10))
cols_of_interest = ['user_1_text', 'user_2_text', 'user_3_text', 'user_4_text', 'user_5_text', 'user_6_text']

def add_audio(example):
    try:
        #first prefill the audio columns with None
        updated_example = {
            "user_1_text_audio": None,
            "user_2_text_audio": None,
            "user_3_text_audio": None,
            "user_4_text_audio": None,
            "user_5_text_audio": None,
            "user_6_text_audio": None
        }

        for col in cols_of_interest:
            print(f"Processing {col}")  
            if not example[col]:
                print(f"Skipping {col} as it is empty")
                updated_example[f'{col}_audio'] = None
                return updated_example
            text = example[col]
            print(f"Text: {text}")
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
            
            print(f"Adding audio for {col}")
            updated_example[f'{col}_audio'] = {
                'array': wav_16k,
                'sampling_rate': 16000
            }
        print("Done processing", updated_example)
        return updated_example
    except Exception as e:
        print(f"Failed to process example: {e}")
        return {
            'answer_audio': None  # Or you could return a default value
        }

ds = ds.map(add_audio, batched=False)

print(ds)
for col in cols_of_interest:
    ds = ds.cast_column(f'{col}_audio', Audio(sampling_rate=16000))


ds.push_to_hub(push_name)
