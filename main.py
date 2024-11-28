from scipy.io.wavfile import write
import msinference 
import random
text = 'Hello world!'
voices_strings = ["f-us-1.wav", "f-us-2.wav", "f-us-3.wav", "f-us-4.wav", "m-us-1.wav", "m-us-2.wav", "m-us-3.wav", "m-us-4.wav"]
voices = [msinference.compute_style("voices/"+voice) for voice in voices_strings]
voice = random.choice(voices)
wav = msinference.inference(text, voice, alpha=0.3, beta=0.7, diffusion_steps=7, embedding_scale=1)
write('result.wav', 24000, wav)
