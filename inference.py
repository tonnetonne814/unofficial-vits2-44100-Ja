import warnings
warnings.filterwarnings(action='ignore')

import os
import time
import torch
import utils
import argparse
import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import cleaned_text_to_sequence
from g2p import pyopenjtalk_g2p_prosody, pyopenjtalk_g2p
import soundcard as sc
import soundfile as sf


def get_text(text, hps):
    text_norm = cleaned_text_to_sequence(text)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def inference(args):

    config_path = args.config
    G_model_path = args.model_path

    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU inference is not allowed."

    device="cuda:0"

    # load config.json
    hps = utils.get_hparams_from_file(config_path)
    
    if "use_noise_scaled_mas" in hps.model.keys() and hps.model.use_noise_scaled_mas == True:
      print("Using noise scaled MAS for VITS2")
      use_noise_scaled_mas = True
      mas_noise_scale_initial = 0.01
      noise_scale_delta = 2e-6
    else:
      print("Using normal MAS for VITS1")
      use_noise_scaled_mas = False
      mas_noise_scale_initial = 0.0
      noise_scale_delta = 0.0
    if "use_mel_posterior_encoder" in hps.model.keys() and hps.model.use_mel_posterior_encoder == True:
      print("Using mel posterior encoder for VITS2")
      posterior_channels = 80 #vits2
      hps.data.use_mel_posterior_encoder = True

    # load model
    net_g = SynthesizerTrn(
      len(symbols),
      posterior_channels,
      hps.train.segment_size // hps.data.hop_length,
      mas_noise_scale_initial = mas_noise_scale_initial,
      noise_scale_delta = noise_scale_delta,
      **hps.model).cuda()

    _ = net_g.eval()
    _ = utils.load_checkpoint(G_model_path, net_g, None)

    # play audio by system default
    speaker = sc.get_speaker(sc.default_speaker().name)

    # parameter settings
    noise_scale     = torch.tensor(0.66)    # adjust z_p noise
    noise_scale_w   = torch.tensor(0.8)    # adjust SDP noise
    length_scale    = torch.tensor(1.0)     # adjust sound length scale (talk speed)

    if args.is_save is True:
        n_save = 0
        save_dir = os.path.join("./infer_logs/")
        os.makedirs(save_dir, exist_ok=True)

    ### Dummy Input ###
    with torch.inference_mode():
        stn_phn = pyopenjtalk_g2p("速度計測のためのダミーインプットです。")
        stn_tst = get_text(stn_phn, hps)
        # generate audio
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        audio = net_g.infer(x_tst, 
                            x_tst_lengths, 
                            noise_scale=noise_scale, 
                            noise_scale_w=noise_scale_w, 
                            length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

    while True:
        # get text
        text = input("Enter text. ==> ")
        if text=="":
            print("Empty input is detected... Exit...")
            break
        
        # measure the execution time 
        torch.cuda.synchronize()
        start = time.time()

        # required_grad is False
        with torch.inference_mode():
            if hps.data.text_cleaners[0] == "prosody":
                stn_phn = pyopenjtalk_g2p_prosody(text)
                print(stn_phn)
            else:
                stn_phn = pyopenjtalk_g2p(text)
                print(stn_phn)
            stn_tst = get_text(stn_phn, hps)

            # generate audio
            x_tst = stn_tst.cuda().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
            audio = net_g.infer(x_tst, 
                                x_tst_lengths, 
                                noise_scale=noise_scale, 
                                noise_scale_w=noise_scale_w, 
                                length_scale=length_scale)[0][0,0].data.cpu().float().numpy()

        # measure the execution time 
        torch.cuda.synchronize()
        elapsed_time = time.time() - start
        print(f"Gen Time : {elapsed_time}")
        
        # play audio
        speaker.play(audio, hps.data.sampling_rate)
        
        # save audio
        if args.is_save is True:
            n_save += 1
            data = audio
            try:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text}.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")
            except:
                save_path = os.path.join(save_dir, str(n_save).zfill(3)+f"_{text[:10]}〜.wav")
                sf.write(
                     file=save_path,
                     data=data,
                     samplerate=hps.data.sampling_rate,
                     format="WAV")

            print(f"Audio is saved at : {save_path}")


    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        #required=True,
                        default="./vits2_pyopenjtalk_prosody/config.json" ,    
                        help='Path to configuration file')
    parser.add_argument('--model_path',
                        type=str,
                        #required=True,
                        default="./vits2_pyopenjtalk_prosody/G_87000.pth",
                        help='Path to checkpoint')
    parser.add_argument('--is_save',
                        type=str,
                        default=True,
                        help='Whether to save output or not')
    args = parser.parse_args()
    
    inference(args)