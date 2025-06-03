import argparse
import os
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sae_training.sae_group import SAEGroup
from sae_vis.model_fns import TransformerLensWrapper
from transformer_lens import HookedTransformer, utils
from feat_fns import get_feature_activations, get_features_activations
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from tqdm import tqdm
from cycler import cycler
import matplotlib as mpl

# For example, to use a larger set of distinct colors:
colors = plt.cm.get_cmap('tab20').colors  # 'tab20' has 20 distinct colors
mpl.rcParams['axes.prop_cycle'] = cycler(color=colors)
# Centralized Plot Configuration
PLOT_CONFIG = {
    'font.size':46,           # Default font size
    'axes.titlesize': 46,      # Font size for subplot titles
    'axes.labelsize': 46,      # Font size for x and y labels
    'legend.fontsize': 35,     # Font size for legends
    'xtick.labelsize': 44,     # Font size for x-axis tick labels
    'ytick.labelsize': 44,     # Font size for y-axis tick labels
    'figure.figsize': (30, 10), # Adjusted figure size for better layout
}
rcParams.update(PLOT_CONFIG)
LINE_WIDTH = 5  # Line thickness for plots

# Parameters
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="Device to run on (e.g., 'cuda' or 'cpu')")
parser.add_argument("--model_type", type=str, required=True, help="Model type to use")
parser.add_argument("--top_k", type=int, default=25, help="Number of top-k features to visualize")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
parser.add_argument("--token_num", type=int, default=8192 * 4, help="Number of tokens to process")

args = parser.parse_args()
device = args.device
model_type = args.model_type
top_k = args.top_k
token_num = args.token_num

feature_indices = [187,99,81,49,68,37,5]
feature_names = ["while","M/movement","up","Hel/hel","license/ed/ing","related to sports divisions","related to psychological, emotional, and personal states"]


# Load model and checkpoints based on model_type
if model_type == "pythia-410m-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    sae_dict_1 = {
        0: "checkpoints/921e0mgk/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        1: "pipe/52lieq0l/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        2: "pipe/18ekwo2k/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        3: "pipe/epmdboe7/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        4: "pipe/l2no0yut/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        5: "pipe/e7shgo9p/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        6: "pipe/ebs8r6qr/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        7: "pipe/vx8xb2k3/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        8: "pipe/v07opmp2/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        9: "pipe/io568vlx/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        10: "pipe/trbwvb9d/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        11: "pipe/x9qb3koe/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        13: "pipe/59zu9leh/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        15: "pipe/hpt1cy9i/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        17: "pipe/2i4w2133/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        19: "pipe/bo3lc5so/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        21: "pipe/fqohwi3s/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        23: "pipe/490ovm5t/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        25: "pipe/dvk5euee/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        27: "pipe/0nvrwgo9/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        29: "pipe/r4k7r6y6/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        31: "pipe/ein8fps8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        33: "pipe/io4vov51/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        53: "pipe/y4x2bx9y/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        73: "pipe/ib1hrafl/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        93: "pipe/3oih20d8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        113: "pipe/nh42ylwt/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        133: "pipe/30737iqa/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
        153: "pipe/07u89qd0/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt"
    }
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-160m-deduped":
    ckpt_range = list(range(13)) + list(range(13, 33, 2)) + list(range(33, 154, 20))
    hkpt = "blocks.4.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    sae_dict_1 = {
        0: "checkpoints/lgbb13td/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        1: "pipe/8q8t8xpw/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        2: "pipe/tnu76xgi/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        3: "pipe/0043k90y/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        4: "pipe/gxg4e8d2/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        5: "pipe/if2frw09/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        6: "pipe/eb3j4739/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        7: "pipe/omju93gv/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        8: "pipe/qz9weoem/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        9: "pipe/vmlbirof/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        10: "pipe/5fo4imbk/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        11: "pipe/k3xy71u9/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        12: "pipe/85na2fvx/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        13: "pipe/ve7q8qar/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        15: "pipe/9dlhhzrw/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        17: "pipe/913fpjov/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        19: "pipe/c8ca4dc6/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        21: "pipe/6rnb2ffm/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        23: "pipe/gs4gqm2x/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        25: "pipe/0xi7t5bi/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        27: "pipe/mjgaqocj/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        29: "pipe/wy1ohgw8/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        31: "pipe/2xdhe64r/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        33: "pipe/hgdsrhs4/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        53: "pipe/dl4ij188/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        73: "pipe/yx04fkry/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        93: "pipe/1y2oepxv/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        113: "pipe/kbwcbnkh/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        133: "pipe/flovd4dt/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
        153: "pipe/lig6rc8y/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt"
    }

    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type == "pythia-1.4b-deduped":
    ckpt_range = list(range(12)) + list(range(13, 33, 4)) + list(range(33, 154, 20))
    hkpt = "blocks.3.hook_resid_pre"
    ckpt_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(
        range(1000, 143000 + 1, 1000)
    )
    sae_dict_1 = {
        0: "checkpoints/sdtcw4c4/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        1: "pipe/jyv0qvm3/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        2: "pipe/uc92ff0h/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        3: "pipe/qafpifmt/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        4: "pipe/vrstj8zf/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        5: "pipe/hc83lcft/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        6: "pipe/v5wro145/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        7: "pipe/nbsiw9o2/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        8: "pipe/4wlpgqwg/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        9: "pipe/v24fvblq/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        10: "pipe/176d50pa/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        11: "pipe/mstensun/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        12: "pipe/v9s7zokh/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        13: "pipe/0amh94id/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        15: "pipe/y45cyyp2/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        17: "pipe/nfbu5g37/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        19: "pipe/tfx7bn2w/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        21: "pipe/kgtbda1v/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        23: "pipe/zykvk9z0/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        25: "pipe/8j6wjlop/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        27: "pipe/0bk3ykh0/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        29: "pipe/d7bb3g8c/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        31: "pipe/6rvb9pg/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        33: "pipe/kp7f5hbr/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        53: "pipe/hnf5u63c/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        73: "pipe/qxs2hhfw/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        93: "pipe/7vu7uvtw/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        113: "pipe/wa0i4z4b/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        133: "pipe/zad2k1so/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
        153: "pipe/b19kk6hj/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt"
    }
    data = load_dataset("NeelNanda/pile-10k", split="train")
elif model_type=="stanford-gpt2-small-a":
    ckpt_range = list(range(0, 11, 1)) + list(range(11, 31, 5)) + list(range(31, 181, 10))+list(range(181, 608, 30))+[608]
    hkpt = "blocks.5.hook_resid_pre"
    ckpt_list = (
            list(range(0, 100, 10))
            + list(range(100, 2000, 50))
            + list(range(2000, 20000, 100))
            + list(range(20000, 400000 + 1, 1000))
    )
    sae_dict_1 = {
        0: "checkpoints/uhss3qpb/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        1: "pipe/88bs7kse/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        2: "pipe/67r88wh5/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        3: "pipe/maii41z3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        4: "pipe/xd1re02u/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        5: "pipe/ms61ywp0/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        6: "pipe/val3dskf/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        7: "pipe/jbl8seww/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        8: "pipe/vumagfnz/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        9: "pipe/tb9oniyb/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        10: "pipe/ibs3te7q/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        11: "pipe/k10vj8iq/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        16: "pipe/8j54tzzq/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        21: "pipe/wuvzy67f/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        26: "pipe/53yeg92i/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        31: "pipe/jlkdgjk3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        41: "pipe/qakrqvrd/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        51: "pipe/vkvp7icm/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        61: "pipe/fwpgvhrz/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        71: "pipe/mukgsepw/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        81: "pipe/w69jl1lo/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        91: "pipe/rjzbhsd4/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        101: "pipe/9j8bc2re/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        111: "pipe/vp79is1t/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        121: "pipe/a0x7mkkm/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        131: "pipe/1tp0gfq9/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        141: "pipe/f71pmc8f/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        151: "pipe/1386o8fd/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        161: "pipe/bptyo7of/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        171: "pipe/8k5f83bu/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        181: "pipe/62otpds3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        211: "pipe/8662efxr/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        241: "pipe/d4u2ex96/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        271: "pipe/bpvva1ib/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        301: "pipe/5piu43ah/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        331: "pipe/hq6t556x/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        361: "pipe/2nck6hak/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        391: "pipe/uub7muyg/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        421: "pipe/bcpmx6p4/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        451: "pipe/tf9ufdgo/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        481: "pipe/r8uj6s2b/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        511: "pipe/2cdgtwrg/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        541: "pipe/l7ox7dce/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        571: "pipe/dfwhow9y/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        601: "pipe/a344cev5/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
        608: "pipe/e9f3k2o0/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt"
    }
    data = load_dataset("stas/openwebtext-10k", split="train")
elif model_type=="stanford-gpt2-medium-a":
    ckpt_range = list(range(0, 11, 1)) + list(range(11, 31, 5)) + list(range(31, 181, 10))+list(range(181, 608, 30))
    hkpt = "blocks.6.hook_resid_pre"
    ckpt_list = (
            list(range(0, 100, 10))
            + list(range(100, 2000, 50))
            + list(range(2000, 20000, 100))
            + list(range(20000, 400000 + 1, 1000))
    )
    sae_dict_1 = {
        0: "checkpoints/7rxmy17f/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        1: "pipe/16wyhaoz/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        2: "pipe/p2cw59an/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        3: "pipe/zmxcinfz/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        4: "pipe/2ynxrti7/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        5: "pipe/2xujl2pt/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        6: "pipe/37siz8xc/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        7: "pipe/rt6lreu1/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        8: "pipe/rop5gel4/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        9: "pipe/quebq5gi/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        10: "pipe/uz5e5pkl/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        11: "pipe/zrx12187/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        16: "pipe/f3l0t000/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        21: "pipe/9w9k6ok5/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        26: "pipe/326spdwy/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        31: "pipe/79gklkc0/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        41: "pipe/lntphptg/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        51: "pipe/jyr6t8ku/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        61: "pipe/7ga3t5ep/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        71: "pipe/3mc26bka/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        81: "pipe/oywu8v3g/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        91: "pipe/jjivvyni/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        101: "pipe/x82nz5s7/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        111: "pipe/tq5mo3qa/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        121: "pipe/7cam9ckk/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        131: "pipe/j922e6ey/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        141: "pipe/fc6ggbzk/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        151: "pipe/mh5n9l5w/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        161: "pipe/uicu5u66/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        171: "pipe/b1fq5b7d/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        181: "pipe/4edztew3/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        211: "pipe/o9z1fp4p/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        241: "pipe/k9czgdws/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        271: "pipe/2dn6zxdv/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        301: "pipe/z2x63qmb/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        331: "pipe/dt2bubb0/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        361: "pipe/0o397mfj/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        391: "pipe/7t3ycthf/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        421: "pipe/ww74jtvm/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        451: "pipe/tilbcxue/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        481: "pipe/vi662lrb/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        511: "pipe/sazeb3jc/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        541: "pipe/36v15b2z/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        571: "pipe/v5gag7br/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        601: "pipe/cmzi2vhb/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt",
        608: "pipe/lkdiza4e/final_sae_group_stanford-gpt2-medium-a_blocks.6.hook_resid_pre_65536.pt"
    }
    data = load_dataset("stas/openwebtext-10k", split="train")
else:
    raise ValueError(f"Invalid model type: {model_type}")

print(f"Processing checkpoints: {ckpt_range}")
print(f"model_type: {model_type}")

# Load and tokenize data
model = HookedTransformer.from_pretrained(model_type).to(device)
tokenized_data = utils.tokenize_and_concatenate(data, model.tokenizer, max_length=128)
tokenized_data = tokenized_data.shuffle(22)
all_tokens = tokenized_data["tokens"]

# Function to calculate average non-self similarity
def calculate_non_self_similarity(activations, top_k):
    act_vec = activations[:top_k]
    cosine_similarity_matrix = F.cosine_similarity(
        act_vec.unsqueeze(1),
        act_vec.unsqueeze(0),
        dim=2
    ).cpu().detach().numpy()
    mask = np.eye(len(act_vec), dtype=bool)
    non_self_similarities = cosine_similarity_matrix[~mask]
    return non_self_similarities.mean()

# Gather random similarities across checkpoints
average_random_similarities = []
token_set_number = 100
token_indices = np.random.choice(all_tokens.shape[0], token_set_number, replace=False)
random_indices = torch.randint(0, all_tokens.shape[1], (token_set_number,))

# Process checkpoints
for ckpt_num in tqdm(ckpt_range, desc="Processing Checkpoints"):
    print(f"Processing checkpoint: {ckpt_num}")
    model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
    model_df_wrapper = TransformerLensWrapper(model_df, hkpt)

    selected_tokens = all_tokens[token_indices].to(device)
    saes = SAEGroup.load_from_pretrained(sae_dict_1[ckpt_num])
    sae = saes.autoencoders[0].to(device)
    acts = get_features_activations(model_wrapper=model_df_wrapper, tokens=selected_tokens, encoder=sae)
    act_vec = acts[torch.arange(token_set_number), random_indices]
    average_random_similarity = calculate_non_self_similarity(act_vec, token_set_number)
    average_random_similarities.append((ckpt_num, average_random_similarity))

# Process features
feature_similarities = {idx: [] for idx in feature_indices}
for feature_idx in tqdm(feature_indices, desc="Processing Features"):
    topk_seqs_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_seqs.pt"
    topk_feats_path = f"features_{model_type}/feature_{feature_idx}_topk_{top_k}_toknum_{token_num}_feats.pt"

    if os.path.exists(topk_seqs_path) and os.path.exists(topk_feats_path):
        topk_seqs = torch.load(topk_seqs_path)
        topk_feats = torch.load(topk_feats_path)
    else:
        raise ValueError(f"No saved topk_seqs found for feature {feature_idx}. Run generate_feat_seqs.py first.")

    flattened_feats = topk_feats.view(-1)
    topk_vals, flat_topk_indices = torch.topk(flattened_feats, top_k)
    topk_2d_indices = torch.stack([flat_topk_indices // 128, flat_topk_indices % 128], dim=1)

    for ckpt_num in tqdm(ckpt_range, desc="Processing Checkpoints"):
        model_df = HookedTransformer.from_pretrained(model_type, checkpoint_index=ckpt_num).to(device)
        model_df_wrapper = TransformerLensWrapper(model_df, hkpt)
        saes = SAEGroup.load_from_pretrained(sae_dict_1[ckpt_num])
        sae = saes.autoencoders[0].to(device)
        acts = get_features_activations(model_wrapper=model_df_wrapper, tokens=topk_seqs, encoder=sae)
        act_vec = acts[topk_2d_indices[:, 0], topk_2d_indices[:, 1]]
        average_similarity = calculate_non_self_similarity(act_vec, top_k)
        feature_similarities[feature_idx].append((ckpt_num, average_similarity))

# Plotting
plt.figure(figsize=PLOT_CONFIG['figure.figsize'])

for i, (feature_idx, similarities) in enumerate(feature_similarities.items()):
    ckpts, avg_sims = zip(*similarities)
    baseline_similarities = [dict(average_random_similarities)[ckpt] for ckpt in ckpts]
    diff_sims = [sim - baseline for sim, baseline in zip(avg_sims, baseline_similarities)]
    x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]
    plt.plot(x_values, diff_sims, linestyle='-', linewidth=LINE_WIDTH, label=f"{feature_names[i]}")

# Plot random baseline as a reference
ckpts, random_avg_sims = zip(*average_random_similarities)
x_values = [ckpt_list[ckpt] + 1 for ckpt in ckpts]
plt.plot(x_values, np.zeros_like(random_avg_sims), linestyle='--', linewidth=LINE_WIDTH, label="Baseline (Random Sampled)")

# Adjust plot settings
plt.xscale('log')
plt.xlabel('Training Steps (log scale)')
plt.ylabel('Progressive Measure')
plt.ylim(-0.2, 1)
# plt.title(f'{model_type}, Hook Point: {hkpt}')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(False)
plt.tight_layout()

# Save and show
random_save_id = np.random.randint(1000)
plt.savefig(f"sae_similarity_diff_{model_type}_{hkpt}_{random_save_id}.svg")
plt.show()
