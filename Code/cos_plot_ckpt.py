import torch
import numpy as np
import plotly.express as px
from sae_training.sae_group import SAEGroup

# Centralized style settings
style_settings = {
    "font_family": "Arial",  # Font family
    "font_size": 50,         # Base font size
    "title_font_size": 60,   # Title font size
    "axis_title_font_size": 52,
    "tick_font_size": 46,    # Tick label font size
    "line_width": 6,         # Default line width
    "plot_width": 1200,      # Plot width
    "plot_height": 800,      # Plot height
    "margin_top": 80,        # Top margin for the plot
    "margin_bottom": 50,     # Bottom margin for the plot
    "margin_left": 50,       # Left margin for the plot
    "margin_right": 50,      # Right margin for the plot
}

# Paths to pretrained autoencoder checkpoints
# sae_dict = {
#     0: "checkpoints/uhss3qpb/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     1: "pipe/88bs7kse/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     2: "pipe/67r88wh5/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     3: "pipe/maii41z3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     4: "pipe/xd1re02u/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     5: "pipe/ms61ywp0/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     6: "pipe/val3dskf/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     7: "pipe/jbl8seww/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     8: "pipe/vumagfnz/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     9: "pipe/tb9oniyb/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     10: "pipe/ibs3te7q/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     11: "pipe/k10vj8iq/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     16: "pipe/ozld8zcb/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     21: "pipe/wuvzy67f/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     26: "pipe/53yeg92i/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     31: "pipe/jlkdgjk3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     41: "pipe/qakrqvrd/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     51: "pipe/vkvp7icm/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     61: "pipe/fwpgvhrz/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     71: "pipe/mukgsepw/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     81: "pipe/w69j1llo/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     91: "pipe/rjzbhsd4/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     101: "pipe/9j8bc2re/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     111: "pipe/vp79is1t/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     121: "pipe/a0x7mkkm/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     131: "pipe/1tp0gfq9/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     141: "pipe/f71pmc8f/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     151: "pipe/1386o8fd/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     161: "pipe/bptyo7of/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     171: "pipe/8k5f83bu/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     181: "pipe/62otpds3/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     211: "pipe/8662efxr/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     241: "pipe/d4u2ex96/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     271: "pipe/bpvva1ib/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     301: "pipe/5piu43ah/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     331: "pipe/hqt556x/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     361: "pipe/2nck6hak/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     391: "pipe/uub7muyg/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     421: "pipe/bcpmx6p4/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     451: "pipe/tf9ufdgo/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     481: "pipe/r8uj6s2b/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     511: "pipe/2cdgtwrg/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     541: "pipe/l7ox7dce/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     571: "pipe/dfwhow9y/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     601: "pipe/a344cev5/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt",
#     608: "pipe/e9f3k2o0/final_sae_group_stanford-gpt2-small-a_blocks.5.hook_resid_pre_49152.pt"
# }

# sae_dict = {
#     0: "checkpoints/921e0mgk/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     1: "pipe/52lieq0l/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     2: "pipe/18ekwo2k/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     3: "pipe/epmdboe7/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     4: "pipe/l2no0yut/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     5: "pipe/e7shgo9p/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     6: "pipe/ebs8r6qr/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     7: "pipe/vx8xb2k3/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     8: "pipe/v07opmp2/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     9: "pipe/io568vlx/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     10: "pipe/trbwvb9d/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     11: "pipe/x9qb3koe/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     13: "pipe/59zu9leh/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     15: "pipe/hpt1cy9i/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     17: "pipe/2i4w2133/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     19: "pipe/bo3lc5so/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     21: "pipe/fqohwi3s/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     23: "pipe/490ovm5t/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     25: "pipe/dvk5euee/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     27: "pipe/0nvrwgo9/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     29: "pipe/r4k7r6y6/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     31: "pipe/ein8fps8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     33: "pipe/io4vov51/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     53: "pipe/y4x2bx9y/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     73: "pipe/ib1hrafl/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     93: "pipe/3oih20d8/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     113: "pipe/nh42ylwt/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     133: "pipe/30737iqa/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt",
#     153: "pipe/07u89qd0/final_sae_group_pythia-410m-deduped_blocks.4.hook_resid_pre_65536.pt"
# }

# sae_dict = {
#         0: "checkpoints/lgbb13td/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         1: "pipe/8q8t8xpw/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         2: "pipe/tnu76xgi/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         3: "pipe/0043k90y/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         4: "pipe/gxg4e8d2/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         5: "pipe/if2frw09/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         6: "pipe/eb3j4739/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         7: "pipe/omju93gv/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         8: "pipe/qz9weoem/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         9: "pipe/vmlbirof/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         10: "pipe/5fo4imbk/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         11: "pipe/k3xy71u9/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         12: "pipe/85na2fvx/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         13: "pipe/ve7q8qar/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         15: "pipe/9dlhhzrw/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         17: "pipe/913fpjov/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         19: "pipe/c8ca4dc6/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         21: "pipe/6rnb2ffm/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         23: "pipe/gs4gqm2x/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         25: "pipe/0xi7t5bi/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         27: "pipe/mjgaqocj/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         29: "pipe/wy1ohgw8/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         31: "pipe/2xdhe64r/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         33: "pipe/hgdsrhs4/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         53: "pipe/dl4ij188/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         73: "pipe/yx04fkry/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         93: "pipe/1y2oepxv/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         113: "pipe/kbwcbnkh/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         133: "pipe/flovd4dt/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt",
#         153: "pipe/lig6rc8y/final_sae_group_pythia-160m-deduped_blocks.4.hook_resid_pre_49152.pt"
#     }

# sae_dict = {
#         0: "checkpoints/sdtcw4c4/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         1: "pipe/jyv0qvm3/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         2: "pipe/uc92ff0h/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         3: "pipe/qafpifmt/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         4: "pipe/vrstj8zf/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         5: "pipe/hc83lcft/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         6: "pipe/v5wro145/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         7: "pipe/nbsiw9o2/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         8: "pipe/4wlpgqwg/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         9: "pipe/v24fvblq/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         10: "pipe/176d50pa/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         11: "pipe/mstensun/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         12: "pipe/v9s7zokh/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         13: "pipe/0amh94id/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         15: "pipe/y45cyyp2/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         17: "pipe/nfbu5g37/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         19: "pipe/tfx7bn2w/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         21: "pipe/kgtbda1v/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         23: "pipe/zykvk9z0/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         25: "pipe/8j6wjlop/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         27: "pipe/0bk3ykh0/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         29: "pipe/d7bb3g8c/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         31: "pipe/6rvb9pg/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         33: "pipe/kp7f5hbr/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         53: "pipe/hnf5u63c/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         73: "pipe/qxs2hhfw/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         93: "pipe/7vu7uvtw/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         113: "pipe/wa0i4z4b/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         133: "pipe/zad2k1so/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt",
#         153: "pipe/b19kk6hj/final_sae_group_pythia-1.4b-deduped_blocks.3.hook_resid_pre_131072.pt"
#     }
sae_dict = sae_dict={
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
# Set random seed for reproducibility
np.random.seed(42)

# Specify the IDs and paths for the two SAEs
# 0,151,301,451
sae1_id = 301

sae1_path = sae_dict[sae1_id]
sae2_id = 608
sae2_path = sae_dict[sae2_id]

# Load the autoencoders
sae1 = SAEGroup.load_from_pretrained(sae1_path)
sae2 = SAEGroup.load_from_pretrained(sae2_path)
sae1 = sae1.autoencoders[0].to("cuda")
sae2 = sae2.autoencoders[0].to("cuda")

# =========================
# Plot 1: Cosine Similarity
# =========================

# Extract decoder weights for both SAEs (each shape: [d_hidden, d_in])
W_dec1 = sae1.W_dec
W_dec2 = sae2.W_dec

# Calculate cosine similarity between decoder weights of sae1 and sae2
cosine_sims = (W_dec1 * W_dec2).sum(dim=-1) / (W_dec1.norm(dim=-1) * W_dec2.norm(dim=-1))

# Visualize the cosine similarity with a histogram
fig1 = px.histogram(
    cosine_sims.to(torch.float32).detach().cpu().numpy(),
    log_y=True,  # Sets the y-axis to log scale
    range_x=[-1, 1],  # Sets the x-axis range from -1 to 1
    nbins=100,  # Adjust this value to change the number of bins
    labels={"value": "Cosine similarity of decoder vectors between SAEs"},
)

# Update layout with centralized settings and plot size
fig1.update_layout(
    title=dict(
        text=f"<b>SAE {sae1_id} and SAE {sae2_id}</b>",
        font=dict(family=style_settings["font_family"], size=style_settings["title_font_size"]),
        
        x=0.5,  # Center-align the title
        y=0.98, # Adjust title position slightly
    ),
    font=dict(
        family=style_settings["font_family"],
        size=style_settings["font_size"],
    ),
    width=style_settings["plot_width"],   # Set plot width
    height=style_settings["plot_height"], # Set plot height
    margin=dict(
        t=style_settings["margin_top"],
        b=style_settings["margin_bottom"],
        l=style_settings["margin_left"],
        r=style_settings["margin_right"],
    ),
    showlegend=False,
)

# Update axis titles and tick labels with centralized settings
fig1.update_xaxes(
    title_text="Cosine Similarity",
    title_font=dict(size=style_settings["axis_title_font_size"]),
    tickfont=dict(size=style_settings["tick_font_size"]),
)
fig1.update_yaxes(
    title_text="Number of Latents (log scale)",
    title_font=dict(size=style_settings["axis_title_font_size"]),
    tickfont=dict(size=style_settings["tick_font_size"]),
)

# Save the cosine similarity figure as an SVG file
fig1.write_image(f"cosine_similarity_sae_{sae1_id}_sae_{sae2_id}.svg")