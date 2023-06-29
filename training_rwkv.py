#@title Google Drive Options { display-mode: "form" }
save_models_to_drive = True #@param {type:"boolean"}
drive_mount = '/content/drive' #@param {type:"string"}
output_dir = 'rwkv-v4neo-rnn-pile-tuning' #@param {type:"string"}
tuned_model_name = 'tuned-python' #@param {type:"string"}


if save_models_to_drive:
    from google.colab import drive
    drive.mount(drive_mount)

output_path = f"{drive_mount}/MyDrive/{output_dir}" if save_models_to_drive else f"/content/{output_dir}"
os.makedirs(f"{output_path}/{tuned_model_name}", exist_ok=True)
os.makedirs(f"{output_path}/base_models/", exist_ok=True)

print(f"Saving models to {output_path}")

!nvidia-smi

# Commented out IPython magic to ensure Python compatibility.
!git clone https://github.com/blinkdl/RWKV-LM
repo_dir = "/content/RWKV-LM/RWKV-v4neo"
# %cd $repo_dir

!pip install transformers pytorch-lightning==1.9 deepspeed wandb ninja



#@title Base Model Options
#@markdown Using any of the listed options will download the checkpoint from huggingface

base_model_name = "RWKV-4-Pile-169M" #@param ["RWKV-4-Pile-1B5", "RWKV-4-Pile-430M", "RWKV-4-Pile-169M"]
base_model_url = f"https://huggingface.co/BlinkDL/{base_model_name.lower()}"

if base_model_name == "RWKV-4-Pile-169M":
    n_layer = 12
    n_embd = 768
elif base_model_name == "RWKV-4-Pile-430M":
    n_layer = 24
    n_embd = 1024
elif base_model_name == "RWKV-4-Pile-1B5":
    n_layer = 24
    n_embd = 2048

!git lfs clone $base_model_url

from glob import glob
base_model_path = glob(f"{base_model_name.lower()}/{base_model_name}*.pth")[0]

print(f"Using {base_model_path} as base")


#@title Training Data Options
#@markdown `input_file` should be the path to a single file that contains the text you want to fine-tune with.
#@markdown Either upload a file to this notebook instance or reference a file in your Google drive.

import numpy as np
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file=f'{repo_dir}/20B_tokenizer.json')

input_file = "/content/drive/MyDrive/pandas_train.txt" #@param {type:"string"}
output_file = 'train.npy'

print(f'Tokenizing {input_file} (VERY slow. please wait)')

data_raw = open(input_file, encoding="utf-8").read()
print(f'Raw length = {len(data_raw)}')

data_code = tokenizer.encode(data_raw)
print(f'Tokenized length = {len(data_code)}')

out = np.array(data_code, dtype='uint16')
np.save(output_file, out, allow_pickle=False)

"""## Training"""

#@title Begin Training with these Options { display-mode: "form" }
n_epoch = 100 #@param {type:"integer"}
epoch_save_frequency = 1 #@param {type:"integer"}
batch_size =  50 #@param {type:"integer"}
ctx_len = 384 #@param {type:"integer"}
precision = 'fp16' #@param ['fp16', 'bf16', 'bf32'] {type:"string"}

epoch_save_path = f"{output_path}/{tuned_model_name}"


!python train.py \
--load_model $base_model_path \
--wandb "" \
--proj_dir $output_dir \
--data_file  "train.npy" \
--data_type "numpy" \
--vocab_size 50277 \
--ctx_len $ctx_len \
--epoch_steps 1000 \
--epoch_count $n_epoch \
--epoch_begin 0 \
--epoch_save $epoch_save_frequency \
--micro_bsz 8 \
--n_layer $n_layer \
--n_embd $n_embd \
--pre_ffn 0 \
--head_qk 0 \
--lr_init 1e-5 \
--lr_final 1e-5 \
--warmup_steps 0 \
--beta1 0.9 \
--beta2 0.999 \
--adam_eps 1e-8 \
--accelerator gpu \
--devices 1 \
--precision $precision \
--strategy deepspeed_stage_2 \
--grad_cp 0
