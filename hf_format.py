"""
export_to_hf_format.py

Converte o checkpoint treinado (formato antigo, lse_norberto.LSEJointModel
como nn.Module puro) para o formato "Hub-ready" (modeling_lse_norberto.
LSEJointModel, PreTrainedModel), pronto pra trust_remote_code=True.

Depois de rodar isso, use o publish_to_hf.py normalmente pra subir a pasta
./hf_model atualizada pro Hugging Face Hub.

Rodar com:
    python export_to_hf_format.py
"""

import shutil

import torch
from transformers import AutoTokenizer

from config import ENCODER_NAME, CHECKPOINT_PATH, ACTION_TAG_NAMES, SLOT_TAG_NAMES, USAR_HEADS_SEMANTICAS
from configuration_lse_norberto import LSENorBERToConfig
from modeling_lse_norberto import LSEJointModel as LSEJointModelHub

OUTPUT_DIR = "./hf_model"

# --- monta o config novo, ja com auto_map apontando pras classes customizadas ---
config = LSENorBERToConfig(
    encoder_name=ENCODER_NAME,
    num_action_tags=len(ACTION_TAG_NAMES),
    num_slot_tags=len(SLOT_TAG_NAMES),
    action_tags=ACTION_TAG_NAMES,
    slot_tags=SLOT_TAG_NAMES,
    usar_heads_semanticas=USAR_HEADS_SEMANTICAS,
)
config.auto_map = {
    "AutoConfig": "configuration_lse_norberto.LSENorBERToConfig",
    "AutoModel": "modeling_lse_norberto.LSEJointModel",
}

# --- instancia o modelo novo (Hub-ready) e carrega os pesos do checkpoint antigo ---
model = LSEJointModelHub(config)
state_dict_antigo = torch.load(CHECKPOINT_PATH, map_location="cpu")
model.load_state_dict(state_dict_antigo)  # mesmos nomes de submodulo, entao carrega direto
model.eval()

# --- salva no formato padrao do transformers (config.json + safetensors) ---
model.save_pretrained(OUTPUT_DIR, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
tokenizer.save_pretrained(OUTPUT_DIR)

# --- copia os arquivos de codigo customizado pra dentro da pasta a ser publicada ---
shutil.copy("configuration_lse_norberto.py", OUTPUT_DIR)
shutil.copy("modeling_lse_norberto.py", OUTPUT_DIR)

print(f"Exportado para {OUTPUT_DIR}/ - rode publish_to_hf.py em seguida para subir ao Hub")