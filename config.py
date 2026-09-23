"""
config.py

Configuracao compartilhada entre train_lse_norberto.py, test_norberto_cli.py e
norberto_service.py. Mude os comandos/slots AQUI - os outros arquivos importam
daqui, entao ficam sempre sincronizados entre si.
"""

ENCODER_NAME = "Itau-Unibanco/NorBERTo-large"  # confirme o path exato no Hugging Face
CHECKPOINT_PATH = "./checkpoints/lse_norberto_v2.pt"

INTENT_NAMES = ["ligar_ar", "desligar_ar", "ajustar_temp", "saudacao", "despedida"]
SLOT_TYPE_NAMES = ["sala", "temperatura"]

ACTION_TAG_NAMES = ["O"] + [f"{p}-{intent}" for intent in INTENT_NAMES for p in ("B", "I")]
SLOT_TAG_NAMES = ["O"] + [f"{p}-{slot}" for slot in SLOT_TYPE_NAMES for p in ("B", "I")]