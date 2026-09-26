"""
config.py

Configuracao compartilhada entre train_lse_norberto.py, test_norberto_cli.py e
norberto_service.py. Mude os comandos/slots AQUI - os outros arquivos importam
daqui, entao ficam sempre sincronizados entre si.
"""

ENCODER_NAME = "Itau-Unibanco/NorBERTo-large"
CHECKPOINT_PATH = "./checkpoints/lse_norberto_v2.pt"

INTENT_NAMES = ["ligar_ar", "desligar_ar", "ajustar_temp", "saudacao", "despedida"]
SLOT_TYPE_NAMES = ["sala", "temperatura"]


ACTION_TAG_NAMES = ["O"]

for intent in INTENT_NAMES:
    for p in ("B", "I"):
        ACTION_TAG_NAMES.append(f"{p}-{intent}")

SLOT_TAG_NAMES = ["O"] 

for slot in SLOT_TYPE_NAMES:
    for p in ("B", "I"):
        SLOT_TAG_NAMES.append(f"{p}-{slot}")

# quais slots cada comando DEVE ter na saida, mesmo que o modelo nao tenha
# detectado (nesse caso vira "" em vez de a chave simplesmente nao existir)
ESQUEMA_SLOTS_POR_COMANDO = {
    "ligar_ar": ["sala"],
    "desligar_ar": ["sala"],
    "ajustar_temp": ["sala", "temperatura"],
    "saudacao": [],
    "despedida": [],
}