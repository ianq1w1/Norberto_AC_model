"""
config.py

Configuracao compartilhada entre nobertoTrain.py, NorBERToTeste.py. 
Mude os comandos/slots AQUI - os outros arquivos importam
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

USAR_HEADS_SEMANTICAS = True  # precisa ser IGUAL no treino e na inferencia

# quais slots cada comando DEVE ter na saida, mesmo que o modelo nao tenha
# detectado (nesse caso vira "" em vez de a chave simplesmente nao existir)
ESQUEMA_SLOTS_POR_COMANDO = {
    "ligar_ar": ["sala"],
    "desligar_ar": ["sala"],
    "ajustar_temp": ["sala", "temperatura"],
    "saudacao": [],
    "despedida": [],
}


DESCRICOES_ACOES = {
    "ligar_ar": "ligar, ativar ou acender o ar-condicionado de uma sala",
    "desligar_ar": "desligar, apagar, desativar ou parar o ar-condicionado de uma sala",
    "ajustar_temp": "mudar, ajustar, definir ou configurar a temperatura do ar-condicionado",
    "saudacao": "cumprimentar, dizer oi, ola ou bom dia",
    "despedida": "se despedir, dizer tchau, ate mais ou agradecer",
}
 
DESCRICOES_SLOTS = {
    "sala": "codigo ou numero identificador de uma sala ou ambiente",
    "temperatura": "valor numerico de temperatura em graus",
}

def gerar_descricoes_por_tag(tag_names: list, descricoes_base: dict) -> list:
    """
    Expande as descricoes base (por rotulo) pra uma lista alinhada com a
    ordem de ACTION_TAG_NAMES/SLOT_TAG_NAMES (com B-/I-/O), pra codificar
    cada tag do esquema BIO com uma frase propria.
    """
    descricoes = []
    for tag in tag_names:
        if tag == "O":
            descricoes.append("nenhum comando ou campo relevante, texto neutro")
        else:
            prefixo, label = tag.split("-", 1)
            base = descricoes_base.get(label, label)
            if prefixo == "B":
                descricoes.append(f"inicio de: {base}")
            else:
                descricoes.append(f"continuacao de: {base}")
    return descricoes