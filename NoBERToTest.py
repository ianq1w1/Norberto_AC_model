"""
test_norberto_cli.py

Loop de terminal para testar o LSEJointModel (action-tagging) interativamente.
So faz sentido rodar DEPOIS de ter um checkpoint treinado (train_lse_norberto.py).

Rodar com:
    python test_norberto_cli.py
"""

import os
import torch
from transformers import AutoTokenizer

from lse_norberto import LSEJointModel, predict
from config import ENCODER_NAME, CHECKPOINT_PATH, INTENT_NAMES, SLOT_TYPE_NAMES, ACTION_TAG_NAMES, SLOT_TAG_NAMES

ENCODER_NAME = os.getenv("NORBERTO_MODEL", "Itau-Unibanco/NorBERTo-large")
CHECKPOINT_PATH = os.getenv("NORBERTO_CHECKPOINT", "./checkpoints/lse_norberto_v2.pt")

# precisam ser EXATAMENTE os mesmos nomes/ordem usados no train_lse_norberto.py
INTENT_NAMES = ["ligar_ar", "desligar_ar", "ajustar_temp", "saudacao", "despedida"]
SLOT_TYPE_NAMES = ["sala", "temperatura"]

ACTION_TAG_NAMES = ["O"] + [f"{p}-{intent}" for intent in INTENT_NAMES for p in ("B", "I")]
SLOT_TAG_NAMES = ["O"] + [f"{p}-{slot}" for slot in SLOT_TYPE_NAMES for p in ("B", "I")]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def carregar_modelo():
    print(f"Carregando tokenizer e encoder ({ENCODER_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = LSEJointModel(
        encoder_name=ENCODER_NAME,
        num_action_tags=len(ACTION_TAG_NAMES),
        num_slot_tags=len(SLOT_TAG_NAMES),
    ).to(DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print(f"Checkpoint carregado de {CHECKPOINT_PATH}")
    else:
        print(
            f"AVISO: nenhum checkpoint encontrado em '{CHECKPOINT_PATH}'. "
            "Rode train_lse_norberto.py primeiro - sem isso as heads estao "
            "com pesos aleatorios e a saida nao vai fazer sentido nenhum."
        )

    model.eval()
    return tokenizer, model


def main():
    tokenizer, model = carregar_modelo()
    print("\nDigite uma mensagem (ou 'sair' para encerrar):\n")

    while True:
        texto = input("> ").strip()
        if texto.lower() in ("sair", "exit", "quit"):
            break
        if not texto:
            continue

        comandos = predict(model, tokenizer, texto, ACTION_TAG_NAMES, SLOT_TAG_NAMES, device=DEVICE)

        if not comandos:
            print("  Nenhum comando detectado\n")
            continue

        for c in comandos:
            print(f"  -> {c['comando']}({c['slots']})")
        print()


if __name__ == "__main__":
    main()