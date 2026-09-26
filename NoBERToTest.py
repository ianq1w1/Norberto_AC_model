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
from config import ENCODER_NAME as _DEFAULT_ENCODER, CHECKPOINT_PATH as _DEFAULT_CHECKPOINT
from config import ACTION_TAG_NAMES, SLOT_TAG_NAMES, ESQUEMA_SLOTS_POR_COMANDO

# permite sobrescrever via variavel de ambiente, mas usa o config.py como padrao
ENCODER_NAME = os.getenv("NORBERTO_MODEL", _DEFAULT_ENCODER)
CHECKPOINT_PATH = os.getenv("NORBERTO_CHECKPOINT", _DEFAULT_CHECKPOINT)

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

        comandos = predict(model, tokenizer, texto, ACTION_TAG_NAMES, SLOT_TAG_NAMES, device=DEVICE,
                            esquema_slots=ESQUEMA_SLOTS_POR_COMANDO)

        if not comandos:
            print("  Nenhum comando detectado\n")
            continue

        for c in comandos:
            print(f"  -> {c['comando']}({c['slots']})")
        print()


if __name__ == "__main__":
    main()