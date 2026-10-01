"""
train_lse_norberto.py

Treino do LSEJointModel na versao "action-tagging": 2 heads de token
classification (BIO) compartilhando o encoder - uma pra segmentar spans de
acao/comando, outra pra segmentar spans de slot. Sem depender de conectivos.

Rodar com:
    python train_lse_norberto.py
"""

import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from safetensors.torch import save_file

from lse_norberto import LSEJointModel, compute_joint_loss, calcular_embeddings_dos_rotulos
from dataset import gerar_dataset
from config import (
    ENCODER_NAME, CHECKPOINT_PATH, ACTION_TAG_NAMES, SLOT_TAG_NAMES,
    gerar_descricoes_por_tag, DESCRICOES_ACOES, DESCRICOES_SLOTS, USAR_HEADS_SEMANTICAS,
)

# ---------------------------------------------------------------------------
# 1. Configuracao (nomes de comando/slot vem do config.py compartilhado)
# ---------------------------------------------------------------------------

ACTION_TAG2ID = {tag: i for i, tag in enumerate(ACTION_TAG_NAMES)}
SLOT_TAG2ID = {tag: i for i, tag in enumerate(SLOT_TAG_NAMES)}
IGNORE_TAG_ID = -100  # tokens especiais / padding ficam de fora da loss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 40
LEARNING_RATE = 3e-5       # encoder (ja pre-treinado, passo pequeno)
HEAD_LEARNING_RATE = 1e-3  # heads (parametros novos do zero, precisam de passo maior)
VAL_FRACTION = 0.2
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 2. Dataset bruto: gerado programaticamente por dataset.py
#    (edite as constantes N_* la pra controlar o tamanho do dataset)
# ---------------------------------------------------------------------------

RAW_EXAMPLES = gerar_dataset()

random.shuffle(RAW_EXAMPLES)
split_idx = int(len(RAW_EXAMPLES) * (1 - VAL_FRACTION))
train_examples = RAW_EXAMPLES[:split_idx]
val_examples = RAW_EXAMPLES[split_idx:]

print(f"Total de exemplos: {len(RAW_EXAMPLES)} ({len(train_examples)} treino / {len(val_examples)} validacao)")


# ---------------------------------------------------------------------------
# 3. Alinhamento de spans (char -> token) + Dataset do PyTorch
# ---------------------------------------------------------------------------

def alinhar_spans_para_bio(offsets_mapping, spans, tag2id):
    """
    offsets_mapping: lista de (char_inicio, char_fim) por token
    spans: lista de (char_inicio, char_fim, label)
    Generico - usado tanto pra spans de acao quanto de slot.

    IMPORTANTE: usa SOBREPOSICAO (overlap), nao contencao estrita. Muitos
    tokenizers incluem o espaco anterior no offset do primeiro token de uma
    palavra nova (ex: token " B" cobre 1 char antes do span anotado) - exigir
    contencao total fazia esse token cair fora e perder a letra/digito inicial
    do valor do slot (ex: "B14" virava so "14"). Sobreposicao resolve isso.
    """
    tags = [tag2id["O"]] * len(offsets_mapping)

    for token_idx, (tok_start, tok_end) in enumerate(offsets_mapping):
        if tok_start == tok_end:  # token especial (CLS, SEP, PAD)
            tags[token_idx] = IGNORE_TAG_ID

    for (span_start, span_end, label) in spans:
        primeiro_token_do_span = True
        for token_idx, (tok_start, tok_end) in enumerate(offsets_mapping):
            if tok_start == tok_end:  # token especial, nunca faz parte de um span
                continue

            sobrepoe = tok_start < span_end and tok_end > span_start
            if not sobrepoe:
                continue

            prefixo = "B" if primeiro_token_do_span else "I"
            tags[token_idx] = tag2id[f"{prefixo}-{label}"]
            primeiro_token_do_span = False

    return tags


class ActionSlotDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        texto, acoes, slots = self.examples[idx]

        encoded = self.tokenizer(
            texto, truncation=True, max_length=self.max_length, return_offsets_mapping=True,
        )

        action_tags = alinhar_spans_para_bio(encoded["offset_mapping"], acoes, ACTION_TAG2ID)
        slot_tags = alinhar_spans_para_bio(encoded["offset_mapping"], slots, SLOT_TAG2ID)

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "action_tag_labels": action_tags,
            "slot_tag_labels": slot_tags,
        }


def collate_fn(batch, tokenizer):
    max_len = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id

    input_ids, attention_mask, action_tags, slot_tags = [], [], [], []
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        action_tags.append(item["action_tag_labels"] + [IGNORE_TAG_ID] * pad_len)
        slot_tags.append(item["slot_tag_labels"] + [IGNORE_TAG_ID] * pad_len)

    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "action_tag_labels": torch.tensor(action_tags),
        "slot_tag_labels": torch.tensor(slot_tags),
    }


# ---------------------------------------------------------------------------
# 4. Loop de treino
# ---------------------------------------------------------------------------

def avaliar(model, dataloader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, _ = compute_joint_loss(outputs, batch, pad_tag_id=IGNORE_TAG_ID)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)

    # --- Label Semantic Expansion: computa os embeddings iniciais das tags a
    # partir das descricoes em config.py, ANTES do modelo principal existir ---
    action_label_embeddings_init = None
    slot_label_embeddings_init = None

    if USAR_HEADS_SEMANTICAS:
        encoder_para_descricoes = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)

        descricoes_acoes = gerar_descricoes_por_tag(ACTION_TAG_NAMES, DESCRICOES_ACOES)
        descricoes_slots = gerar_descricoes_por_tag(SLOT_TAG_NAMES, DESCRICOES_SLOTS)

        action_label_embeddings_init = calcular_embeddings_dos_rotulos(
            encoder_para_descricoes, tokenizer, descricoes_acoes, device=DEVICE
        )
        slot_label_embeddings_init = calcular_embeddings_dos_rotulos(
            encoder_para_descricoes, tokenizer, descricoes_slots, device=DEVICE
        )
        del encoder_para_descricoes

    model = LSEJointModel(
        encoder_name=ENCODER_NAME,
        num_action_tags=len(ACTION_TAG_NAMES),
        num_slot_tags=len(SLOT_TAG_NAMES),
        action_label_embeddings_init=action_label_embeddings_init,
        slot_label_embeddings_init=slot_label_embeddings_init,
    ).to(DEVICE)

    train_ds = ActionSlotDataset(train_examples, tokenizer)
    val_ds = ActionSlotDataset(val_examples, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=lambda b: collate_fn(b, tokenizer))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer))

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": LEARNING_RATE},
        {"params": model.action_tagging_head.parameters(), "lr": HEAD_LEARNING_RATE},
        {"params": model.slot_tagging_head.parameters(), "lr": HEAD_LEARNING_RATE},
    ])
    os.makedirs("./checkpoints", exist_ok=True)

    melhor_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss, _ = compute_joint_loss(outputs, batch, pad_tag_id=IGNORE_TAG_ID)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = avaliar(model, val_loader)

        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss <= melhor_val_loss:
            melhor_val_loss = val_loss

            # checkpoint "de trabalho" (formato antigo, pt) - usado pelo
            # test_norberto_cli.py, avaliar_modelo.py, norberto_service.py
            torch.save(model.state_dict(), CHECKPOINT_PATH)

            # export direto pro formato Hub-ready (sem passar por
            # export_to_hf_format.py) - mantem o que voce ja tinha
            os.makedirs("./hf_model", exist_ok=True)

            save_file(model.state_dict(), "./hf_model/model.safetensors")
            tokenizer.save_pretrained("./hf_model")

            with open("./hf_model/config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "encoder_name": ENCODER_NAME,
                    "num_action_tags": len(ACTION_TAG_NAMES),
                    "num_slot_tags": len(SLOT_TAG_NAMES),
                    "action_tags": ACTION_TAG_NAMES,
                    "slot_tags": SLOT_TAG_NAMES,
                    "usar_heads_semanticas": USAR_HEADS_SEMANTICAS,
                }, f, ensure_ascii=False, indent=2)

            print(f"  -> novo melhor checkpoint salvo em {CHECKPOINT_PATH} e ./hf_model/")

    print(f"\nTreino concluido. Melhor val_loss: {melhor_val_loss:.4f}")


if __name__ == "__main__":
    main()