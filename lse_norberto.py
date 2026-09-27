"""
lse_norberto.py

Implementacao do LSE-NLU (encoder compartilhado + multiplas subtarefas treinadas
juntas, com a loss somada) adaptada para o NorBERTo, resolvendo o problema de
MULTIPLAS ACOES na mesma mensagem sem depender de conectivos ("e", virgula, etc).

A ideia central desta versao: em vez de uma head de intencao (classificacao da
frase inteira) separada de uma head de slot generica, usamos DUAS heads de
TOKEN CLASSIFICATION (BIO tagging) que compartilham o mesmo encoder:

  1. action_tagging_head: BIO tagging onde a propria tag JA E o nome do comando
     (ex: B-ligar_ar, I-ligar_ar, B-desligar_ar, I-desligar_ar, O). Isso segmenta
     a frase em "spans de acao" aprendidos pelo modelo a partir dos dados de
     treino - sem depender de palavras-chave de separacao.

  2. slot_tagging_head: BIO tagging comum para os tipos de slot (ex: B-sala,
     I-sala, B-temperatura, I-temperatura, O), igual ao que tinhamos antes.

Depois da inferencia, uma etapa de pos-processamento (fora do modelo) associa
cada slot ao span de acao em que ele cai, resolvendo o problema de "qual slot
pertence a qual comando" sem ambiguidade e sem qualquer regra de pontuacao.

Equivalencia com o artigo original: NLL(Theta) = loss_action + loss_slot, as
duas losses somadas sem peso, ambas propagando gradiente pelo mesmo encoder
Theta compartilhado - o mesmo principio das equacoes (1) e (2) do LSE-NLU.
"""

import itertools

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class LSEJointModel(nn.Module):
    """Encoder compartilhado (NorBERTo) + 2 heads de token classification (BIO)."""

    def __init__(self, encoder_name: str, num_action_tags: int, num_slot_tags: int,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.action_tagging_head = nn.Linear(hidden_size, num_action_tags)
        self.slot_tagging_head = nn.Linear(hidden_size, num_slot_tags)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (batch, seq_len, hidden)

        action_tag_logits = self.action_tagging_head(sequence_output)  # (batch, seq_len, num_action_tags)
        slot_tag_logits = self.slot_tagging_head(sequence_output)      # (batch, seq_len, num_slot_tags)

        return {
            "action_tag_logits": action_tag_logits,
            "slot_tag_logits": slot_tag_logits,
        }


def compute_joint_loss(outputs, batch, pad_tag_id: int = -100):
    """
    Equivalente as equacoes (1) e (2) do artigo original: soma das losses das
    2 subtarefas (action tagging + slot tagging), ambas compartilhando Theta.
    """
    ce = nn.CrossEntropyLoss(ignore_index=pad_tag_id)

    action_logits_flat = outputs["action_tag_logits"].view(-1, outputs["action_tag_logits"].size(-1))
    action_labels_flat = batch["action_tag_labels"].view(-1)
    loss_action = ce(action_logits_flat, action_labels_flat)

    slot_logits_flat = outputs["slot_tag_logits"].view(-1, outputs["slot_tag_logits"].size(-1))
    slot_labels_flat = batch["slot_tag_labels"].view(-1)
    loss_slot = ce(slot_logits_flat, slot_labels_flat)

    total_loss = loss_action + loss_slot  # equacao (2): soma simples, sem pesos

    return total_loss, {
        "loss_action": loss_action.item(),
        "loss_slot": loss_slot.item(),
        "total_loss": total_loss.item(),
    }


# ---------------------------------------------------------------------------
# Decodificacao: logits -> spans de acao/slot -> comandos estruturados
# ---------------------------------------------------------------------------

def decode_bio_spans(tag_ids, id2tag):
    """
    Converte uma sequencia de ids de tag BIO em uma lista de spans contiguos.
    Trata uma I- "orfa" (sem B- antes, ou com label diferente do span atual)
    como se fosse um B- novo - e uma heuristica comum pra tornar a decodificacao
    robusta a erros do modelo.
    """
    spans = []
    current = None

    for idx, tid in enumerate(tag_ids):
        tag = id2tag.get(tid, "O")

        if tag == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue

        prefix, label = tag.split("-", 1)

        if prefix == "B" or current is None or current["label"] != label:
            if current is not None:
                spans.append(current)
            current = {"start": idx, "end": idx, "label": label}
        else:  # prefix == "I" e continua o mesmo label do span atual
            current["end"] = idx

    if current is not None:
        spans.append(current)

    return spans


def mesclar_spans_proximos(spans, gap_maximo: int = 2):
    """
    Junta spans consecutivos do MESMO label que estao a poucos tokens de
    distancia um do outro (gap_maximo). Isso suaviza a fragmentacao causada
    por hesitacao do modelo entre tokens adjacentes (comum com pouco dado de
    treino) - nao resolve a causa raiz, so reduz o sintoma visivel.
    """
    if not spans:
        return spans

    spans_ordenados = sorted(spans, key=lambda s: s["start"])
    mesclados = [dict(spans_ordenados[0])]

    for span in spans_ordenados[1:]:
        anterior = mesclados[-1]
        gap = span["start"] - anterior["end"] - 1
        if span["label"] == anterior["label"] and gap <= gap_maximo:
            anterior["end"] = span["end"]
        else:
            mesclados.append(dict(span))

    return mesclados


def associar_slots_a_acoes(action_spans, slot_spans):
    """
    Associa cada slot ao span de acao correspondente:
      1. se o slot esta CONTIDO dentro do span de uma acao, associa a ela
      2. senao, associa a acao mais recente que comeca antes do slot
         (cobre casos onde o slot fica levemente fora do span, ex: "no 17"
         apos o fim estrito do span da acao)

    Cada tipo de slot guarda uma LISTA de valores (nao um valor unico) -
    isso permite capturar "sala B15, A18, F20" como 3 valores do mesmo tipo
    "sala" associados a UMA acao, sem um sobrescrever o outro.
    """
    resultado = [{"comando": acao["label"], "slots": {}, "_start": acao["start"], "_end": acao["end"]}
                 for acao in action_spans]

    for slot in slot_spans:
        candidatos_contidos = [
            r for r in resultado if r["_start"] <= slot["start"] and slot["end"] <= r["_end"]
        ]
        if candidatos_contidos:
            alvo = candidatos_contidos[0]
        else:
            anteriores = [r for r in resultado if r["_start"] <= slot["start"]]
            alvo = max(anteriores, key=lambda r: r["_start"]) if anteriores else None

        if alvo is not None:
            alvo["slots"].setdefault(slot["label"], []).append(slot["text"])

    for r in resultado:
        del r["_start"]
        del r["_end"]

    return resultado


def expandir_comandos_multivalor(comandos_com_listas):
    """
    Pega a saida de associar_slots_a_acoes (onde cada slot e uma LISTA de
    valores) e desdobra qualquer acao com mais de 1 valor no mesmo tipo de
    slot em VARIOS comandos separados - um por valor. Ex: 1 acao ligar_ar
    com slots={"sala": ["B15", "A18", "F20"]} vira 3 comandos ligar_ar,
    cada um com uma sala diferente. Slots com 1 valor so viram escalar
    normal (nao ficam como lista de 1 elemento no resultado final).
    """
    resultado = []

    for comando in comandos_com_listas:
        slots = comando["slots"]
        tipos_multivalor = [tipo for tipo, valores in slots.items() if len(valores) > 1]

        if not tipos_multivalor:
            slots_flat = {tipo: (valores[0] if valores else "") for tipo, valores in slots.items()}
            resultado.append({"comando": comando["comando"], "slots": slots_flat})
            continue

        listas_multivalor = [slots[tipo] for tipo in tipos_multivalor]
        for combinacao in itertools.product(*listas_multivalor):
            slots_expandido = {
                tipo: (valores[0] if valores else "")
                for tipo, valores in slots.items() if tipo not in tipos_multivalor
            }
            for tipo, valor in zip(tipos_multivalor, combinacao):
                slots_expandido[tipo] = valor
            resultado.append({"comando": comando["comando"], "slots": slots_expandido})

    return resultado


def calcular_confianca_media(span, confidences):
    """Media da confianca (probabilidade do argmax) dos tokens dentro do span."""
    valores = confidences[span["start"]: span["end"] + 1]
    return sum(valores) / len(valores) if valores else 0.0


def filtrar_por_confianca(spans, confidences, limiar: float):
    """Descarta spans cuja confianca media fica abaixo do limiar - sem isso,
    o argmax sempre "escolhe algo" mesmo quando o modelo esta essencialmente
    chutando entre opcoes quase empatadas."""
    if limiar <= 0:
        return spans
    return [s for s in spans if calcular_confianca_media(s, confidences) >= limiar]


def padronizar_slots(comandos, esquema_slots: dict):
    """
    Garante que cada comando tenha SEMPRE as mesmas chaves de slot, na mesma
    ordem, preenchendo com "" quando o modelo nao detectou aquele slot.
    `esquema_slots` e um dict tipo {"ajustar_temp": ["sala", "temperatura"],
    "ligar_ar": ["sala"], ...} definindo quais slots cada comando deveria ter.
    Isso evita que quem consome a API precise checar se a chave existe antes
    de usar - o formato fica previsivel independente do que foi detectado.
    """
    resultado = []
    for c in comandos:
        chaves_esperadas = esquema_slots.get(c["comando"], list(c["slots"].keys()))
        slots_padronizados = {chave: c["slots"].get(chave, "") for chave in chaves_esperadas}
        resultado.append({"comando": c["comando"], "slots": slots_padronizados})
    return resultado


@torch.no_grad()
def predict(model, tokenizer, text: str, action_tag_names, slot_tag_names, device: str = "cpu",
            limiar_confianca: float = 0.0, esquema_slots: dict = None):
    """Roda a frase inteira pelo modelo e devolve a lista de comandos estruturados.

    limiar_confianca: se > 0, descarta spans (de acao ou slot) cuja probabilidade
    media (softmax) fique abaixo desse valor.
    esquema_slots: se fornecido (ex: config.ESQUEMA_SLOTS_POR_COMANDO), padroniza
    as chaves de slot de cada comando pro schema esperado, preenchendo com "".
    """
    model.eval()
    encoded = tokenizer(
        text, return_tensors="pt", truncation=True, return_special_tokens_mask=True,
    ).to(device)
    outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])

    action_probs_full = torch.softmax(outputs["action_tag_logits"][0], dim=-1)
    slot_probs_full = torch.softmax(outputs["slot_tag_logits"][0], dim=-1)



    action_ids = action_probs_full.argmax(dim=-1).tolist()
    slot_ids = slot_probs_full.argmax(dim=-1).tolist()
    action_confidences = action_probs_full.max(dim=-1).values.tolist()
    slot_confidences = slot_probs_full.max(dim=-1).values.tolist()

    # tokens especiais (CLS/SEP/PAD) NUNCA foram treinados a prever nada -
    # a previsao do modelo ali e arbitraria (sempre foram IGNORE_TAG_ID no
    # treino). Forca essas posicoes pra "O" antes de decodificar, senao viram
    # spans fantasma (ex: uma acao "despedida" aparecendo em toda frase,
    # sempre com texto vazio, porque a posicao nao corresponde a texto nenhum)
    special_mask = encoded["special_tokens_mask"][0].tolist()
    action_o_id = action_tag_names.index("O")
    slot_o_id = slot_tag_names.index("O")
    action_ids = [action_o_id if m == 1 else t for t, m in zip(action_ids, special_mask)]
    slot_ids = [slot_o_id if m == 1 else t for t, m in zip(slot_ids, special_mask)]

    action_id2tag = dict(enumerate(action_tag_names))
    slot_id2tag = dict(enumerate(slot_tag_names))

    # NAO mesclar spans de acao: 2 ocorrencias legitimas do MESMO comando
    # separadas so por "e" ficam indistinguiveis de ruido de 1 token no meio
    # da mesma acao - mesclar aqui arriscava PERDER um comando real (pior
    # que so mostrar uma linha duplicada por ruido). Slots mesclar continua
    # ok, o risco de colisao la e bem mais raro.
    action_spans = decode_bio_spans(action_ids, action_id2tag)
    slot_spans = mesclar_spans_proximos(decode_bio_spans(slot_ids, slot_id2tag))

    action_spans = filtrar_por_confianca(action_spans, action_confidences, limiar_confianca)
    slot_spans = filtrar_por_confianca(slot_spans, slot_confidences, limiar_confianca)

   
    #print(f"  -> {[action_spans]}")
    #print(f"  -> {[slot_spans]}")
    # reconstroi o texto de cada span a partir dos proprios input_ids (lida bem
    # com subtokens/wordpieces, ao contrario de tentar juntar strings na mao)
    for span in action_spans + slot_spans:
        token_slice = encoded["input_ids"][0][span["start"]: span["end"] + 1]
        span["text"] = tokenizer.decode(token_slice, skip_special_tokens=True).strip()

    comandos_com_listas = associar_slots_a_acoes(action_spans, slot_spans)
    comandos = expandir_comandos_multivalor(comandos_com_listas)

    if esquema_slots is not None:
        comandos = padronizar_slots(comandos, esquema_slots)

    return comandos


# ---------------------------------------------------------------------------
# Exemplo de uso / loop de treino minimo (substitua pelos seus dados reais)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ENCODER_NAME = "Itau-Unibanco/NorBERTo-base"

    INTENT_NAMES = ["ligar_ar", "desligar_ar", "ajustar_temp"]
    SLOT_TYPE_NAMES = ["sala", "temperatura"]

    ACTION_TAG_NAMES = ["O"] + [f"{p}-{intent}" for intent in INTENT_NAMES for p in ("B", "I")]
    SLOT_TAG_NAMES = ["O"] + [f"{p}-{slot}" for slot in SLOT_TYPE_NAMES for p in ("B", "I")]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = LSEJointModel(
        encoder_name=ENCODER_NAME,
        num_action_tags=len(ACTION_TAG_NAMES),
        num_slot_tags=len(SLOT_TAG_NAMES),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

    # --- batch de exemplo (troque por um DataLoader real com seus dados rotulados) ---
    texto = "desligue o ar da sala A14 e ligue o ar da b24 no 17"
    encoded = tokenizer(texto, return_tensors="pt", truncation=True).to(device)
    seq_len = encoded["input_ids"].size(1)

    batch = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "action_tag_labels": torch.zeros(1, seq_len, dtype=torch.long, device=device),  # ids reais aqui
        "slot_tag_labels": torch.zeros(1, seq_len, dtype=torch.long, device=device),    # ids reais aqui
    }

    num_epochs = 1  # troque pelo numero real de epochs
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        loss, loss_breakdown = compute_joint_loss(outputs, batch)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch}: {loss_breakdown}")

    # exemplo de inferencia
    resultado = predict(model, tokenizer, texto, ACTION_TAG_NAMES, SLOT_TAG_NAMES, device=device)
    print(resultado)