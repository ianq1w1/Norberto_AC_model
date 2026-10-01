"""
modeling_lse_norberto.py

Versao "Hub-ready" do LSEJointModel: herda de PreTrainedModel (em vez de
nn.Module puro), pra funcionar com AutoModel.from_pretrained(repo_id,
trust_remote_code=True). Self-contained - inclui aqui dentro tambem as
funcoes de decodificacao (decode_bio_spans, associar_slots_a_acoes, etc),
porque este arquivo precisa ir sozinho pro repositorio do Hub, sem depender
de importar lse_norberto.py de outro lugar.

Este arquivo precisa ir JUNTO dos pesos no repositorio do Hugging Face Hub.
"""

import itertools

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel

try:
    from .model_configuration import LSENorBERToConfig  # quando carregado via trust_remote_code
except ImportError:
    from model_configuration import LSENorBERToConfig  # quando usado localmente, pasta plana


# ---------------------------------------------------------------------------
# Decodificacao (mesma logica do lse_norberto.py original)
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


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class LSEJointModel(PreTrainedModel):
    """Encoder compartilhado (NorBERTo) + 2 heads de token classification (BIO)."""

    config_class = LSENorBERToConfig

    def __init__(self, config: LSENorBERToConfig):
        super().__init__(config)
        self.encoder = AutoModel.from_pretrained(config.encoder_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(config.dropout)
        self.action_tagging_head = nn.Linear(hidden_size, config.num_action_tags)
        self.slot_tagging_head = nn.Linear(hidden_size, config.num_slot_tags)

        self.post_init()

    def _init_weights(self, module):
        # o encoder ja vem pre-treinado (carregado via from_pretrained acima);
        # as heads usam a inicializacao padrao do nn.Linear, sem necessidade
        # de customizar nada aqui.
        pass

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        encoder_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            encoder_kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**encoder_kwargs)
        sequence_output = self.dropout(outputs.last_hidden_state)

        action_tag_logits = self.action_tagging_head(sequence_output)
        slot_tag_logits = self.slot_tagging_head(sequence_output)

        return {
            "action_tag_logits": action_tag_logits,
            "slot_tag_logits": slot_tag_logits,
        }

    @torch.no_grad()
    def detectar_comandos(self, tokenizer, text: str, limiar_confianca: float = 0.0,
                           esquema_slots: dict = None):
        """
        Metodo de conveniencia: ja usa os action_tags/slot_tags guardados no
        proprio config, entao so precisa do tokenizer e do texto.

        Uso:
            model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model.detectar_comandos(tokenizer, "ligue o ar da sala B15")
        """
        self.eval()
        device = next(self.parameters()).device
        encoded = tokenizer(
            text, return_tensors="pt", truncation=True, return_special_tokens_mask=True,
        ).to(device)
        outputs = self(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])

        action_probs_full = torch.softmax(outputs["action_tag_logits"][0], dim=-1)
        slot_probs_full = torch.softmax(outputs["slot_tag_logits"][0], dim=-1)

        action_ids = action_probs_full.argmax(dim=-1).tolist()
        slot_ids = slot_probs_full.argmax(dim=-1).tolist()
        action_confidences = action_probs_full.max(dim=-1).values.tolist()
        slot_confidences = slot_probs_full.max(dim=-1).values.tolist()

        action_tag_names = self.config.action_tags
        slot_tag_names = self.config.slot_tags

        special_mask = encoded["special_tokens_mask"][0].tolist()
        action_o_id = action_tag_names.index("O")
        slot_o_id = slot_tag_names.index("O")
        action_ids = [action_o_id if m == 1 else t for t, m in zip(action_ids, special_mask)]
        slot_ids = [slot_o_id if m == 1 else t for t, m in zip(slot_ids, special_mask)]

        action_id2tag = dict(enumerate(action_tag_names))
        slot_id2tag = dict(enumerate(slot_tag_names))

        action_spans = decode_bio_spans(action_ids, action_id2tag)
        slot_spans = mesclar_spans_proximos(decode_bio_spans(slot_ids, slot_id2tag))

        action_spans = filtrar_por_confianca(action_spans, action_confidences, limiar_confianca)
        slot_spans = filtrar_por_confianca(slot_spans, slot_confidences, limiar_confianca)

        for span in action_spans + slot_spans:
            token_slice = encoded["input_ids"][0][span["start"]: span["end"] + 1]
            span["text"] = tokenizer.decode(token_slice, skip_special_tokens=True).strip()

        comandos_com_listas = associar_slots_a_acoes(action_spans, slot_spans)
        comandos = expandir_comandos_multivalor(comandos_com_listas)

        if esquema_slots is not None:
            comandos = padronizar_slots(comandos, esquema_slots)

        return comandos