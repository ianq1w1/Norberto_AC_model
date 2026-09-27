"""
avaliar_modelo.py

Roda um conjunto curado de casos de teste contra o modelo treinado e compara
com o resultado ESPERADO, calculando acerto por categoria (equivalente ao
O-Acc do artigo do LSE-NLU: so conta como acerto se TODOS os comandos e
slots da frase baterem, nem mais nem menos).

Objetivo: parar de testar frase por frase manualmente no terminal e ter um
numero objetivo de "quantos % dos casos conhecidos passam", alem de saber
EM QUAL categoria o modelo ainda esta fraco.

Rodar com:
    python avaliar_modelo.py
"""

import torch
from transformers import AutoTokenizer

from lse_norberto import LSEJointModel, predict
from config import ENCODER_NAME, CHECKPOINT_PATH, ACTION_TAG_NAMES, SLOT_TAG_NAMES, ESQUEMA_SLOTS_POR_COMANDO

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIAR_CONFIANCA = 0.0  # ajuste se quiser testar com filtro de confianca ativo


# ---------------------------------------------------------------------------
# Casos de teste - cada um com o resultado ESPERADO (schema completo, com ""
# nos slots que o comando nao deveria ter). Agrupados por categoria, pra dar
# pra ver onde o modelo esta fraco especificamente.
# ---------------------------------------------------------------------------

def _cmd(comando, sala="", temperatura=""):
    """Helper pra montar um comando esperado no formato ja padronizado pelo schema."""
    slots = {}
    if comando in ("ligar_ar", "desligar_ar"):
        slots = {"sala": sala}
    elif comando == "ajustar_temp":
        slots = {"sala": sala, "temperatura": temperatura}
    return {"comando": comando, "slots": slots}


CASOS_DE_TESTE = [
    # --- basico: 1 acao, 1 sala ---
    {"categoria": "basico", "texto": "ligue o ar da sala B15",
     "esperado": [_cmd("ligar_ar", sala="B15")]},
    {"categoria": "basico", "texto": "desligue o ar da sala F03",
     "esperado": [_cmd("desligar_ar", sala="F03")]},
    {"categoria": "basico", "texto": "mude a temperatura para 22 graus na sala B33",
     "esperado": [_cmd("ajustar_temp", sala="B33", temperatura="22 graus")]},

    # --- saudacao / despedida ---
    {"categoria": "saudacao_despedida", "texto": "oi",
     "esperado": [_cmd("saudacao")]},
    {"categoria": "saudacao_despedida", "texto": "bom dia",
     "esperado": [_cmd("saudacao")]},
    {"categoria": "saudacao_despedida", "texto": "tchau",
     "esperado": [_cmd("despedida")]},

    # --- multi-acao: 2 acoes diferentes, com conectivo ---
    {"categoria": "multi_acao", "texto": "desligue o ar da sala B08 e ligue o ar da sala F04",
     "esperado": [_cmd("desligar_ar", sala="B08"), _cmd("ligar_ar", sala="F04")]},
    {"categoria": "multi_acao", "texto": "ligue o ar da sala F04 e desligue o ar da sala B02",
     "esperado": [_cmd("ligar_ar", sala="F04"), _cmd("desligar_ar", sala="B02")]},

    # --- multi-acao: 3 acoes, incluindo ajustar_temp ---
    {"categoria": "multi_acao", "texto": "desligue o ar da B08, ligue o ar da F04 e mude a temperatura da F07 para 27 graus",
     "esperado": [_cmd("desligar_ar", sala="B08"), _cmd("ligar_ar", sala="F04"),
                  _cmd("ajustar_temp", sala="F07", temperatura="27 graus")]},

    # --- multi-sala: mesma acao, varias salas ---
    {"categoria": "multi_sala", "texto": "ligue o ar das salas B15 e F09",
     "esperado": [_cmd("ligar_ar", sala="B15"), _cmd("ligar_ar", sala="F09")]},
    {"categoria": "multi_sala", "texto": "desligue o ar das salas: B04, F04 e L04",
     "esperado": [_cmd("desligar_ar", sala="B04"), _cmd("desligar_ar", sala="F04"),
                  _cmd("desligar_ar", sala="L04")]},
    {"categoria": "multi_sala", "texto": "ligue o ar condicionado das seguintes salas B08, B09, B10",
     "esperado": [_cmd("ligar_ar", sala="B08"), _cmd("ligar_ar", sala="B09"),
                  _cmd("ligar_ar", sala="B10")]},

    # --- multi-sala + ajustar_temp junto ---
    {"categoria": "multi_sala_temp", "texto": "nas salas B15 e A18, mude a temperatura para 22 graus",
     "esperado": [_cmd("ajustar_temp", sala="B15", temperatura="22 graus"),
                  _cmd("ajustar_temp", sala="A18", temperatura="22 graus")]},

    # --- elipse: "da sala X e da Y" ---
    {"categoria": "elipse", "texto": "ligue o ar condicionado da sala B04 e da B08",
     "esperado": [_cmd("ligar_ar", sala="B04"), _cmd("ligar_ar", sala="B08")]},

    # --- formatos de temperatura ---
    {"categoria": "formato_temperatura", "texto": "ajuste a temperatura para 21°",
     "esperado": [_cmd("ajustar_temp", temperatura="21°")]},
    {"categoria": "formato_temperatura", "texto": "coloque a temperatura em 25°c",
     "esperado": [_cmd("ajustar_temp", temperatura="25°c")]},
    {"categoria": "formato_temperatura", "texto": "mude a temperatura para 18 graus celsius",
     "esperado": [_cmd("ajustar_temp", temperatura="18 graus celsius")]},

    # --- salas sem letra (numero puro) ---
    {"categoria": "sala_numero_puro", "texto": "ligue o ar da sala 305",
     "esperado": [_cmd("ligar_ar", sala="305")]},

    # --- variacao de caixa ---
    {"categoria": "case_variado", "texto": "LIGUE O AR DA SALA B15",
     "esperado": [_cmd("ligar_ar", sala="B15")]},
    {"categoria": "case_variado", "texto": "Desligue o ar da sala F09",
     "esperado": [_cmd("desligar_ar", sala="F09")]},

    # --- linguagem informal / girias (categoria de limitacao conhecida) ---
    {"categoria": "informal_giria", "texto": "eae bot, desligue o ar da sala B08",
     "esperado": [_cmd("desligar_ar", sala="B08")]},
    {"categoria": "informal_giria", "texto": "bot ligue o ar da sala F10 por favor",
     "esperado": [_cmd("ligar_ar", sala="F10")]},
]


# ---------------------------------------------------------------------------
# Comparacao entre esperado e obtido (ignora ordem, compara como conjunto)
# ---------------------------------------------------------------------------

def _normalizar_comando(c):
    slots_norm = tuple(sorted((k, str(v).strip().lower()) for k, v in c["slots"].items()))
    return (c["comando"], slots_norm)


def bateu(esperado, obtido):
    return sorted(_normalizar_comando(c) for c in esperado) == sorted(_normalizar_comando(c) for c in obtido)


# ---------------------------------------------------------------------------
# Execucao
# ---------------------------------------------------------------------------

def carregar_modelo():
    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    model = LSEJointModel(
        encoder_name=ENCODER_NAME,
        num_action_tags=len(ACTION_TAG_NAMES),
        num_slot_tags=len(SLOT_TAG_NAMES),
    ).to(DEVICE)

    import os
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint nao encontrado em {CHECKPOINT_PATH} - rode train_lse_norberto.py primeiro"
        )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    return tokenizer, model


def main():
    tokenizer, model = carregar_modelo()

    resultados_por_categoria = {}
    total_ok = 0

    print(f"Rodando {len(CASOS_DE_TESTE)} casos de teste...\n")

    for caso in CASOS_DE_TESTE:
        obtido = predict(
            model, tokenizer, caso["texto"], ACTION_TAG_NAMES, SLOT_TAG_NAMES,
            device=DEVICE, limiar_confianca=LIMIAR_CONFIANCA, esquema_slots=ESQUEMA_SLOTS_POR_COMANDO,
        )
        ok = bateu(caso["esperado"], obtido)
        total_ok += int(ok)

        cat = caso["categoria"]
        resultados_por_categoria.setdefault(cat, {"ok": 0, "total": 0})
        resultados_por_categoria[cat]["total"] += 1
        resultados_por_categoria[cat]["ok"] += int(ok)

        marcador = "OK  " if ok else "FALHOU"
        print(f"[{marcador}] ({cat}) {caso['texto']!r}")
        if not ok:
            print(f"         esperado: {caso['esperado']}")
            print(f"         obtido:   {obtido}")

    print("\n" + "=" * 60)
    print(f"RESULTADO GERAL (equivalente ao O-Acc): {total_ok}/{len(CASOS_DE_TESTE)} "
          f"({100 * total_ok / len(CASOS_DE_TESTE):.1f}%)")
    print("=" * 60)
    print("\nPor categoria:")
    for cat, r in sorted(resultados_por_categoria.items()):
        pct = 100 * r["ok"] / r["total"]
        print(f"  {cat:<25} {r['ok']}/{r['total']} ({pct:.0f}%)")


if __name__ == "__main__":
    main()