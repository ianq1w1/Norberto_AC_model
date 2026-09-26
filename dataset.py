"""
dataset_generator.py

Gera o dataset sintetico de treino combinando templates (verbos x objetos x
salas x formatos de temperatura), em vez de listar exemplos na mao um por um.
O objetivo e cobrir MUITO mais variacao de como as pessoas realmente falam,
pra o modelo aprender o padrao geral em vez de decorar frases especificas.

Uso:
    from dataset_generator import gerar_dataset
    RAW_EXAMPLES = gerar_dataset()

Ajuste as constantes N_* la embaixo pra controlar o tamanho final do dataset.
"""

import itertools
import random

SEED = 123
random.seed(SEED)

# ---------------------------------------------------------------------------
# Vocabularios base - EDITE AQUI pra ampliar a variedade
# ---------------------------------------------------------------------------

VERBOS_LIGAR = ["ligue", "ligar", "liga", "ativa", "ativar", "acende", "acender"]
VERBOS_DESLIGAR = ["desligue", "desligar", "desliga", "desativa", "desativar", "apaga", "apagar", "para"]
OBJETOS_AR = ["o ar", "o ar condicionado", "ar", "ar condicionado", "ar-condicionado"]

VERBOS_TEMP = [
    "mudar temperatura para", "ajustar temperatura para", "coloca a temperatura em",
    "define a temperatura para", "configura a temperatura para", "muda a temperatura pra", "mude a temperatura para",
    "altere a temperatura para", "alterar a temperatura para", "coloque a temperatura em"
]

SAUDACOES = ["oi", "ola", "bom dia", "boa tarde", "boa noite", "e ai", "opa", "salve"]
DESPEDIDAS = ["tchau", "ate mais", "falou", "ate logo", "obrigado", "valeu", "ate amanha"]

SEPARADORES_MULTIACAO = [" e ", " ", ", ", " e tambem ", " depois ", " em seguida "]

# prefixos pra mencionar a sala JUNTO com um comando de temperatura - cobre o
# padrao "no ar da sala X, mude a temperatura para Y" (sala antes do verbo,
# numa oracao separada), que e uma estrutura de frase diferente da que
# ajustar_temp cobria sozinho antes (so verbo + valor, sem sala nenhuma)
PREFIXOS_SALA_TEMP = ["no ar da sala", "no ar condicionado da sala", "na sala"]

# mesma ideia, mas no PLURAL - pro caso de varias salas recebendo a mesma
# temperatura (ex: "nas salas B15, A18, mude a temperatura para 22 graus")
PREFIXOS_MULTI_SALA_TEMP = [
    "nas salas", "no ar das salas", "no ar condicionado das salas",
    "na sala", "no ar da sala", "nas salas:", "na sala:",
]

# graus + formato de escrita (varia justamente pra nao depender so de "X graus")
GRAUS_VALORES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30]
FORMATOS_TEMPERATURA = ["{g} graus", "{g}°", "{g}°c", "{g} graus celsius"]

# codigos de sala: mistura letra+numero (a maioria dos exemplos manuais que tinhamos)
# com numero puro (o formato que estava falhando antes)
_LETRAS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
SALAS_LETRA_NUMERO = [f"{l}{n}" for l in random.sample(_LETRAS, 12) for n in random.sample(range(1, 50), 4)]
SALAS_NUMERO_PURO = [str(n) for n in random.sample(range(100, 999), 20)]
SALAS = SALAS_LETRA_NUMERO + SALAS_NUMERO_PURO

# ---------------------------------------------------------------------------
# Controle de tamanho do dataset final - AJUSTE AQUI
# ---------------------------------------------------------------------------

N_SEM_SALA_POR_INTENT = 20      # quantas combinacoes verbo+objeto sem sala, por intent (ligar/desligar)
N_COM_SALA_POR_INTENT = 60      # quantas combinacoes verbo+objeto+sala, por intent
N_MULTI_SALA_POR_INTENT = 40    # quantos exemplos com LISTA de varias salas, por intent (ligar/desligar)
N_TEMPERATURA = 40              # quantas combinacoes verbo+valor de temperatura (sem mencionar sala)
N_TEMPERATURA_COM_SALA = 60     # quantas combinacoes verbo+valor+1 sala (ex: "no ar da sala X, mude para Y")
N_TEMPERATURA_MULTI_SALA = 40   # quantas combinacoes verbo+valor+VARIAS salas (ex: "nas salas X, Y, mude para Z")
N_MULTIACAO = 300               # quantos exemplos combinados de 2 acoes
FRACAO_VARIACAO_CASE = 0.15     # chance de gerar uma copia MAIUSCULA/minuscula/Capitalizada de cada exemplo


def span_por_valor(texto: str, valor: str, tipo: str):
    inicio = texto.index(valor)  # da erro se `valor` nao existir em `texto` - avisa bug cedo
    return (inicio, inicio + len(valor), tipo)


# ---------------------------------------------------------------------------
# Geracao dos exemplos de 1 acao so
# ---------------------------------------------------------------------------

def _gerar_ligar_desligar(intent: str, verbos: list, n_sem_sala: int, n_com_sala: int):
    exemplos = []

    combos_sem_sala = list(itertools.product(verbos, OBJETOS_AR))
    random.shuffle(combos_sem_sala)
    for verbo, objeto in combos_sem_sala[:n_sem_sala]:
        frase = f"{verbo} {objeto}"
        exemplos.append((frase, [(0, len(frase), intent)], []))

    combos_com_sala = list(itertools.product(verbos, OBJETOS_AR, SALAS))
    random.shuffle(combos_com_sala)
    for verbo, objeto, sala in combos_com_sala[:n_com_sala]:
        frase = f"{verbo} {objeto} da sala {sala}"
        slot = span_por_valor(frase, sala, "sala")
        exemplos.append((frase, [(0, len(frase), intent)], [slot]))

    return exemplos


def _gerar_ligar_desligar_multi_sala(intent: str, verbos: list, n_total: int):
    """
    Gera frases com VARIAS salas pra UMA SO acao de ligar/desligar (ex: "ligue
    o ar das salas B15, A18, F20" ou "desligue o ar da sala: B04, F04 e A08"),
    com cada codigo marcado como seu proprio span de slot "sala" (nao um span
    so cobrindo a lista inteira). Varia entre "da sala"/"das salas" (singular
    e plural, ja que na fala as pessoas nem sempre usam o plural certo) e
    com/sem dois-pontos antes da lista.
    """
    exemplos = []
    combos_verbo_objeto = list(itertools.product(verbos, OBJETOS_AR))
    prefixos = ["das salas", "da sala", "das salas:", "da sala:"]

    for _ in range(n_total):
        verbo, objeto = random.choice(combos_verbo_objeto)
        prefixo = random.choice(prefixos)
        k = random.randint(2, 4)
        codigos = random.sample(SALAS, k)

        if random.random() < 0.5:
            lista_str = ", ".join(codigos)
        else:
            lista_str = ", ".join(codigos[:-1]) + " e " + codigos[-1]

        frase = f"{verbo} {objeto} {prefixo} {lista_str}"
        slots = [span_por_valor(frase, codigo, "sala") for codigo in codigos]
        exemplos.append((frase, [(0, len(frase), intent)], slots))

    return exemplos


def _gerar_ajustar_temp(n_total: int):
    exemplos = []
    valores_formatados = [
        fmt.format(g=g) for g in GRAUS_VALORES for fmt in FORMATOS_TEMPERATURA
    ]
    combos = list(itertools.product(VERBOS_TEMP, valores_formatados))
    random.shuffle(combos)
    for verbo, valor in combos[:n_total]:
        frase = f"{verbo} {valor}"
        slot = span_por_valor(frase, valor, "temperatura")
        exemplos.append((frase, [(0, len(frase), "ajustar_temp")], [slot]))
    return exemplos


def _gerar_ajustar_temp_com_sala(n_total: int):
    """
    Cobre o padrao "no ar da sala X, mude a temperatura para Y" e variacoes -
    a sala aparece numa oracao separada, ANTES ou DEPOIS do verbo, ao contrario
    de _gerar_ajustar_temp (que nunca menciona sala). Gera os 2 slots (sala e
    temperatura) associados a UMA SO acao ajustar_temp.
    """
    exemplos = []
    valores_formatados = [
        fmt.format(g=g) for g in GRAUS_VALORES for fmt in FORMATOS_TEMPERATURA
    ]
    combos = list(itertools.product(PREFIXOS_SALA_TEMP, SALAS, VERBOS_TEMP, valores_formatados))
    random.shuffle(combos)

    for prefixo, sala, verbo, valor in combos[:n_total]:
        virgula = random.choice([", ", " "])
        sala_primeiro = random.random() < 0.5

        if sala_primeiro:
            # "no ar da sala B14, altere a temperatura para 22 graus"
            frase = f"{prefixo} {sala}{virgula}{verbo} {valor}"
        else:
            # "altere a temperatura para 22 graus no ar da sala B14"
            frase = f"{verbo} {valor}{virgula}{prefixo} {sala}"

        slot_sala = span_por_valor(frase, sala, "sala")
        slot_temp = span_por_valor(frase, valor, "temperatura")
        exemplos.append((frase, [(0, len(frase), "ajustar_temp")], [slot_sala, slot_temp]))

    return exemplos


def _gerar_ajustar_temp_multi_sala(n_total: int):
    """
    Cobre "nas salas B15, A18, F20, mude a temperatura para 22 graus" - a
    mesma temperatura aplicada a VARIAS salas na mesma acao. Gera 1 span de
    temperatura + N spans de sala (um por codigo), todos associados a uma
    unica acao ajustar_temp. Antes essa combinacao (multi-sala + ajustar_temp)
    reaproveitava por engano a funcao de ligar/desligar, gerando frases sem
    valor de temperatura nenhum - essa funcao corrige isso, gerando a
    estrutura certa (verbo + valor + lista de salas).
    """
    exemplos = []
    valores_formatados = [
        fmt.format(g=g) for g in GRAUS_VALORES for fmt in FORMATOS_TEMPERATURA
    ]

    for _ in range(n_total):
        verbo = random.choice(VERBOS_TEMP)
        valor = random.choice(valores_formatados)
        prefixo = random.choice(PREFIXOS_MULTI_SALA_TEMP)

        k = random.randint(2, 4)
        codigos = random.sample(SALAS, k)
        if random.random() < 0.5:
            lista_str = ", ".join(codigos)
        else:
            lista_str = ", ".join(codigos[:-1]) + " e " + codigos[-1]

        virgula = random.choice([", ", " "])
        sala_primeiro = random.random() < 0.5

        if sala_primeiro:
            # "nas salas B15, A18, mude a temperatura para 22 graus"
            frase = f"{prefixo} {lista_str}{virgula}{verbo} {valor}"
        else:
            # "mude a temperatura para 22 graus nas salas B15, A18"
            frase = f"{verbo} {valor}{virgula}{prefixo} {lista_str}"

        slot_temp = span_por_valor(frase, valor, "temperatura")
        slots_sala = [span_por_valor(frase, codigo, "sala") for codigo in codigos]
        exemplos.append((frase, [(0, len(frase), "ajustar_temp")], [slot_temp] + slots_sala))

    return exemplos


def _variar_case(exemplos: list, fracao: float):
    """
    Pra cada exemplo, com probabilidade `fracao`, adiciona uma copia extra em
    MAIUSCULO, minusculo ou Com A Primeira Letra Maiuscula. Como essas
    transformacoes preservam o TAMANHO da string (nao mudam quantos
    caracteres tem cada palavra), os spans de acao/slot continuam validos
    sem precisar recalcular nada - so trocamos o texto, mantendo os indices.
    """
    variados = list(exemplos)  # mantem todos os originais (minusculo, como gerados)

    for texto, acoes, slots in exemplos:
        if random.random() < fracao:
            variados.append((texto.upper(), acoes, slots))
        if random.random() < fracao:
            texto_capitalizado = texto[:1].upper() + texto[1:]
            variados.append((texto_capitalizado, acoes, slots))

    return variados


def _gerar_saudacao_despedida():
    exemplos = []
    for frase in SAUDACOES:
        exemplos.append((frase, [(0, len(frase), "saudacao")], []))
    for frase in DESPEDIDAS:
        exemplos.append((frase, [(0, len(frase), "despedida")], []))
    return exemplos


def gerar_exemplos_simples():
    exemplos = []
    exemplos += _gerar_ligar_desligar("ligar_ar", VERBOS_LIGAR, N_SEM_SALA_POR_INTENT, N_COM_SALA_POR_INTENT)
    exemplos += _gerar_ligar_desligar("desligar_ar", VERBOS_DESLIGAR, N_SEM_SALA_POR_INTENT, N_COM_SALA_POR_INTENT)
    exemplos += _gerar_ligar_desligar_multi_sala("ligar_ar", VERBOS_LIGAR, N_MULTI_SALA_POR_INTENT)
    exemplos += _gerar_ligar_desligar_multi_sala("desligar_ar", VERBOS_DESLIGAR, N_MULTI_SALA_POR_INTENT)
    exemplos += _gerar_ajustar_temp(N_TEMPERATURA)
    exemplos += _gerar_ajustar_temp_com_sala(N_TEMPERATURA_COM_SALA)
    exemplos += _gerar_ajustar_temp_multi_sala(N_TEMPERATURA_MULTI_SALA)
    exemplos += _gerar_saudacao_despedida()
    exemplos = _variar_case(exemplos, FRACAO_VARIACAO_CASE)
    return exemplos


# ---------------------------------------------------------------------------
# Geracao dos exemplos MULTI-ACAO (combina 2 exemplos simples num so)
# ---------------------------------------------------------------------------

def _combinar_dois(exemplo_a, exemplo_b, separador):
    texto_a, acoes_a, slots_a = exemplo_a
    texto_b, acoes_b, slots_b = exemplo_b

    texto = f"{texto_a}{separador}{texto_b}"
    offset_b = len(texto_a) + len(separador)

    acoes = list(acoes_a) + [(s + offset_b, e + offset_b, label) for (s, e, label) in acoes_b]
    slots = list(slots_a) + [(s + offset_b, e + offset_b, label) for (s, e, label) in slots_b]

    return texto, acoes, slots


def gerar_combos_multiacao(exemplos_simples: list, n_combos: int):
    combos = []
    for _ in range(n_combos):
        exemplo_a = random.choice(exemplos_simples)
        exemplo_b = random.choice(exemplos_simples)
        separador = random.choice(SEPARADORES_MULTIACAO)
        combos.append(_combinar_dois(exemplo_a, exemplo_b, separador))
    return combos


# ---------------------------------------------------------------------------
# Funcao principal
# ---------------------------------------------------------------------------

def gerar_dataset():
    exemplos_simples = gerar_exemplos_simples()
    exemplos_multiacao = gerar_combos_multiacao(exemplos_simples, N_MULTIACAO)

    dataset = exemplos_simples + exemplos_multiacao
    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    dataset = gerar_dataset()
    print(f"Total de exemplos gerados: {len(dataset)}")
    print(f"  Salas disponiveis: {len(SALAS)} ({len(SALAS_LETRA_NUMERO)} letra+numero, {len(SALAS_NUMERO_PURO)} numero puro)")

    print("\nExemplos do padrao ajustar_temp + multi-sala (novo):")
    exemplos_multi_sala_temp = [
        e for e in dataset
        if e[1] and e[1][0][2] == "ajustar_temp" and sum(1 for s in e[2] if s[2] == "sala") >= 2
    ]
    for texto, acoes, slots in random.sample(exemplos_multi_sala_temp, min(3, len(exemplos_multi_sala_temp))):
        print(f"  texto:  {texto!r}")
        print(f"  acoes:  {acoes}")
        print(f"  slots:  {slots}\n")