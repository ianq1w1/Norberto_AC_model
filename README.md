# NorBERTo-Ar-condicionado

Fine-tuning do **NorBERTo-large** para Natural Language Understanding (NLU) aplicada ao controle de sistemas de ar-condicionado.

O modelo foi desenvolvido para identificar **múltiplas ações (comandos)** e **múltiplos slots (parâmetros)** dentro de uma única mensagem em português brasileiro.

> **Modelo:** `ianzeraA/NorBERTo-Ar-condicionado`
> **Modelo base:** `Itau-Unibanco/NorBERTo-large`

---

## Visão geral

Este modelo utiliza o **NorBERTo-large** como encoder compartilhado e adiciona duas heads independentes de **Token Classification** treinadas conjuntamente:

1. **Action Tagging Head** — identifica e segmenta ações/comandos.
2. **Slot Tagging Head** — identifica e segmenta os parâmetros associados aos comandos.

A arquitetura permite processar mensagens contendo múltiplas ações sem depender de conectivos específicos como `"e"`, vírgulas ou outras palavras usadas para separar comandos.

Por exemplo:

```text
"desligue o ar da sala A14 e ligue o ar da sala B24 em 17 graus"
```

pode ser interpretado como múltiplos comandos:

```json
[
  {
    "comando": "desligar_ar",
    "slots": {
      "sala": "A14"
    }
  },
  {
    "comando": "ligar_ar",
    "slots": {
      "sala": "B24",
      "temperatura": "17"
    }
  }
]
```

A separação das ações é realizada pelo próprio modelo através do **BIO tagging**, e não por regras baseadas em palavras de ligação.

---

## Arquitetura

A arquitetura utiliza um encoder compartilhado baseado no NorBERTo-large:

```text
                    Entrada
                       │
                       ▼
              ┌─────────────────┐
              │  NorBERTo-large │
              │     Encoder     │
              └────────┬────────┘
                       │
                Hidden States
                       │
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
     ┌────────────────┐  ┌────────────────┐
     │ Action Tagging │  │ Slot Tagging   │
     │      Head      │  │      Head      │
     └───────┬────────┘  └───────┬────────┘
             │                   │
             ▼                   ▼
          BIO Tags            BIO Tags
             │                   │
             └─────────┬─────────┘
                       ▼
                Pós-processamento
                       │
                       ▼
             Comandos estruturados
```

As duas heads compartilham o mesmo encoder e são treinadas simultaneamente.

A loss total utilizada durante o treinamento é:

```text
Loss_total = Loss_action + Loss_slot
```

ou seja, as losses das duas tarefas são somadas sem pesos adicionais.

---

## Action Tagging

A primeira head realiza classificação por token utilizando o esquema BIO.

As tags representam diretamente o tipo de comando.

Exemplo:

```text
"ligue o ar da sala A14"
```

pode produzir algo conceitualmente semelhante a:

```text
ligue    B-ligar_ar
o        O
ar       O
da       O
sala     O
A14      O
```

Para comandos compostos, múltiplos spans podem ser identificados na mesma entrada.

Exemplo:

```text
"desligue o ar da sala A14 e ligue o ar da B24"
```

O modelo pode identificar:

```text
B-desligar_ar ... 
        ...
B-ligar_ar ...
```

O objetivo é que cada ocorrência de uma ação seja representada como um span independente.

---

## Slot Tagging

A segunda head também utiliza BIO tagging, mas as tags representam tipos de parâmetros.

Exemplos de slots:

```text
sala
temperatura
```

Uma entrada como:

```text
"ligue o ar da sala A14 em 18 graus"
```

pode produzir spans correspondentes a:

```text
A14   → sala
18    → temperatura
```

Os slots são posteriormente associados às ações detectadas.

---

## Múltiplas ações

Uma das principais características deste modelo é permitir que uma única mensagem contenha várias ações.

Exemplo:

```text
"desligue o ar da sala A14, ligue o ar da B24 e ajuste a temperatura da C10 para 20 graus"
```

O modelo pode identificar múltiplos spans de ação:

```text
desligue → desligar_ar
ligue    → ligar_ar
ajuste   → ajustar_temp
```

e múltiplos slots:

```text
A14 → sala
B24 → sala
C10 → sala
20  → temperatura
```

O pós-processamento associa os slots às ações correspondentes.

Isso permite transformar uma mensagem natural em uma estrutura de comandos consumível por uma aplicação.

---

## Multi-value slots

O modelo também foi projetado para lidar com múltiplos valores do mesmo tipo de slot.

Por exemplo:

```text
"ligue o ar das salas A14, B24 e C10"
```

pode resultar em:

```json
[
  {
    "comando": "ligar_ar",
    "slots": {
      "sala": "A14"
    }
  },
  {
    "comando": "ligar_ar",
    "slots": {
      "sala": "B24"
    }
  },
  {
    "comando": "ligar_ar",
    "slots": {
      "sala": "C10"
    }
  }
]
```

A expansão dos valores é realizada durante o pós-processamento.

---

## Modelo base

O encoder utilizado neste fine-tuning é:

`Itau-Unibanco/NorBERTo-large`

O NorBERTo é uma família de modelos encoder baseados na arquitetura ModernBERT, treinados especificamente para português.

O presente repositório adiciona ao encoder duas heads específicas para a tarefa de NLU de comandos de ar-condicionado.

---

## Treinamento

O treinamento foi realizado com fine-tuning conjunto das duas tarefas:

```text
NorBERTo-large
      │
      ├── Action Token Classification
      │
      └── Slot Token Classification
```

### Configuração

Os principais parâmetros utilizados no treinamento incluem:

```text
Batch size:        8
Epochs:            20
Learning rate:     3e-5
Optimizer:         AdamW
Validation split:  20%
Seed:              42
Maximum length:    64 tokens
```

A melhor versão do modelo é selecionada de acordo com a menor `validation loss`.

---

## Dataset

O dataset utilizado no treinamento é **sintético e gerado programaticamente**.

Os exemplos contêm:

* texto de entrada;
* spans de ações;
* spans de slots;
* labels BIO correspondentes.

O alinhamento entre os spans de caracteres e os tokens é realizado utilizando os offsets fornecidos pelo tokenizer.

O treinamento utiliza `-100` para tokens especiais e padding, fazendo com que essas posições sejam ignoradas pela Cross Entropy Loss.

---

## Loss

Cada token é classificado independentemente pelas duas heads.

Para a head de ações:

```text
Loss_action = CrossEntropy(action_logits, action_labels)
```

Para a head de slots:

```text
Loss_slot = CrossEntropy(slot_logits, slot_labels)
```

A loss final é:

```text
Loss_total = Loss_action + Loss_slot
```

O gradiente das duas tarefas é propagado pelo encoder compartilhado.

---

## Pós-processamento

A saída das duas heads não é diretamente o resultado final.

Após a inferência, os BIO tags são convertidos em spans.

O pipeline é:

```text
Texto
  ↓
Tokenizer
  ↓
NorBERTo-large
  ↓
Action Head ──→ Action BIO spans
  │
  └────────────→ Slot Head ──→ Slot BIO spans
                              ↓
                    Associação slot → ação
                              ↓
                    Expansão multi-value
                              ↓
                  Comandos estruturados
```

O pós-processamento inclui:

* decodificação dos spans BIO;
* tratamento de tags `I-` órfãs;
* agrupamento de spans de slots próximos;
* associação de slots às ações;
* suporte a múltiplos valores do mesmo slot;
* filtragem opcional por confiança;
* padronização do schema de slots.

---


---
Resultado esperado, conceitualmente:

```json
[
  {
    "comando": "desligar_ar",
    "slots": {
      "sala": "A14"
    }
  },
  {
    "comando": "ligar_ar",
    "slots": {
      "sala": "B24"
    }
  }
]
```

---

## Confiança

A inferência permite utilizar um limiar de confiança opcional:

```python
resultado = predict(
    model,
    tokenizer,
    texto,
    ACTION_TAG_NAMES,
    SLOT_TAG_NAMES,
    limiar_confianca=0.70,
)
```

Quando um limiar maior que `0` é utilizado, spans cuja confiança média dos tokens fique abaixo do limite são descartados.

Isso pode ser útil para aplicações em que seja preferível não executar um comando quando a previsão estiver pouco confiável.

---

## Limitações

Este modelo foi desenvolvido especificamente para **NLU de comandos relacionados a ar-condicionado**.

Ele não deve ser considerado um modelo genérico de compreensão de linguagem.

As principais limitações incluem:

* domínio específico de comandos de climatização;
* dependência da distribuição e qualidade do dataset sintético;
* possíveis erros de segmentação BIO;
* possíveis ambiguidades na associação entre slots e ações;
* capacidade limitada para construções linguísticas não representadas nos dados de treinamento;
* limite de comprimento utilizado durante o treinamento;
* ausência de garantia de generalização para outros domínios.

Para utilização em produção, recomenda-se avaliar o modelo com dados reais representativos das mensagens que serão recebidas pelo sistema.

---

### `nobertoTrain.py`

Responsável pelo treinamento e validação do modelo.

### `dataset.py`

Responsável pela geração dos exemplos utilizados no treinamento.

### `config.py`

Contém as configurações do projeto, incluindo nomes de comandos, tipos de slots e tags BIO.

### `lse_norberto.py`

Implementa a arquitetura do modelo, a função de loss, a decodificação dos spans e o pipeline de inferência.

---

## Referências

### NorBERTo

NorBERTo é um modelo encoder baseado na arquitetura ModernBERT, treinado para português. O modelo base está disponível em:

`Itau-Unibanco/NorBERTo-large`

O trabalho do NorBERTo descreve o treinamento em um grande corpus de português e sua aplicação em tarefas de NLP.

### BIO Tagging

O modelo utiliza o esquema BIO (Beginning, Inside, Outside) para segmentação de spans de ações e slots.

---

## Autor

Modelo fine-tuned desenvolvido por **ianzeraA**.

Hugging Face:

`ianzeraA/NorBERTo-Ar-condicionado`
