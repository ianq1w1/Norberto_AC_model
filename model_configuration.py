"""
configuration_lse_norberto.py

Classe de Config no padrao do transformers - necessaria pro AutoConfig/
AutoModel saberem como reconstruir o LSEJointModel a partir do config.json,
via trust_remote_code=True.

Este arquivo precisa ir JUNTO dos pesos no repositorio do Hugging Face Hub
(nao fica so na sua maquina) - e o que permite `AutoModel.from_pretrained(
repo_id, trust_remote_code=True)` funcionar sem a pessoa precisar clonar
nenhum outro repositorio separado.
"""

from transformers import PretrainedConfig


class LSENorBERToConfig(PretrainedConfig):
    model_type = "lse_norberto"

    def __init__(
        self,
        encoder_name: str = "Itau-Unibanco/NorBERTo-large",
        num_action_tags: int = 11,
        num_slot_tags: int = 5,
        action_tags=None,
        slot_tags=None,
        dropout: float = 0.1,
        usar_heads_semanticas: bool = True,
        **kwargs,
    ):
        self.encoder_name = encoder_name
        self.num_action_tags = num_action_tags
        self.num_slot_tags = num_slot_tags
        self.action_tags = action_tags or []
        self.slot_tags = slot_tags or []
        self.dropout = dropout
        self.usar_heads_semanticas = usar_heads_semanticas
        super().__init__(**kwargs)