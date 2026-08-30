"""Python mirror of packages/model-ir/src/index.ts.

Kept field-for-field identical (camelCase via Pydantic aliases, since the
frontend consumes this as JSON straight from `@aperture/model-ir`'s
TypeScript types) so the backend and frontend never drift. If you add a
field on one side, add it on the other.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NodeType = Literal[
    "model",
    "block_group",
    "input",
    "embedding",
    "positional_embedding",
    "transformer_block",
    "layer_norm",
    "rms_norm",
    "attention",
    "q_projection",
    "k_projection",
    "v_projection",
    "qkv_projection",
    "output_projection",
    "rope",
    "ffn",
    "linear",
    "activation",
    "elementwise_mul",
    "residual",
    "lm_head",
    "output",
    "moe_layer",
    "router",
    "expert",
    "custom",
]


class Camel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class TensorSpec(Camel):
    dims: list[int | str]


class TensorSlice(Camel):
    ranges: list[dict[str, int]] | None = None


class ParameterRef(Camel):
    name: str
    shape: list[int]
    dtype: str
    num_elements: int = Field(serialization_alias="numElements")
    bytes: int
    provider_id: str = Field(serialization_alias="providerId")
    slice: TensorSlice | None = None
    logical_shape: list[int] = Field(serialization_alias="logicalShape")


class ModelNode(Camel):
    id: str
    type: NodeType
    name: str
    inputs: list[TensorSpec] = Field(default_factory=list)
    outputs: list[TensorSpec] = Field(default_factory=list)
    parameters: list[ParameterRef] = Field(default_factory=list)
    children: list[str] = Field(default_factory=list)
    parent_id: str | None = Field(default=None, serialization_alias="parentId")
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelEdge(Camel):
    id: str
    source: str
    target: str
    label: str | None = None


class ModelConfig(Camel):
    model_type: str = Field(serialization_alias="modelType")
    num_layers: int = Field(serialization_alias="numLayers")
    num_heads: int = Field(serialization_alias="numHeads")
    hidden_size: int = Field(serialization_alias="hiddenSize")
    intermediate_size: int = Field(serialization_alias="intermediateSize")
    vocab_size: int = Field(serialization_alias="vocabSize")
    context_length: int = Field(serialization_alias="contextLength")
    extra: dict[str, Any] = Field(default_factory=dict)


class Model(Camel):
    id: str
    name: str
    architecture: str
    config: ModelConfig
    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    nodes: dict[str, ModelNode]
    edges: list[ModelEdge]
    root_id: str = Field(serialization_alias="rootId")
