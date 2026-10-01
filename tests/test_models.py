import torch

from fly_abstraction.graph.connectome import ConnectomeGraph, tiny_synthetic_graph
from fly_abstraction.models.baselines import GRUBaseline, MLPBaseline
from fly_abstraction.models.connectome import (
    FixedConnectomeReservoir,
    RandomGraphReservoir,
    TrainableConnectomeRNN,
)


def test_single_forward_backward_without_optimizer_step() -> None:
    graph = tiny_synthetic_graph(8, 17).normalized()
    model = TrainableConnectomeRNN(graph, vocab_size=32, embedding_dim=4, expression_vocab_size=32)
    token_ids = torch.randint(0, 32, (2, 6))
    numeric = torch.zeros(2, 6, 1)
    output = model(token_ids, numeric)
    loss = output.answer.square().mean() + output.expression_logits.square().mean()
    loss.backward()
    assert model.cell.edge_scale.grad is not None
    assert model.recurrent_parameter_count == graph.num_edges
    assert output.expression_logits.shape == (2, 6, 32)
    assert model.parameter_counts()["trainable"] > graph.num_edges


def test_all_required_model_types_construct() -> None:
    graph = tiny_synthetic_graph(8, 17)
    models = [
        FixedConnectomeReservoir(graph, 32, 4, 32),
        TrainableConnectomeRNN(graph, 32, 4, 32),
        RandomGraphReservoir(graph, 32, 4, 32),
        GRUBaseline(32, 4, 8, 32),
        MLPBaseline(32, 4, 8, 32),
    ]
    assert all(model.parameter_counts()["total"] > 0 for model in models)
    assert models[0].parameter_counts()["frozen"] > 0


def test_connectome_model_uses_declared_io_node_mappings() -> None:
    base = tiny_synthetic_graph(8, 17)
    graph = ConnectomeGraph(
        num_nodes=base.num_nodes,
        edge_index=base.edge_index,
        edge_weight=base.edge_weight,
        input_nodes=torch.tensor([0, 2, 4]),
        output_nodes=torch.tensor([1, 3]),
    )
    model = TrainableConnectomeRNN(graph, 32, 4, 32)
    assert model.cell.input_projection.out_features == 3
    assert model.answer_head[0].normalized_shape == (2,)
    output = model(torch.randint(0, 32, (2, 5)), torch.zeros(2, 5, 1))
    assert output.answer.shape == (2,)
