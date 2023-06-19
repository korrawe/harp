import torch


def arap_loss(meshes, ref_meshes, target_length: float = 0.0):
    """
    Compute as-rigid-as-possible loss term. Trying to maintain mess edge length
    compared to the template.

    Computes mesh edge length regularization loss averaged across all meshes
    in a batch. Each mesh contributes equally to the final loss, regardless of
    the number of edges per mesh in the batch by weighting each mesh with the
    inverse number of edges. For example, if mesh 3 (out of N) has only E=4
    edges, then the loss for each edge in mesh 3 should be multiplied by 1/E to
    contribute to the final loss.

    Args:
        meshes: Meshes object with a batch of meshes.
        ref_meshes: Reference Meshes.
        target_length: Resting value for the edge length.

    Returns:
        loss: Average loss across the batch. Returns 0 if meshes contains
        no meshes or all empty meshes.
    """
    if meshes.isempty():
        return torch.tensor(
            [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
        )

    N = len(meshes)
    edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
    verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
    edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    num_edges_per_mesh = meshes.num_edges_per_mesh()  # N

    if len(ref_meshes) == 1:
        ref_meshes = ref_meshes.extend(N)
    ref_edges_packed = ref_meshes.edges_packed()  # (sum(E_n), 3)
    ref_verts_packed = ref_meshes.verts_packed()  # (sum(V_n), 3)
    ref_edge_to_mesh_idx = ref_meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )
    ref_num_edges_per_mesh = ref_meshes.num_edges_per_mesh()  # N

    # Determine the weight for each edge based on the number of edges in the
    # mesh it corresponds to.
    weights = num_edges_per_mesh.gather(0, edge_to_mesh_idx)
    weights = 1.0 / weights.float()

    verts_edges = verts_packed[edges_packed]
    v0, v1 = verts_edges.unbind(1)

    # Ref edges
    ref_verts_edges = ref_verts_packed[ref_edges_packed]
    ref_v0, ref_v1 = ref_verts_edges.unbind(1)
    # import pdb; pdb.set_trace()
    loss = ((v0 - v1).norm(dim=1, p=2) * 1000.0 - (ref_v0 - ref_v1).norm(dim=1, p=2) * 1000.0) ** 2.0
    loss = loss * weights

    return loss.sum() / N