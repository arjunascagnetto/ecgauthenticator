"""
Numerical reproduction of the contrastive-loss example in
CONTRASTIVE_LOSS_MECHANICS.tex using only the Python standard library.

The script reports embeddings, distances, per-pair losses, and the
gradient contributions with respect to the first element of each pair.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, Iterable, Tuple

Vector3 = Tuple[float, float, float]
Vector2 = Tuple[float, float]

MARGIN = 1.0

W = (
    (1.0, 0.2, 0.0),
    (0.0, 0.8, 0.4),
)

SAMPLES: Dict[str, Vector3] = {
    "a1": (1.0, 0.9, 0.1),
    "a2": (0.8, 1.1, 0.2),
    "b1": (2.1, 1.9, 0.4),
    "c1": (2.0, 0.4, 1.2),
}


def matvec(mat: Tuple[Vector3, Vector3], vec: Vector3) -> Vector2:
    return tuple(sum(m_i * v_i for m_i, v_i in zip(row, vec)) for row in mat)  # type: ignore[return-value]


def vec_sub(u: Vector2, v: Vector2) -> Vector2:
    return (u[0] - v[0], u[1] - v[1])


def vec_add(u: Vector2, v: Vector2) -> Vector2:
    return (u[0] + v[0], u[1] + v[1])


def vec_scale(u: Vector2, s: float) -> Vector2:
    return (u[0] * s, u[1] * s)


def vec_norm(u: Vector2) -> float:
    return sqrt(u[0] ** 2 + u[1] ** 2)


def contrastive_loss_and_grad(z_a: Vector2, z_b: Vector2, y: int, margin: float) -> Tuple[float, float, Vector2]:
    diff = vec_sub(z_a, z_b)
    dist = vec_norm(diff)
    if y == 1:
        loss = dist ** 2
        grad = vec_scale(diff, 2.0)
    else:
        if dist >= margin:
            loss = 0.0
            grad = (0.0, 0.0)
        else:
            residual = margin - dist
            scale = -2.0 * residual / dist
            grad = vec_scale(diff, scale)
            loss = residual ** 2
    return loss, dist, grad


def pprint_vec(tag: str, vec: Iterable[float]) -> None:
    vals = ", ".join(f"{v:.4f}" for v in vec)
    print(f"{tag}: ({vals})")


def main() -> None:
    embeddings = {name: matvec(W, vec) for name, vec in SAMPLES.items()}
    print("Embedding 2D:")
    for name, emb in embeddings.items():
        pprint_vec(f"  {name}", emb)
    print()

    pairs = [
        ("a1", "a2", 1, "positiva"),
        ("a1", "b1", 0, "negativa (oltre margin)"),
        ("a1", "c1", 0, "negativa (attiva)"),
    ]

    total_grad = (0.0, 0.0)

    for anchor, other, label, desc in pairs:
        loss, dist, grad = contrastive_loss_and_grad(
            embeddings[anchor], embeddings[other], label, MARGIN
        )
        total_grad = vec_add(total_grad, grad)
        print(f"Coppia {anchor}-{other} [{desc}]")
        print(f"  distanza = {dist:.4f}")
        print(f"  loss     = {loss:.4f}")
        pprint_vec("  grad_anchor", grad)
        print()

    pprint_vec(f"Somma gradienti su {pairs[0][0]}", total_grad)


if __name__ == "__main__":
    main()
