"""
models/unet.py
===============
Construcción del modelo de segmentación U-Net usando
`segmentation-models-pytorch` (SMP), que ya implementa encoders
pre-entrenados en ImageNet (fine-tuning) y es el paquete que definiste en
tu stack tecnológico.

Se adapta el encoder para aceptar `in_channels` != 3 (aquí 8: 6 bandas
Sentinel-2 + NDVI + sigma_NIR), lo cual SMP soporta nativamente
reinicializando la primera capa convolucional.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 8,
    classes: int = 1,
) -> nn.Module:
    """
    Devuelve un modelo U-Net listo para entrenar.

    Nota sobre encoder_weights + in_channels != 3: SMP re-inicializa los
    pesos de la primera capa conv cuando in_channels no es 3, promediando/
    replicando los pesos pre-entrenados de RGB a los canales extra — sigue
    siendo mejor que entrenar desde cero (transfer learning parcial).
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # logits crudos; la sigmoid se aplica en la loss/métricas
    )
    return model


if __name__ == "__main__":
    import torch

    model = build_unet()
    x = torch.randn(2, 8, 256, 256)
    y = model(x)
    print("Input:", x.shape, "-> Output:", y.shape)  # esperado: (2, 1, 256, 256)
