import torch
from depthwizard.calibration_net import CalibrationNet

# Test semantic_mode="joint"
net = CalibrationNet(semantic_mode="joint", sem_classes=6, sem_aux_head=True, widths=(16, 32, 64))
dn = torch.randn(2, 1, 64, 64)
# since sem is NOT provided in input but it's not needed as input if we just predict it from aux head
# Wait, if semantic_mode="joint", it predicts it and injects it. Let's see if it works.
out = net(dn)
assert out["pred"].shape == (2, 1, 64, 64)
assert out["sem_logits"].shape == (2, 6, 64, 64)
print("Semantic joint test passed!")
