import torch
from depthwizard.config import CalibrationConfig
from depthwizard.calibration_net import CalibrationNet

# Test absolute_affine
cfg1 = CalibrationConfig()
net1 = CalibrationNet(parameterization=cfg1.model.parameterization, widths=cfg1.model.widths)
dn = torch.randn(2, 1, 64, 64)
out1 = net1(dn)
assert out1["pred"].shape == (2, 1, 64, 64)
assert out1["a"].shape == (2, 1, 64, 64)

# Test hybrid_residual
net2 = CalibrationNet(parameterization="hybrid_residual", widths=[16,32,64,128])
out2 = net2(dn)
assert out2["pred"].shape == (2, 1, 64, 64)

# Test dual encoder
net3 = CalibrationNet(in_ch=4, fusion_mode="dual_encoder")
rgb = torch.randn(2, 3, 64, 64)
out3 = net3(dn, rgb=rgb)
assert out3["pred"].shape == (2, 1, 64, 64)

# Test context module
net4 = CalibrationNet(context_module="aspp")
out4 = net4(dn)
assert out4["pred"].shape == (2, 1, 64, 64)

# Test uncertainty
net5 = CalibrationNet(use_uncertainty=True)
out5 = net5(dn)
assert "log_var" in out5

print("All V2 unit tests passed!")
