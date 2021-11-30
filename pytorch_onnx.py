import torch
import torch.onnx
from utils import mobilenet
model_file_dir = "model_v3/tomato_ripeness_56_.pt"

model = mobilenet(32, 1, 37, 256)
model.load_state_dict(torch.load(model_file_dir))
dummy_input = torch.randn(1, 1, 32, 100)
torch.onnx.export(model, dummy_input, "crnn.onnx", verbose=True)