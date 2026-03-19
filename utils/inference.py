import torch
from qnet import DQNet


def load_model():
    model = DQNet()
    model.load_state_dict(torch.load(r"C:\Users\jencikt\PycharmProjects\ai-model-pokemon\models\stage0.pth", map_location="cpu"))
    model.eval()

    return model

def predict(x, model):
    with torch.no_grad():
        x = torch.tensor(x).float()
        output = model(x)
        return output.tolist()

loaded_model = load_model()

