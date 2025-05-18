import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

def load_esrgan_model(model_path='models/RRDB_ESRGAN_x4.pth'):
    from basicsr.archs.rrdbnet_arch import RRDBNet
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    return model

def upscale_image(img, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img = ToTensor()(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img).clamp(0, 1)
    return ToPILImage()(output.squeeze(0).cpu())
