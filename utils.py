from PIL import Image
from torchvision import transforms

def load_image(path, max_size=256):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0))
    ])
    return transform(img)

def save_image(tensor, path):
    img = tensor.squeeze(0).clamp(0, 1)
    transforms.ToPILImage()(img).save(path)
