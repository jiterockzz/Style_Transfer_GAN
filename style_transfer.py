import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential(*list(vgg.children())[:21])
        for param in self.slice.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.slice(x)

def gram_matrix(tensor):
    B, C, H, W = tensor.size()
    features = tensor.view(B, C, H * W)
    G = torch.bmm(features, features.transpose(1, 2))
    return G / (C * H * W)

def style_transfer(content_img, style_img, num_steps=100, style_weight=1e6, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn = VGGFeatures().to(device).eval()

    input_img = content_img.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([input_img])

    style_features = cnn(style_img)
    content_features = cnn(content_img)
    style_gram = gram_matrix(style_features)

    def closure():
        optimizer.zero_grad()
        input_features = cnn(input_img)
        input_gram = gram_matrix(input_features)
        content_loss = content_weight * torch.nn.functional.mse_loss(input_features, content_features)
        style_loss = style_weight * torch.nn.functional.mse_loss(input_gram, style_gram)
        total_loss = content_loss + style_loss
        total_loss.backward()
        return total_loss

    for i in range(num_steps):
        optimizer.step(closure)

    return input_img.detach()
