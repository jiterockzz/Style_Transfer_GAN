from utils import load_image, save_image
from style_transfer import style_transfer
from upscaling import load_esrgan_model, upscale_image
from PIL import Image

# Load images
content = load_image('images/content.jpg')
style = load_image('images/style.jpg')

# Apply neural style transfer
output = style_transfer(content, style)
save_image(output, 'images/stylized.jpg')

# Load ESRGAN and upscale
styled = Image.open('images/stylized.jpg')
esrgan = load_esrgan_model('models/RRDB_ESRGAN_x4.pth')
final = upscale_image(styled, esrgan)

# Save final output
final.save('images/output.jpg')
print("Process complete. Output saved to images/output.jpg")
