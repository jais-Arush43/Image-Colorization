import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from model import ColorizationModel
from resnet import build_res_unet
# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Function to perform colorization
# @st.cache(allow_output_mutation=True)
def colorize_image(model, input_image):
    # Convert image to grayscale
    input_gray = input_image.convert("L")
    input_tensor = transform(input_gray).unsqueeze(0)

    # Perform colorization
    colorized_img = get_colorized_image(model, input_tensor)
    return colorized_img

def get_colorized_image(model, input_tensor):
    model.net_G.eval()
    with torch.no_grad():
        model.forward(input_tensor)
    model.net_G.train()
    fake_color = model.fake_color.detach()
    L = input_tensor
    colorized_img = lab_to_rgb(L, fake_color)
    return colorized_img

def lab_to_rgb(L, ab):
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1)
    Lab_numpy = Lab.cpu().numpy()
    rgb_imgs = []
    for img in Lab_numpy:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

def main():
    st.title("Image Colorization App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Load the pre-trained model
        model_weights_path = "C:\Image Colorization/final_model_weights.pt"
        net_G = build_res_unet(n_input=1, n_output=2, size=256)
        model = ColorizationModel(net_G=net_G)
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
        model.eval()

        # Display the uploaded image
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Original Image", use_column_width=True)

        # Colorize the image
        with st.spinner("Colorizing..."):
            colorized_img = colorize_image(model, input_image)

        # Display the colorized image
        st.image(colorized_img[0], caption="Colorized Image", use_column_width=True)

if __name__ == "__main__":
    main()