from fastapi import File, UploadFile, FastAPI
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn


app = app = FastAPI()

def create_srgan_generator():
    model = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )
    return model

def create_denoising_generator():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, padding=1)
    )

def create_cyclegan_generator():
    return nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 3, kernel_size=3, padding=1)
    )

@app.post("/predict/srgan/")
async def predict(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_srgan_generator().to(device)
    
    # Load the model with map_location to handle CPU-only machines
    model.load_state_dict(torch.load("srgan_model.pth", map_location=device))
    model.eval()

    # Read and preprocess the input image
    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Convert output tensor to image
    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())

    # Save or return the output image
    output_image.save("output_image.png")
    
    return {"message": "Super-resolution image generated and saved as output_image.png"}

# Denoising GAN Inference
@app.post("/predict/denoising/")
async def predict_denoising(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the entire model object
    model = torch.load("denoising_model.pth", map_location=device)
    model.to(device)
    model.eval()

    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_image.save("denoising_output.png") # save image in your dir
    return {"message": "Denoising output saved as denoising_output.png"}

# CycleGAN Inference
@app.post("/predict/cyclegan/")
async def predict_cyclegan(file: UploadFile = File(...)):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the entire model object
    model = torch.load("cyclegan_model.pth", map_location=device)
    model.to(device)
    model.eval()

    image = Image.open(file.file).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)

    output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_image.save("cyclegan_output.png") # save image in your dir
    return {"message": "CycleGAN output saved as cyclegan_output.png"}
# Status Endpoint
@app.get("/status/")
async def status():
    return {"status": "API is running"}
