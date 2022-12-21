import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
from Unet import Unet
from utlis import load_checkpoint, save_checkpoint, get_loader, check_accuracy, save_predictions_as_images


learning_rate = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 16
num_epochs = 3
num_workers = 1
image_height = 160
image_width = 240
pin_memory = True
Load_model = False
train_img_dir = "./data/train/"
train_mask_dir = "./data/train_masks/"
val_img_dir = "./data/val/"
val_mask_dir = "./data/val_masks/"

def train(loader, model, optimizer , loss_fn, scaler):
    qbar = tqdm(loader)

    for idx, (data, targets) in enumerate(qbar):
        data = data.to(device)
        targets = targets.float().unsquuenze(1).to(device)

        with torch.cuda.amp.autocast():
            pred = model(data)
            loss = loss_fn(pred, targets)
        
        #backward

        optimizer.zero_grad()
        scaler.scalar(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        qbar.set_postfix(loss = loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height= image_height , width= image_width),
            A.Rotate(limit = 35 , p = 1.0),
            A.HorizontalFlip(p= 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean= [0.0, 0.0, 0.0],
                std = [1.0,1.0,1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    val_transform = A.Compose(
        [
            A.resize(height= image_height, width= image_width),
            A.Normalize( mean = [0.0, 0.0, 0.0],
            std= [1.0, 1.0, 1.0],
            max_pixel_value= 255.0

            )
        ]
    )

    model = Unet(in_channels= 3 , out_channels= 1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    train_loader, val_loader = get_loader(
        train_dir= train_img_dir , 
        train_maskdir= train_mask_dir,
        val_dir= val_img_dir,
        val_maskdir= val_mask_dir,
        train_transform= train_transform,
        val_transform= val_transform
    )

    if Load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    #check_accuracy(val_loader, model= model, device= device)
    scalar = torch.cuda.amp.grad_scaler()

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, loss_fn, scalar)

        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        check_accuracy(val_loader, model, device)
        save_predictions_as_images(val_loader, model, folder= "./saved_images/", device= device)



if __name__ == "__main__":
    main()









