import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from Dataset import ImageDataset, ImageOnHotDataset
from models import Resnet34, Resnet101_8class

from tqdm.auto import tqdm

from models import F1Loss
from adamp import AdamP

LR = 0.00005
EPOCHS = 5
torch.manual_seed(42)

train_transform = transforms.Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
])
# dataset = ImageDataset(csv_path='../../input/data/train/train.csv',
#                        image_path='../../input/data/train/',
#                        activate_transform=True,
#                        extra_path='../../input/data/train/not_wear.csv',
#                        transform=train_transform)

dataset = ImageOnHotDataset(csv_path='../../input/data/train/train.csv',
                            image_path='../../input/data/train/',
                            activate_transform=True,
                            extra_path='../../input/data/train/not_wear.csv',
                            transform=train_transform)

train_data_loader = DataLoader(dataset,
                               batch_size=64,
                               shuffle=True)

device = torch.device('cuda')
model = Resnet101_8class(num_classes=8).to(device)


CE_gender = nn.CrossEntropyLoss()
CE_age = nn.CrossEntropyLoss()
CE_mask = nn.CrossEntropyLoss()

F1_gender = F1Loss(classes=2)
F1_age = F1Loss(classes=3)
F1_mask = F1Loss(classes=3)

optimizer = AdamP(model.parameters(), lr=LR)

print("Start training !")
# Training loop
for epoch in tqdm(range(EPOCHS)):
    printer = 0
    for batch_img, batch_lab in tqdm(train_data_loader):
        printer += 1

        X = batch_img.to(device)
        Y = batch_lab.to(device)

        # Inference & Calculate los
        y_pred = model.forward(X)

        gender_CEloss = CE_gender(y_pred[:, :2], Y[:, :2].argmax(dim=-1))
        age_CEloss = CE_age(y_pred[:, 2:5], Y[:, 2:5].argmax(dim=-1))
        mask_CEloss = CE_mask(y_pred[:, 5:], Y[:, 5:].argmax(dim=-1))

        gender_F1lloss = CE_gender(y_pred[:, :2], Y[:, :2].argmax(dim=-1))
        age_F1lloss = CE_age(y_pred[:, 2:5], Y[:, 2:5].argmax(dim=-1))
        mask_F1lloss = CE_mask(y_pred[:, 5:], Y[:, 5:].argmax(dim=-1))

        CE_loss = gender_CEloss + age_CEloss + mask_CEloss
        F1_loss = gender_F1lloss + age_F1lloss + mask_F1lloss

        total_loss = CE_loss + F1_loss

        optimizer.zero_grad()
        total_loss.backward()  # Ensemble backward
        optimizer.step()


        if printer % 30 == 0:
            gender_acc =(float(sum(y_pred[:, :2].cpu().argmax(dim=-1) == Y[:, :2].cpu().argmax(dim=-1))) /
                         float(len(Y[:, :2].cpu().argmax(dim=-1))))

            age_acc = (float(sum(y_pred[:, 2:5].cpu().argmax(dim=-1) == Y[:, 2:5].cpu().argmax(dim=-1))) /
                       float(len(Y[:, :2].cpu().argmax(dim=-1))))

            mask_acc = (float(sum(y_pred[:, 5:].cpu().argmax(dim=-1) == Y[:, 5:].cpu().argmax(dim=-1))) /
                        float(len(Y[:, 5:].cpu().argmax(dim=-1))))

            print('gender acc: ', gender_acc)
            print('age acc: ', age_acc)
            print('mask acc: ',mask_acc)

            print('\n Total loss : ', total_loss.cpu())

    #save checkpoint
    torch.save({'EPOCHS': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'age_acc': age_acc,
                }, f'./checkpoints/agefilter/checkpoint_at_{epoch}_agefilter.pickle')

print("Training Done !")

#save endside model
torch.save(model.state_dict(), './learned_models/agefilter_F1CE_state_dict.pt')
