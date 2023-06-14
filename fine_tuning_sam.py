from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2

from segment_anything import SamPredictor, sam_model_registry
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
from tqdm import tqdm


checkpoint = 'sam_vit_b_01ec64.pth'
#sam = sam_model_registry['vit_b'](checkpoint=checkpoint)
#predictor = SamPredictor(sam)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mask_dir, transform=None, transform_mask=None):
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.transform = None#predictor.transform
        self.transform_mask = transform_mask
        self.images = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.images[idx])
        mask_name = os.path.join(self.mask_dir, self.images[idx][:3]+"_label.PNG")

        image = np.asarray(Image.open(img_name))
        mask = np.asarray(Image.open(mask_name))

        if self.transform:
            image = self.transform.apply_image(image)
            mask = self.transform_mask(mask)
            #print("Mask worked")

        return image, mask

# Load custom dataset
dataset = CustomDataset(root_dir='/content/Images', mask_dir='/content/Masks')

# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor.model.to(device)
predictor.model.train()

criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([60]).to(device))
optimizer = torch.optim.SGD(predictor.model.mask_decoder.parameters(), lr=.1, momentum=0.9)

for epoch in range(4):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader)):
        inputs, labels = data
        labels = labels.to(device)/255
        #labels[labels == 0] = -1

        inputs = inputs.to(device)

        optimizer.zero_grad()

        predictor.set_image(np.asarray(inputs[0].cpu()))
        masks, scores, logits = predictor.predict(
            multimask_output=False,
            return_logits=True
        )
        

        loss = criterion(masks[0], labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"EPOCH LOSS: {running_loss/29}")

print('Finished Training')

