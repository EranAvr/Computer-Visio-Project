import os
import numpy as np  # linear algebra
import matplotlib.pyplot as plt     # data visualization
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd     # recording raw data in tables

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from citrus_data_aug import RandomGaussianNoiseTransform, RandomUniformNoiseTransform, RandomGaussianBlur


########################################################################
#                  MODULE CONFIGURATIONS                               #
########################################################################

# Directories:
BASE_DIR = 'D:\\Datasets\\good_citrus_dataset_cut'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
TEST_DIR = os.path.join(BASE_DIR, 'test')
ANOTHER_TEST_DATASET = 'D:\\Datasets\\another_citrus_dataset\\test'
RECORD_DIR = 'records/'
if not os.path.exists(RECORD_DIR):
    os.mkdir(RECORD_DIR)

# NN Model Training Settings:
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 0.00005
SHUFFLE_DATA = True



########################################################################
#                  NEURAL NETWORK MODEL                                #
########################################################################

MODEL = nn.Sequential(resnet50(ResNet50_Weights.DEFAULT), nn.Linear(in_features=1000, out_features=2))
OPTIMIZER = optim.Adam(params=MODEL.parameters(), lr=LEARNING_RATE)
LOSS_F = F.cross_entropy



########################################################################
#                  DATASETS AND DATALOADERS                            #
########################################################################

# Transforms:
advanced_transformer = T.Compose(
        [  # Applying Augmentation
            T.ToTensor(),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomCrop(175),
            RandomGaussianBlur((31,31), 1, probability=0.3),
            RandomUniformNoiseTransform(0.2, probability=0.4),
            RandomGaussianNoiseTransform(0.05, probability=0.3),
            #T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
)


# Datasets and Dataloaders:
train_dataset = ImageFolder(TRAIN_DIR, transform=advanced_transformer)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                              shuffle=SHUFFLE_DATA)

test_dataset = ImageFolder(TEST_DIR, transform=advanced_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=SHUFFLE_DATA)

foreign_dataset = ImageFolder(ANOTHER_TEST_DATASET, transform=advanced_transformer)
foreign_loader = DataLoader(dataset=foreign_dataset,
                                     batch_size=BATCH_SIZE,
                                     shuffle=SHUFFLE_DATA)

########################################################################
#                  TRAINING AND TESTS                                  #
########################################################################

def train(model, optimizer, data_loader: DataLoader, record=True):
    records_df = pd.DataFrame({'epoch': [], 'batch': [], 'avg_loss': [],
                               'running_loss': []})
    model.train()
    x_ax = []
    y_ax = []
    for epoch in range(EPOCHS):
        print(f'>>> Epoch #{epoch}')
        batch_count = 0
        img_count = 0
        running_loss = 0
        for batch in data_loader:
            batch_count += 1
            images, labels = batch
            # to cuda:
            #   model = model.to(device)
            #   images = images.to(device)
            #   labels = labels.to(device)
            # 1. feed model
            predictions = model(images)
            # 2. calc loss
            loss = LOSS_F(predictions, labels)
            running_loss += loss.item() * images.size(0)
            # 3. backward propagataion
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # to cpu:
            #     model = model.to('cpu')
            #     images = images.to('cpu')
            #     labels = labels.to('cpu')

            img_count += len(images)
            print(
                f'Train: complete={round((img_count / len(train_dataset)) * 100, 3)}%\timages '
                f'processed={img_count}/{len(train_dataset)}\t\tloss={loss}\trunning loss:{running_loss}')
            if record:
                records_df.loc[len(records_df.index)] = [epoch, batch_count,
                                                         loss.item(),
                                                         running_loss]
        # for each epoch:
        torch.save(model.state_dict(), os.path.join(os.getcwd(), f'model_state_dict{epoch}.pth'))
        x_ax.append(epoch)
        y_ax.append(running_loss)
    # end of test:
    if record:
        records_df.to_csv(f'{RECORD_DIR}citrus_dis_train_record.csv')
        plt.plot(x_ax, y_ax, marker='o')
        plt.title(f'Training: batch={BATCH_SIZE} lr={LEARNING_RATE}')
        plt.xlabel('Epochs')
        plt.ylabel('Running Loss')
        plt.savefig(f'{RECORD_DIR}train_graph.png')


def test(model, data_loader: DataLoader, record=True):
    records_df = pd.DataFrame({'processed': [], 'missed': [], 'success': [],
                               'predictions': [], 'true labels': [],
                               'differences': []})
    with torch.no_grad():
        img_count = 0
        model.eval()
        batch_count = 0
        for batch in data_loader:
            images, labels = batch
            predictions = model(images)
            predictions = pred_to_binary(predictions)

            img_count += len(images)
            misses_vector = np.bitwise_xor(predictions.numpy(), labels.numpy())
            img_missed = np.sum(misses_vector)

            print(f'processed: {img_count}/{len(test_dataset)}\t'
                  f'missed: {img_missed}/{len(predictions)}\n'
                  f'predis: {predictions}\nlabels: {labels}')
            if record:
                # build Dataframe records table:
                records_df.loc[len(records_df.index)] = [img_count, img_missed,
                                                         ((images.size(0)-img_missed)/images.size(0))*100,
                                                         predictions.tolist(),
                                                         labels.tolist(),
                                                         misses_vector.tolist()]
                # plotting image grid:
                dim = int(np.sqrt(BATCH_SIZE))
                fig = plt.figure(figsize=(10., 10.))
                fig.suptitle(f'processed: {len(images)}   missed: {img_missed}')
                grid = ImageGrid(fig, 111,
                                 nrows_ncols=(dim, dim),
                                 axes_pad=0.5)
                idx = 0
                for ax, im in zip(grid, [img.permute(1, 2, 0) for img in images]):
                    ax.imshow(im)
                    ax.set_title(f'true: {test_dataset.classes[labels[idx]]}\n'
                                f'pred: {test_dataset.classes[predictions[idx]]}')

                    idx += 1
                plt.savefig(f'{RECORD_DIR}test_batch{batch_count}.png')
            batch_count += 1
    records_df.to_csv(f'{RECORD_DIR}citrus_dis_test_record.csv')
def pred_to_binary(pred: torch.Tensor):
    dim = len(pred)
    pred = torch.sigmoid(pred).numpy()
    res = torch.tensor([pred[i].tolist().index(np.amax(pred[i])) for i in range(dim)])
    return res