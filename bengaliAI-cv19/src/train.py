import config
import model
import dataset
from tqdm import tqdm
import torch
import torch.nn as nn

def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2,t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1+l2+l3)/3

def train(dataset, data_loader, model, optimizer):
    model.train()
    
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset))/data_loader.batch_size):
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(config.DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(config.DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(config.DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(config.DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root,vowel_diacritic,consonant_diacritic)
        loss = loss_fn(outputs, targets)
        print("Current loss:{}".format(loss))
        loss.backward()
        optimizer.step()

def evaluate(dataset, data_loader, model, optimizer):
    model.eval()
    final_loss = 0
    counter = 0
    for bi, d in tqdm(enumerate(data_loader), total=int(len(dataset))/data_loader.batch_size):
        counter += 1
        image = d["image"]
        grapheme_root = d["grapheme_root"]
        vowel_diacritic = d["vowel_diacritic"]
        consonant_diacritic = d["consonant_diacritic"]

        image = image.to(config.DEVICE, dtype=torch.float)
        grapheme_root = grapheme_root.to(config.DEVICE, dtype=torch.long)
        vowel_diacritic = vowel_diacritic.to(config.DEVICE, dtype=torch.long)
        consonant_diacritic = consonant_diacritic.to(config.DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(image)
        targets = (grapheme_root,vowel_diacritic,consonant_diacritic)
        loss = loss_fn(outputs, targets)
        final_loss += loss
        print("Counter: {},  loss:{}".format(counter, loss))
    return final_loss / counter



def main():

    modell = model.Resnet34().to(config.DEVICE)

    train_dataset = dataset.BengaliHandWrittenDigits(config.TRAINING_FOLDS, config.IMG_HEIGHT, config.IMG_WIDTH, config.MEAN, config.STD)
    valid_dataset = dataset.BengaliHandWrittenDigits(config.VALIDATION_FOLDS, config.IMG_HEIGHT, config.IMG_WIDTH, config.MEAN, config.STD)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    optimizer = torch.optim.Adam(modell.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.3, verbose=True)

    for epoch in range(config.EPOCHS):
        train(train_dataset, train_loader,modell,optimizer)
        val_score = evaluate(valid_dataset, valid_loader, modell, optimizer)
        scheduler.step(val_score)
        torch.save(model.state_dict(), f"resnet34_fold{config.VALIDATION_FOLDS[0]}.h5")

if __name__=="__main__":
    main()
