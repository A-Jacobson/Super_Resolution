from collections import defaultdict
from shutil import copyfile

import torch
from tqdm import tqdm_notebook
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np


def prep_img(img):
    return Variable(img.unsqueeze(0)).cuda()


class AverageMeter(object):
    """Computes and stores the average and current value"""


    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fit_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = AverageMeter()
    t = tqdm_notebook(loader, total=len(loader))
    for data, target in t:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        running_loss.update(loss.data[0])
        t.set_description("[ loss: {:.4f} ".format(running_loss.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return running_loss.avg


def fit(model, train, criterion, optimizer, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, cuda=True, num_workers=-1):
    # TODO: implement CUDA flags, optional metrics and lr scheduler
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train), len(validation_data)))
        val_loader = DataLoader(validation_data, batch_size, shuffle, num_workers=num_workers)
    else:
        print('Train on {} samples'.format(len(train)))
    train_loader = DataLoader(train, batch_size, shuffle, num_workers=num_workers)
    history = defaultdict(list)
    t = tqdm_notebook(range(nb_epoch), total=nb_epoch)
    for epoch in t:
        loss = _fit_epoch(model, train_loader, criterion, optimizer)
        history['loss'].append(loss)
        if validation_data:
            val_loss, val_acc = validate(model, val_loader, criterion)
            print("[Epoch {} - loss: {:.4f} - val_loss: {:.4f}]".format(epoch + 1, loss, val_loss))
            history['val_loss'].append(val_loss)
        else:
            print("[loss: {:.4f} ]".format(loss))
    return history


def validate(model, val_loader, criterion):
    model.eval()
    val_loss = AverageMeter()
    for data, target in val_loader:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.data[0])
    return val_loss.avg


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')

