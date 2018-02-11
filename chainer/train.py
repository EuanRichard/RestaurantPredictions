import csv
import numpy as np
import chainer
from chainer import optimizers, training, reporter
from chainer.training import extensions, Trainer

from iterator import ParallelSequentialIterator
from updater import BPTTUpdater
from model import Loss, Model

def normalize(nparray):
    # normalize so input variables are near 1, because chainer likes this kind of data.
    # this is an ugly way of doing it with factors estimated by hand,
    # but what the hey, it was the fastest solution.
    nparray[:,:,1] = nparray[:,:,1]*0.08
    nparray[:,:,10] = nparray[:,:,10]*0.001
    nparray[:,:,11] = nparray[:,:,11]*0.001
    nparray[:,:,12] = nparray[:,:,12]*0.005
    nparray[:,:,13] = nparray[:,:,13]*0.005
    return nparray

def get_dataset(start, end):
    # start and end specifies the date slice (train, eval, test, etc.).
    # the input numpy array has axes [stores, days, variables].
    nparray = np.load("datapoop.npy")
    nparray = nparray.astype("float32")
    # slice 
    nparray = nparray[:,start:end,:]
    # normalize
    nparray = normalize(nparray)
    # append each array slice to a list
    alist = []
    for i in range(len(nparray[0,:,0])):
        alist.append(nparray[:,i,:])
    return alist


def main(epochs=257*8, lr=0.38, seq_len=120, pred_len=39, out="result", device=0):
    
    # CHOOSE ONE:
    # get the training dataset but keep a slice for validation
    dataset = get_dataset(182, -39 -39)
    # get the entire dataset
    #dataset = get_dataset(182, -39)
    
    iter = ParallelSequentialIterator(dataset, pred_len=1, repeat=True)

    model = Model(pred_len=pred_len, dropout=0.1)
    if device >= 0:
        model.to_gpu()

    # Try some different optimizers
    #optimizer = optimizers.Adam(alpha=lr)
    #optimizer = optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer = optimizers.RMSpropGraves(lr=lr, alpha=0.95, momentum=0.2)
    #optimizer = optimizers.RMSprop(lr=lr, alpha=0.5)

    optimizer.setup(model)

    #optimizer.add_hook(chainer.optimizer.GradientClipping(5))#grad_clip))
    #optimizer.add_hook(chainer.optimizer.WeightDecay(1.E-7))

    updater = BPTTUpdater(iter, optimizer, seq_len=seq_len, pred_len=pred_len, data_len=len(dataset), device=device)
    trainer = Trainer(updater, (epochs, 'epoch'), out=out)

    interval = 10

    # Try some learning-rate decay
    #trainer.extend(extensions.ExponentialShift('lr', 0.995)) #0.1, lr, lr * 0.1), trigger=(10, 'epoch'))

    trainer.extend(extensions.observe_lr(), trigger=(interval, "iteration"))
    
    trainer.extend(extensions.LogReport(trigger=(interval, 'iteration')))
    trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'loss', 'lr']),
            trigger=(interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=interval))

    # export snapshots to resume training
    trainer.extend(extensions.snapshot(filename='snapshot_epoch_{.updater.epoch}'), trigger=(257*6, "epoch"))
    trainer.extend(extensions.snapshot_object(model, "model_epoch_{.updater.epoch}"), trigger=(257*2, "epoch"))

    # change to True to resume from file
    if False:
        chainer.serializers.load_npz('result/snapshot_epoch_1030', trainer)
        
    trainer.run()


    # save model
    from chainer import serializers
    serializers.save_npz('restaurant.model', model)
   

                                                                
    
if __name__ == "__main__":
    main(out="result")




