from __future__ import division
from __future__ import print_function
import argparse

import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, reporter, report_scope
from chainer.training import extensions
from chainer.cuda import to_cpu

# Custom loss function for Root-Mean-Square-Log-Error
def rmsle(true,pred):

    # don't score predictions below 1 by log, but give them a penalty
    negatives = F.clip(pred, -np.inf, 0.)
    penalty = 0.01*F.mean((negatives)*(negatives))
    np.savetxt("preds.csv",to_cpu(pred.data),fmt="%.8f", delimiter=",")
    pred = F.clip(pred,0.,np.inf)

    # don't score days with true=0
    score_true = F.clip(true, 0., 1.)
    # dont' train too hard on outliers
    true = F.clip(true,0., 200.)

    # save for visualization
    np.savetxt("true.csv",to_cpu(true.data),fmt="%.8f", delimiter=",")

    # rmsle calculations
    le = F.log(pred+1.) - F.log(true+1.) 
    le = le*score_true # skip non-scorable days
    msle = F.mean(le * le)
    return F.sqrt(msle) + penalty

# Custom updater for truncated BackProp Through Time (BPTT)
class BPTTUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer, seq_len, pred_len, data_len, loss_func=rmsle, device=-1):
        super().__init__(
            train_iter, optimizer, device=device)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.data_len = data_len
        self.loss_func = loss_func

    # The core part of the update routine can be customized by overriding.
    def update_core(self):
        # When we pass one iterator and optimizer to StandardUpdater.__init__,
        # they are automatically named 'main'.
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')
        optimizer.target.reset_state()


        # get epoch, and an offset that will be increased by one each epoch
        # so the training window scans ahead by one day each time
        epoch = int(train_iter.epoch_detail)
        offset = epoch % self.data_len

        # if we are going to cross the end of the data set in this training,
        # exhaust the iterator until we get back to the start.
        # this is inefficient but there are some weird
        # chainer things going on, so please excuse the hack for now...
        max_offset = self.data_len - (self.seq_len + self.pred_len)
        if offset > max_offset:
            for i in range((self.seq_len+self.pred_len)*self.data_len):
                batch = train_iter.__next__()
            offset = 0
            #print("Skipping end of dataset and setting offset to 0")

        # skip the first [offset] days of data
        for i in range(offset):
            batch = train_iter.__next__()

        # train for a length of seq_len
        for i in range(self.seq_len):
            batch = train_iter.__next__()
            x, t = self.converter(batch, self.device)
           
            # here optimizer.target is actually just a trick for calling model
            # the first calls in this loop initialize the memory in the RNN
            # the final call predicts the future
            # so, y contains predictions for next day

            # set "future predictors" (conditional info) and predict
            # here x and t.shape = (1, 821, 69)
            to_train = F.concat((x,t[:,:,3:]), axis=2)
            # here to_train.shape = (1, 821, 135)
            y = optimizer.target(to_train)


        # get the targets and predictions for the next day (at least 1)
        targets = F.reshape(t[:,:,0], (821, ))
        predictions = y
        x, t = self.converter(train_iter.__next__(), self.device)
        # extend the targets and predictions further over the length of pred_len-1
        for i in range(1, self.pred_len):
            # set todays visitors to what we predicted last loop
            y = F.reshape(y,(1,821,1))
            x = F.concat(  (y, x[:,:,1:]),  axis=2)

            # update predictions
            to_train = F.concat((x,t[:,:,3:]), axis=2)
            y = optimizer.target(to_train)
            # concat into a list of targets and predictions
            targets = F.concat((targets, F.reshape(t[:,:,0],(821,))), axis=0)
            predictions = F.concat((predictions, y), axis=0)
            # next day
            x, t = self.converter(train_iter.__next__(), self.device)

        # Exhaust the iterator in preparation for next scan
        exhaust = self.data_len - (self.pred_len + self.seq_len + offset)
        for i in range(exhaust):
            batch = train_iter.__next__()

        #print("skipped",offset,"scanned",self.seq_len,"predicted",self.pred_len,"skipped",exhaust)

        # Get the loss
        loss = self.loss_func(targets, predictions)


        # housekeeping
        reporter.report({"loss": loss})
        optimizer.target.cleargrads()  # Clear the parameter gradients
        loss.backward()  # Backprop
        loss.unchain_backward()  # Truncate the graph
        optimizer.update()  # Update the parameters
