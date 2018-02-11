from train import *
from chainer import serializers
from model import Loss, Model
from iterator import ParallelSequentialIterator
from chainer.dataset import concat_examples as converter

import chainer.functions as F 
from sklearn.metrics import mean_squared_log_error
from math import sqrt, log

# --- User Options ---

model_filename = 'result/model_epoch_2058_best'

# Optionally, use dropout to obtain an ensemble of predictions
dropout = 0

# Set either eval or final test data
is_eval = True

# initialize seq_len and pred_len - should match those of the saved model
seq_len = 120
pred_len = 39

# change to 0 for GPU, but CPU is fine here
device = -1


# custom RMSLE, does not score when true visitor numbers = 0
def get_score(true, pred):
    a, b = [], []
    for i in range(len(true)):
        if true[i] != 0:
            a.append(true[i])
            # Guard against negative predictions
            if pred[i] > 1:
                b.append(pred[i])
            else:
                b.append(1)
    return sqrt(mean_squared_log_error(a,b))


# --- Load Model and run Predictions ---

# Load the saved model
model = Model(pred_len=pred_len, dropout=dropout)
serializers.load_npz(model_filename, model)

if (dropout == 0):
    chainer.using_config('train', False)


print("Now running the predictions.")

# get eval or test data
if is_eval:
    eval_dataset = get_dataset(-39-seq_len, None)
else:
    eval_dataset = get_dataset(-78-seq_len, -78+pred_len)

eval_iter = ParallelSequentialIterator(eval_dataset, pred_len=pred_len, repeat=False)



# feed in training sequence
for i in range(seq_len):
    batch = eval_iter.__next__()
    x, t = converter(batch, device)
    to_train = F.concat((x,t[:,:,3:]), axis=2)
    y = model(to_train)


# feed out predictions
targets = F.reshape(t[:,:,0], (821, ))
predictions = y
x, t = converter(eval_iter.__next__(), device)
for i in range(1, pred_len):
    y = F.reshape(y,(1,821,1))
    x = F.concat(  (y, x[:,:,1:]),  axis=2)
    to_train = F.concat((x,t[:,:,3:]), axis=2)
    y = model(to_train)
    targets = F.concat((targets, F.reshape(t[:,:,0],(821,))), axis=0)
    predictions = F.concat((predictions, y), axis=0)
    x, t = converter(eval_iter.__next__(), device)


if (device > -1):
    from chainer.cuda import to_cpu
    predictions = to_cpu(predictions)
    targets = to_cpu(targets)


predictions = F.reshape(predictions, (-1,)).data
targets = F.reshape(targets, (-1,)).data

# score
print("Score:", get_score(targets.data, predictions.data) )

# save
np.savetxt("preds.csv",predictions,fmt="%.8f", delimiter=",")
np.savetxt("true.csv",targets,fmt="%.8f", delimiter=",")
