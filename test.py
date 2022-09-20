from utils import *
from torch import optim
import torch
import logging # modified eval_paper_authors: added for logging
from config import args
from link_prediction import link_prediction
from evolution import calc_raw_mrr, calc_filtered_test_mrr, calc_time_filtered_test_mrr # modified eval_paper_authors: calc_time_filtered_test_mrr
import warnings
warnings.filterwarnings(action='ignore')

torch.set_num_threads(2)

use_cuda = args.gpu >= 0 and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# added by eval_paper_authors/ to integrate logging
n = 'CyGNet'
log_dir = f'../logs_22/{n}.log'
CHECK_FOLDER = os.path.isdir(log_dir)
if not CHECK_FOLDER:
    os.makedirs(log_dir)
    print("created folder : ", log_dir)
else:
    print(log_dir, "folder already exists.")

logging.basicConfig(filename=log_dir, filemode='a',
                    format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

if args.feedvalid: # # modified eval_paper_authors: if true, do use the validation set tduring testin. acc. to our paper this should always be true.
    feed_valid_eval_paper_authors = True
else:
    feed_valid_eval_paper_authors = False
# end added by eval_paper_authors/ to integrate logging

# modified by eval_paper_authors/ to not leak test set
# if args.dataset == 'ICEWS14':
# 	train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
# 	test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
# 	dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
# else:
# end modified by eval_paper_authors/ to not leak test set
train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
test_data, test_times = load_quadruples('./data/{}'.format(args.dataset), 'test.txt')
dev_data, dev_times = load_quadruples('./data/{}'.format(args.dataset), 'valid.txt')

all_times = np.concatenate([train_times, dev_times, test_times])

num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')
num_times = int(max(all_times) / args.time_stamp) + 1
print('num_times', num_times)

model = link_prediction(num_e, args.hidden_dim, num_r, num_times, use_cuda)
model.to(device)

all_tail_seq_obj = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
all_tail_seq_sub = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
if feed_valid_eval_paper_authors ==False: #added eval_paper_authors. by default this is False.
    for i in range(len(train_times)):
        tim_tail_seq_obj = sp.load_npz(
            './data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[i]))
        tim_tail_seq_sub = sp.load_npz(
            './data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, train_times[i]))
        all_tail_seq_obj = all_tail_seq_obj + tim_tail_seq_obj
        all_tail_seq_sub = all_tail_seq_sub + tim_tail_seq_sub
else: #modified eval_paper_authors: if feed_valid_eval_paper_authors==True: the validation timesteps and validation information is allowed to be used. it should always be true
    used_times = len(train_times) +len(dev_times)
    for i in range(used_times):
        tim_tail_seq_obj = sp.load_npz(
            './data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, all_times[i]))
        tim_tail_seq_sub = sp.load_npz(
            './data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, all_times[i]))
        all_tail_seq_obj = all_tail_seq_obj + tim_tail_seq_obj
        all_tail_seq_sub = all_tail_seq_sub + tim_tail_seq_sub   
# end modified eval_paper_authors

model_state_file_obj = './results/bestmodel/{}/{}_model_state.pth'.format(args.dataset,args.setting) #modified eval_paper_authors to name model acc. to metric setting
model_state_file_sub = './results/bestmodel/{}_sub/{}_model_state.pth'.format(args.dataset,args.setting) #modified eval_paper_authors to name model acc. to metric setting
batch_size = args.batch_size

print("\nstart object testing:")
# use best model checkpoint
checkpoint_obj = torch.load(model_state_file_obj)
# if use_cuda:
# model.cpu()  # test on CPU
model.train()
model.load_state_dict(checkpoint_obj['state_dict'])
print("Using best epoch: {}".format(checkpoint_obj['epoch']))

obj_test_mrr, obj_test_hits1, obj_test_hits3, obj_test_hits10 = 0, 0, 0, 0
n_batch = (test_data.shape[0] + batch_size - 1) // batch_size

# added eval_paper_authors
    #for logging scores
import inspect
import sys
import os
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
sys.path.insert(1, currentdir) 
sys.path.insert(1, os.path.join(sys.path[0], '..'))        

eval_paper_authors_logging_dict = {}
import evaluation_utils
exp_nr = int(args.runnr) #seed
if args.multi_step:
    print('eval_paper_authors multistep multistep')
    eval_paper_authors_multistep_bool =True
    steps = 'multistep'
    if feed_valid_eval_paper_authors:
        steps = steps+'feedvalid'
else:
    print('eval_paper_authors multistep not multistep')
    eval_paper_authors_multistep_bool = False
    steps ='singlestep'

## END added eval_paper_authors

#
for idx in range(n_batch):
    batch_start = idx * batch_size
    batch_end = min(test_data.shape[0], (idx + 1) * batch_size)
    test_batch_data = test_data[batch_start: batch_end]

    test_label = torch.LongTensor(test_batch_data[:, 2])
    seq_idx = test_batch_data[:, 0] * num_r + test_batch_data[:, 1]

    tail_seq = torch.Tensor(all_tail_seq_obj[seq_idx].todense())
    one_hot_tail_seq_obj = tail_seq.masked_fill(tail_seq != 0, 1)

    if use_cuda:
        test_label, one_hot_tail_seq_obj = test_label.to(device), one_hot_tail_seq_obj.to(device)
    test_score = model(test_batch_data, one_hot_tail_seq_obj, entity='object')

    # modified by eval_paper_authors/ to integrate different filter settings
    if args.setting == 'raw':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(test_score, test_label, hits=[1, 3, 10])
    elif args.setting == 'static':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_test_mrr(num_e, test_score,
                                                                           torch.LongTensor(
                                                                               train_data),
                                                                           torch.LongTensor(
                                                                               dev_data),
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           torch.LongTensor(
                                                                               test_batch_data),
                                                                           entity='object',
                                                                           hits=[1, 3, 10])
    elif args.setting == 'time':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_time_filtered_test_mrr(num_e, test_score,
                                                                           torch.LongTensor(
                                                                               train_data),
                                                                           torch.LongTensor(
                                                                               dev_data),
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           torch.LongTensor(
                                                                               test_batch_data),
                                                                           entity='object',
                                                                           hits=[1, 3, 10])
    # endmodified by eval_paper_authors/ to integrate different filter settings
    obj_test_mrr += tim_mrr * len(test_batch_data)
    obj_test_hits1 += tim_hits1 * len(test_batch_data)
    obj_test_hits3 += tim_hits3 * len(test_batch_data)
    obj_test_hits10 += tim_hits10 * len(test_batch_data)

    # added eval_paper_authors
    # # # logging scores
    log = True
    if log == True:
        for index, quad in enumerate(test_batch_data):
            query_name, quad_array = evaluation_utils.query_name_from_quadruple_cygnet(quad, ob_pred=True)
            ground_truth = test_label[index].cpu().detach().numpy()
            scores_quad = test_score[index].cpu().detach().numpy()
            eval_paper_authors_logging_dict[query_name] = [scores_quad, ground_truth]
    # end added eval_paper_authors

obj_test_mrr = obj_test_mrr / test_data.shape[0]
obj_test_hits1 = obj_test_hits1 / test_data.shape[0]
obj_test_hits3 = obj_test_hits3 / test_data.shape[0]
obj_test_hits10 = obj_test_hits10 / test_data.shape[0]

print("test object-- Epoch {:04d} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
      format(checkpoint_obj['epoch'], obj_test_mrr, obj_test_hits1, obj_test_hits3, obj_test_hits10))


print("\nstart subject testing:")
# use best model checkpoint
checkpoint_sub = torch.load(model_state_file_sub)
# if use_cuda:
# model.cpu()  # test on CPU
model.train()
model.load_state_dict(checkpoint_sub['state_dict'])
print("Using best epoch: {}".format(checkpoint_sub['epoch']))

sub_test_mrr, sub_test_hits1, sub_test_hits3, sub_test_hits10 = 0, 0, 0, 0
n_batch = (test_data.shape[0] + batch_size - 1) // batch_size

for idx in range(n_batch):
    batch_start = idx * batch_size
    batch_end = min(test_data.shape[0], (idx + 1) * batch_size)
    test_batch_data = test_data[batch_start: batch_end]

    test_label = torch.LongTensor(test_batch_data[:, 0])
    seq_idx = test_batch_data[:, 2] * num_r + test_batch_data[:, 1]

    tail_seq = torch.Tensor(all_tail_seq_sub[seq_idx].todense())
    one_hot_tail_seq_sub = tail_seq.masked_fill(tail_seq != 0, 1)

    if use_cuda:
        test_label, one_hot_tail_seq_sub = test_label.to(device), one_hot_tail_seq_sub.to(device)
    test_score = model(test_batch_data, one_hot_tail_seq_sub, entity='subject')
    # modified by eval_paper_authors/ to integrate different filter settings
    if args.setting == 'raw':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_raw_mrr(test_score, test_label, hits=[1, 3, 10])
    elif args.setting == 'static':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_filtered_test_mrr(num_e, test_score,
                                                                           torch.LongTensor(
                                                                               train_data),
                                                                           torch.LongTensor(
                                                                               dev_data),
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           torch.LongTensor(
                                                                               test_batch_data),
                                                                           entity='subject',
                                                                           hits=[1, 3, 10])
    elif args.setting == 'time':
        tim_mrr, tim_hits1, tim_hits3, tim_hits10 = calc_time_filtered_test_mrr(num_e, test_score,
                                                                           torch.LongTensor(
                                                                               train_data),
                                                                           torch.LongTensor(
                                                                               dev_data),
                                                                           torch.LongTensor(
                                                                               test_data),
                                                                           torch.LongTensor(
                                                                               test_batch_data),
                                                                           entity='subject',
                                                                           hits=[1, 3, 10])
    # end modified by eval_paper_authors/ to integrate different filter settings
    sub_test_mrr += tim_mrr * len(test_batch_data)
    sub_test_hits1 += tim_hits1 * len(test_batch_data)
    sub_test_hits3 += tim_hits3 * len(test_batch_data)
    sub_test_hits10 += tim_hits10 * len(test_batch_data)

    # added eval_paper_authors
    # # # logging scores
    if log == True:
        for index, quad in enumerate(test_batch_data):
            query_name, quad_array = evaluation_utils.query_name_from_quadruple_cygnet(quad, ob_pred=False)
            ground_truth = test_label[index].cpu().detach().numpy()
            scores_quad = test_score[index].cpu().detach().numpy()
            eval_paper_authors_logging_dict[query_name] = [scores_quad, ground_truth]
    # end added eval_paper_authors

## ADDED eval_paper_authors fpr logging
if log == True:
    import pathlib
    import pickle
    method = 'cygnet'
    eval_paper_authors_datasetname = str(args.dataset)
    filter = str(args.setting)
    logname = method + '-' + eval_paper_authors_datasetname + '-' +str(exp_nr) + '-' +steps + '-' + filter
    dirname = os.path.join(pathlib.Path().resolve(), 'results' )
    eval_paper_authorsfilename = os.path.join(dirname, logname + ".pkl")
    # if not os.path.isfile(eval_paper_authorsfilename):
    with open(eval_paper_authorsfilename,'wb') as file:
        pickle.dump(eval_paper_authors_logging_dict, file, protocol=4) 
    file.close()
## END ADDED eval_paper_authors

sub_test_mrr = sub_test_mrr / test_data.shape[0]
sub_test_hits1 = sub_test_hits1 / test_data.shape[0]
sub_test_hits3 = sub_test_hits3 / test_data.shape[0]
sub_test_hits10 = sub_test_hits10 / test_data.shape[0]

print("test subject-- Epoch {:04d} | Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
      format(checkpoint_sub['epoch'], sub_test_mrr, sub_test_hits1, sub_test_hits3, sub_test_hits10))



test_mrr = (obj_test_mrr + sub_test_mrr) / 2
test_hits1 = (obj_test_hits1 + sub_test_hits1) / 2
test_hits3 = (obj_test_hits3 + sub_test_hits3) / 2
test_hits10 = (obj_test_hits10 + sub_test_hits10) / 2

print("\n\nfinal test --| Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}".
      format(test_mrr, test_hits1, test_hits3, test_hits10))


logging.debug('Train: {}\t Valid: {}\t Test: {}'.format(len(train_data), len(dev_data), len(test_data))) # modified eval_paper_authors: added for logging
logging.debug('Entity Prediction: --| Best MRR {:.4f} | Hits@1 {:.4f} | Hits@3 {:.4f} | Hits@10 {:.4f}'.
      format(test_mrr, test_hits1, test_hits3, test_hits10)) # modified eval_paper_authors: added for logging

print('end')
