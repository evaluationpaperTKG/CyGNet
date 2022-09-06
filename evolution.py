

import torch
import numpy as np

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(score, labels, hits=[]):
    with torch.no_grad():

        ranks = sort_and_rank(score, labels)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_h = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for h in range(num_entities):
        if (h, target_r, target_t) not in triplets_to_filter:
            filtered_h.append(h)
    return torch.LongTensor(filtered_h)

def filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities):
    target_h, target_r, target_t = int(target_h), int(target_r), int(target_t)
    filtered_t = []

    # Do not filter out the test triplet, since we want to predict on it
    if (target_h, target_r, target_t) in triplets_to_filter:
        triplets_to_filter.remove((target_h, target_r, target_t))
    # Do not consider an object if it is part of a triplet to filter
    for t in range(num_entities):
        if (target_h, target_r, t) not in triplets_to_filter:
            filtered_t.append(t)
    return torch.LongTensor(filtered_t)

def get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity):
    """ Perturb object in the triplets
    """
    num_entities = num_entity
    ranks = []

    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]
        # print('t',target_t)
        if entity == 'object':
            filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(num_entity, score, train_triplets, valid_triplets, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()

def calc_filtered_test_mrr(num_entity, score, train_triplets, valid_triplets, valid_triplets2, test_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_triplets[:, 0]
        r = test_triplets[:, 1]
        t = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])
        valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])
        valid_triplets2 = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets2])
        test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets])

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, valid_triplets2, test_triplets]).tolist()

        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_filtered_rank(num_entity, score, h, r, t, test_size, triplets_to_filter, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


#######################################################################
#
# Modifications for time aware filtering eval_paper_authors
#
# some lines commented
# additionally also renamed a few variable to have a better understanding;
# original variables can be a it confusing and did not modify anything on those functions.

#######################################################################

def get_time_filtered_rank(num_entity, score, h, r, t, test_size, test_triplets, entity):
    # Some major modifications here

    num_entities = num_entity
    ranks = []
    for idx in range(test_size):
        target_h = h[idx]
        target_r = r[idx]
        target_t = t[idx]

        # subset based on query timestamp
        current_timestamp = test_triplets[idx, 3]
        triplets_to_filter = test_triplets[test_triplets[:, 3] == current_timestamp]

        # somewhat using old logics here to get results with least manipulations.
        # this logic is computationally expensive according to eval_paper_authors and should be changed (if required)
        triplets_to_filter = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in triplets_to_filter])  # test set
        triplets_to_filter = torch.cat([triplets_to_filter]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        if entity == 'object':
            filtered_t = filter_t(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_t_idx = int((filtered_t == target_t).nonzero())
            _, indices = torch.sort(score[idx][filtered_t], descending=True)  # filtered_t are entities
            rank = int((indices == target_t_idx).nonzero())
        if entity == 'subject':
            filtered_h = filter_h(triplets_to_filter, target_h, target_r, target_t, num_entities)
            target_h_idx = int((filtered_h == target_h).nonzero())
            _, indices = torch.sort(score[idx][filtered_h], descending=True)
            rank = int((indices == target_h_idx).nonzero())

        ranks.append(rank)
    return torch.LongTensor(ranks)


def calc_time_filtered_test_mrr(num_entity, score, train_triplets, valid_triplets, test_triplets, test_batch_triplets, entity, hits=[]):
    with torch.no_grad():
        h = test_batch_triplets[:, 0]
        r = test_batch_triplets[:, 1]
        t = test_batch_triplets[:, 2]
        test_size = test_batch_triplets.shape[0]

        # train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])  # train set
        # valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])  # valid set
        # test_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_triplets]) # test set
        # test_batch_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in test_batch_triplets])  # test batch set
        #
        # Why would they want to add the batch triples again which already exist in test_triplets.
        # triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets, test_batch_triplets]).tolist()

        ranks = get_time_filtered_rank(num_entity, score, h, r, t, test_size, test_triplets, entity)

        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


def calc_time_filtered_mrr(num_entity, score, train_triplets, valid_triplets, valid_batch_triplets, entity, hits=[]):
    with torch.no_grad():
        h = valid_batch_triplets[:, 0]
        r = valid_batch_triplets[:, 1]
        t = valid_batch_triplets[:, 2]
        test_size = valid_batch_triplets.shape[0]


        # train_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in train_triplets])  #train set
        # valid_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_triplets])  # valid set
        # valid_batch_triplets = torch.Tensor([[quad[0], quad[1], quad[2]] for quad in valid_batch_triplets])  # valid batch set
        # triplets_to_filter = torch.cat([train_triplets, valid_triplets, valid_batch_triplets]).tolist()
        # triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}

        ranks = get_time_filtered_rank(num_entity, score, h, r, t, test_size, valid_triplets, entity)

        ranks += 1  # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())

        hits1 = torch.mean((ranks <= hits[0]).float())
        hits3 = torch.mean((ranks <= hits[1]).float())
        hits10 = torch.mean((ranks <= hits[2]).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()
