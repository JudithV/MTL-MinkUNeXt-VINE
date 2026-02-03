# Code adapted from MinkLoc3Dv2 repo: https://github.com/jac99/MinkLoc3Dv2.git
# Institute for Engineering Research of Elche (I3E)
# Automation, Robotics and Computer Vision lab (ARCV)
# Author: Judith Vilella Cantos

import os
import numpy as np
import torch
from config import PARAMS
import tqdm
import pathlib
import wandb
from losses.contrastive_loss import BatchHardContrastiveLossWithMasks
from losses.matryoshka import MatryoshkaLoss
from datasets.dataset_utils import make_dataloaders
from pnv_evaluate import evaluate, print_eval_stats, pnv_write_eval_stats
from pytorch_metric_learning.distances import LpDistance, CosineSimilarity


def print_global_stats(phase, stats):
    s = f"{phase}  loss: {stats['loss']:.4f}   embedding norm: {stats['avg_embedding_norm']:.3f}  "
    if 'num_triplets' in stats:
        s += f"Triplets (all/active): {stats['num_triplets']:.1f}/{stats['num_non_zero_triplets']:.1f}  " \
             f"Mean dist (pos/neg): {stats['mean_pos_pair_dist']:.3f}/{stats['mean_neg_pair_dist']:.3f}   "
    if 'positives_per_query' in stats:
        s += f"#positives per query: {stats['positives_per_query']:.1f}   "
    if 'best_positive_ranking' in stats:
        s += f"best positive rank: {stats['best_positive_ranking']:.1f}   "
    if 'recall' in stats:
        s += f"Recall@1: {stats['recall'][1]:.4f}   "
    if 'ap' in stats:
        s += f"AP: {stats['ap']:.4f}   "

    print(s)

def median_frequency_balancing(labels, num_classes):
    labels = np.asarray(labels)

    counts = np.bincount(labels, minlength=num_classes)
    print(f"Labels: {labels}")
    print(f"Counts: {counts}")
    median_count = np.median(counts[counts > 0])

    weights = median_count / counts
    weights[counts == 0] = 0.0  # safety

    return weights

def print_stats(phase, stats):
    print_global_stats(phase, stats['global'])


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


def training_step(global_iter, model, phase, device, optimizer, loss_fn):
    assert phase in ['train', 'val']

    batch, positives_mask, negatives_mask = next(global_iter)
    batch = {e: batch[e].to(device) for e in batch}
    if phase == 'train':
        model.train()
    else:
        model.eval()

    optimizer.zero_grad()
    loss_fn_matryoshka = MatryoshkaLoss(loss_fn, dims=[64, 128, 192], weights=[1.0, 0.5, 0.25])
    with torch.set_grad_enabled(phase == 'train'):
        y = model(batch)

        stats = model.stats.copy() if hasattr(model, 'stats') else {}

        embeddings = y['global']

        loss, temp_stats = loss_fn_matryoshka(embeddings, positives_mask, negatives_mask)
        if PARAMS.use_cross_entropy:
            ce_loss = torch.tensor(0.0, device=device)
            if 'logits' in y and 'labels' in batch:
                logits = y['logits']           # [B, num_labels]
                labels = batch['labels'].squeeze().long() # [B]
                labels = labels[y['batch_idx']]
                weight_classes = median_frequency_balancing(labels.cpu(), 3)
                weight_classes = torch.tensor(weight_classes, dtype=torch.float32, device=logits.device)

                ce_fn = torch.nn.CrossEntropyLoss(weight=weight_classes)
                ce_loss = ce_fn(logits, labels)
                stats['ce_loss'] = ce_loss.item()  # opcional, para logs
                lambda_ce = PARAMS.cross_entropy_importance
                loss = loss + lambda_ce * ce_loss
        temp_stats = tensors_to_numbers(temp_stats)
        stats.update(temp_stats)
        if phase == 'train':
            loss.backward()
            optimizer.step()

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    return stats


def multistaged_training_step(global_iter, model, phase, device, optimizer, loss_fn):
    # Training step using multistaged backpropagation algorithm as per:
    # "Learning with Average Precision: Training Image Retrieval with a Listwise Loss"
    # This method will break when the model contains Dropout, as the same mini-batch will produce different embeddings.
    # Make sure mini-batches in step 1 and step 3 are the same (so that BatchNorm produces the same results)
    # See some exemplary implementation here: https://gist.github.com/ByungSun12/ad964a08eba6a7d103dab8588c9a3774

    assert phase in ['train', 'val']
    batch, positives_mask, negatives_mask = next(global_iter)
    loss_fn_matryoshka = MatryoshkaLoss(loss_fn, dims=[64, 128, 192], weights=[1.0, 0.5, 0.25])
    if phase == 'train':
        model.train()
    else:
        model.eval()

    # Stage 1 - calculate descriptors of each batch element (with gradient turned off)
    # In training phase network is in the train mode to update BatchNorm stats
    embeddings_l = []
    if PARAMS.use_cross_entropy:
        clustering_logits_l = []
        labels_l = []
        batch_idx_l = []
    
    with torch.set_grad_enabled(False):
        for minibatch in batch:
            minibatch = {e: minibatch[e].to(device) for e in minibatch}
            y = model(minibatch)
            embeddings_l.append(y['global'])
            if PARAMS.use_cross_entropy and 'labels' in minibatch:
                logits, batch_idx = y['logits'], y['batch_idx']
                labels_mb = minibatch['labels'].squeeze().long().to(logits.device)

                labels_for_logits = labels_mb[batch_idx]

                clustering_logits_l.append(logits)
                labels_l.append(labels_for_logits)

    torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors

    # Stage 2 - compute gradient of the loss w.r.t embeddings
    embeddings = torch.cat(embeddings_l, dim=0)
    
    embeddings_grad = None
    clustering_logits_grad = None

    if PARAMS.use_cross_entropy:
        clustering_logits = torch.cat(clustering_logits_l, dim=0)
        labels = torch.cat(labels_l, dim=0) 

    with torch.set_grad_enabled(phase == 'train'):
        if phase == 'train':
            embeddings.requires_grad_(True)
            if PARAMS.use_cross_entropy:
                clustering_logits.requires_grad_(True)

        loss, stats = loss_fn_matryoshka(embeddings, positives_mask, negatives_mask)
        stats = tensors_to_numbers(stats)

        total_loss = loss
        
        if PARAMS.use_cross_entropy:
            weight_classes = median_frequency_balancing(labels.cpu(), 3)
            weight_classes = torch.tensor(weight_classes, dtype=torch.float32, device=clustering_logits.device)

            ce_fn = torch.nn.CrossEntropyLoss(weight=weight_classes)
            ce_loss = ce_fn(clustering_logits, labels.long())
            
            total_loss = total_loss + (PARAMS.cross_entropy_importance * ce_loss)

        if phase == 'train':
            total_loss.backward()
            
            embeddings_grad = embeddings.grad
            if PARAMS.use_cross_entropy:
                clustering_logits_grad = clustering_logits.grad

    # Clean
    embeddings_l, embeddings, loss = None, None, None
    if PARAMS.use_cross_entropy:
        clustering_logits = None

    # ---------------- Stage 3: Backpropagation through Backbone ----------------
    if phase == 'train':
        optimizer.zero_grad()
        i = 0
        with torch.set_grad_enabled(True):
            for minibatch in batch:
                minibatch = {e: minibatch[e].to(device) for e in minibatch}
                y = model(minibatch)
                
                # Output Global
                embeddings_mb = y['global']
                minibatch_size = embeddings_mb.shape[0]
                
                if embeddings_grad is not None:
                    retain_graph = PARAMS.use_cross_entropy
                    embeddings_mb.backward(gradient=embeddings_grad[i: i+minibatch_size], retain_graph=retain_graph)

                if PARAMS.use_cross_entropy:
                    if clustering_logits_grad is not None:
                        clustering_logits_mb = y['logits']
                        clustering_logits_mb.backward(gradient=clustering_logits_grad[i: i+minibatch_size])

                i += minibatch_size

            optimizer.step()

    torch.cuda.empty_cache()
    return stats, 0, 0


def create_weights_folder():
    # Create a folder to save weights of trained models
    this_file_path = pathlib.Path(__file__).parent.absolute()
    temp, _ = os.path.split(this_file_path)
    weights_path = os.path.join(temp, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
    return weights_path
