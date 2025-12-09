import argparse
import torch
import torch.nn as nn
import sys
import random
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from config import PARAMS 
from trainer import *
from model.minkunext import model
#from model.segment_predictor import model_seg
from losses.truncated_smoothap import TruncatedSmoothAP
from losses.contrastive_loss import BatchHardContrastiveLossWithMasks
import time
from pytorch_metric_learning.distances import LpDistance
import matplotlib.pyplot as plt

def get_datetime():
    return time.strftime("%Y%m%d_%H%M")

def seed_everything(seed=42):
    # 1. Fijar semilla de Python (librerías base)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 2. Fijar semilla de NumPy (preprocesamiento)
    np.random.seed(seed)
    
    # 3. Fijar semilla de PyTorch (CPU y GPU)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si usas multi-GPU
    
    # 4. Configurar el backend de CUDNN (IMPORTANTE)
    # Desactivar la búsqueda de algoritmos óptimos (puede variar)
    torch.backends.cudnn.benchmark = False 
    # Forzar el uso de algoritmos deterministas (más lento, pero reproducible)
    torch.backends.cudnn.deterministic = True

def do_train(model):
    # Create model class
    if PARAMS.loss == 'TruncatedSmoothAP':
        loss_fn = TruncatedSmoothAP(tau1=PARAMS.tau1, similarity=PARAMS.similarity,
                                        positives_per_query=PARAMS.positives_per_query)
    else:
        distance = LpDistance(normalize_embeddings=PARAMS.normalize_embeddings)
        loss_fn = BatchHardContrastiveLossWithMasks(pos_margin=PARAMS.pos_margin, neg_margin=PARAMS.neg_margin,
                distance=distance)
    seed_everything(42)
    s = get_datetime()
    model_name = 'MinkUNeXt_' + PARAMS.protocol + '_' + s
    weights_path = create_weights_folder()
    model_pathname = os.path.join(weights_path, model_name)
    if PARAMS.print_model_info:
        print(model)
        n_params = sum([param.nelement() for param in model.parameters()])
        print('Number of model parameters: {}'.format(n_params))

    # Move the model to the proper device before configuring the optimizer
    device = torch.device(PARAMS.cuda_device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    model.to(device)
    memory_allocated = []
    memory_reserved = []
    print('Model device: {}'.format(device))

    # set up dataloaders
    dataloaders = make_dataloaders(device)

    # Training elements
    print('OPTIMIZER: ', PARAMS.optimizer)
    if PARAMS.optimizer == 'Adam':
        optimizer_fn = torch.optim.Adam
    elif PARAMS.optimizer == 'AdamW':
        optimizer_fn = torch.optim.AdamW
    else:
        raise NotImplementedError(f"Unsupported optimizer: {PARAMS.optimizer}")

    if PARAMS.weight_decay is None or PARAMS.weight_decay == 0:
        optimizer = optimizer_fn(model.parameters(), lr=PARAMS.initial_lr)
    else:
        optimizer = optimizer_fn(model.parameters(), lr=PARAMS.initial_lr, weight_decay=PARAMS.weight_decay)

    if PARAMS.scheduler is None:
        scheduler = None
    else:
        if PARAMS.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PARAMS.epochs+1,
                                                                   eta_min=PARAMS.initial_lr)
        elif PARAMS.scheduler == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, PARAMS.scheduler_milestones, gamma=0.1)
        else:
            raise NotImplementedError('Unsupported LR scheduler: {}'.format(PARAMS.scheduler))

    if PARAMS.batch_split_size is None or PARAMS.batch_split_size == 0:
        train_step_fn = training_step
    else:
        # Multi-staged training approach with large batch split into multiple smaller chunks with batch_split_size elems
        train_step_fn = multistaged_training_step

    ###########################################################################
    # Initialize Weights&Biases logging service
    ###########################################################################

    # Create a dictionary with the parameters
    params_dict = vars(PARAMS)
    wandb.init(project='MinkUNeXt', config=params_dict)

    ###########################################################################
    #
    ###########################################################################
    # Manual seeds...  

    # Establecer la semilla para PyTorch
    torch.manual_seed(42)

    # Si estás usando un GPU, también necesitas establecer la semilla para el backend CUDA
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Save seeds...
    # Obtener la semilla actual de PyTorch
    """torch_seed = torch.initial_seed()

    # Obtener la semilla de Python
    python_seed = random.getstate()

    # Obtener la semilla de NumPy
    numpy_seed = np.random.get_state()
    with open("seeds.txt", 'a') as f:
        line = model_name + ", Torch seed: " + str(torch_seed) +", Python: " + str(python_seed[1][0]) + ", Numpy: " + str(numpy_seed[1][0])
        f.write(line)"""
        
    # Training statistics
    stats = {'train': [], 'eval': []}

    if 'val' in dataloaders:
        # Validation phase
        phases = ['train', 'val']
        stats['val'] = []
    else:
        phases = ['train']

    for epoch in tqdm.tqdm(range(1, PARAMS.epochs + 1)):
        metrics = {'train': {}, 'val': {}}      # Metrics for wandb reporting
        first_iter = True
        for phase in phases:
            running_stats = []  # running stats for the current epoch and phase
            count_batches = 0

            if phase == 'train':
                global_iter = iter(dataloaders['train'])
            else:
                global_iter = None if dataloaders['val'] is None else iter(dataloaders['val'])

            while True:
                count_batches += 1
                batch_stats = {}
                if PARAMS.debug and count_batches > 2:
                    break

                try:
                    temp_stats, m_a, m_r = train_step_fn(global_iter, model, phase, device, optimizer, loss_fn)
                    if first_iter:
                        memory_allocated.append(m_a)
                        memory_reserved.append(m_r)
                        first_iter = False
                    batch_stats['global'] = temp_stats

                except StopIteration:
                    # Terminate the epoch when one of dataloders is exhausted
                    break

                running_stats.append(batch_stats)

            # Compute mean stats for the phase
            epoch_stats = {}
            for substep in running_stats[0]:
                epoch_stats[substep] = {}
                for key in running_stats[0][substep]:
                    temp = [e[substep][key] for e in running_stats]
                    if type(temp[0]) is dict:
                        epoch_stats[substep][key] = {key: np.mean([e[key] for e in temp]) for key in temp[0]}
                    elif type(temp[0]) is np.ndarray:
                        # Mean value per vector element
                        epoch_stats[substep][key] = np.mean(np.stack(temp), axis=0)
                    else:
                        epoch_stats[substep][key] = np.mean(temp)

            stats[phase].append(epoch_stats)
            print_stats(phase, epoch_stats)

            # Log metrics for wandb
            metrics[phase]['loss1'] = epoch_stats['global']['loss']
            if 'num_non_zero_triplets' in epoch_stats['global']:
                metrics[phase]['active_triplets1'] = epoch_stats['global']['num_non_zero_triplets']

            if 'positive_ranking' in epoch_stats['global']:
                metrics[phase]['positive_ranking'] = epoch_stats['global']['positive_ranking']

            if 'recall' in epoch_stats['global']:
                metrics[phase]['recall@1'] = epoch_stats['global']['recall'][1]

            if 'ap' in epoch_stats['global']:
                metrics[phase]['AP'] = epoch_stats['global']['ap']

        # ******* FINALIZE THE EPOCH *******
        wandb.log(metrics)
        #if epoch_stats['global']['recall'][1] == 1.0:
            #break
        if scheduler is not None:
            scheduler.step()

        #if params.save_freq > 0 and epoch % params.save_freq == 0:
        #    torch.save(model.state_dict(), model_pathname + "_" + str(epoch) + ".pth")

        if PARAMS.batch_expansion_th is not None:
            # Dynamic batch size expansion based on number of non-zero triplets
            # Ratio of non-zero triplets
            le_train_stats = stats['train'][-1]  # Last epoch training stats
            if 'num_non_zero_triplets' not in le_train_stats:
                print('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                rnz = le_train_stats['global']['num_non_zero_triplets'] / le_train_stats['global']['num_triplets']
                if rnz < PARAMS.batch_expansion_th:
                    dataloaders['train'].batch_sampler.expand_batch()

    print('')

    # Save final model weights
    final_model_path = model_pathname + '_final.pth'
    print(f"Saving weights: {final_model_path}")
    torch.save(model.state_dict(), final_model_path)
    
    # Generate GPU usage graphic
    """fig_name = "gpu_training_memory_"+model_pathname+".png"
    plt.plot(range(1, PARAMS.epochs + 1), memory_allocated, label="Allocated memory (MB)")
    plt.plot(range(1, PARAMS.epochs + 1), memory_reserved, label="Reserved memory (MB)")
    plt.xlabel("Epoch")
    plt.ylabel("GPU Memory(MB)")
    plt.title("GPU memory usage by minibatches during training")
    plt.legend()
    plt.grid(True)
    plt.savefig(fig_name, dpi=300)  # Save image
    plt.close()"""

    # Evaluate the final
    # PointNetVLAD datasets evaluation protocol
    stats = evaluate(model, device, log=False)
    print_eval_stats(stats)

    print('.')

    # Append key experimental metrics to experiment summary file
    prefix = "{}, {}".format(PARAMS.protocol, model_name)
    pnv_write_eval_stats("results.txt", prefix, stats)


if __name__ == '__main__':
    start_time = time.time()
    do_train(model)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"Tiempo de entrenamiento: {training_time} segundos")
