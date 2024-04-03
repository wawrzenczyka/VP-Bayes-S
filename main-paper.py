# %%
import argparse
import json
import threading
import multiprocessing
import os
import time
import numpy as np
import torch
import tensorflow as tf
from config import config
from vae_pu_occ.vae_pu_occ_trainer import VaePuOccTrainer
from data_loading.vae_pu_dataloaders import create_vae_pu_adapter, get_dataset
from external.LBE import train_LBE, eval_LBE
from external.sar_experiment import SAREMThreadedExperiment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform VAE-PU+OCC experiments described in the paper.')
    parser.add_argument('--method', dest='method', choices=['VAE-PU+OCC', 'VAE-PU', 'VAE-PU-orig', 'SAR-EM'],
                        default='VAE-PU+OCC', help='Method selected for experiments, should correspond to one of the files in configs/datasets directory')
    parser.add_argument('--dataset', dest='dataset', choices=['MNIST 3v5', 'MNIST OvE', 'CIFAR CarTruck', 'CIFAR MachineAnimal', 'STL MachineAnimal', 'Gas Concentrations', 'CIFAR MachineAnimal SCAR'],
                        default='MNIST 3v5', help='Dataset name, should correspond to one of the files in configs/methods directory')

    args = parser.parse_args()

    with open(os.path.join('configs', 'base_config.json'), 'r') as f:
        config = json.load(f)
    with open(os.path.join('configs', 'methods', f'{args.method}.json'), 'r') as f:
        method_config = json.load(f)
    with open(os.path.join('configs', 'datasets', f'{args.dataset}.json'), 'r') as f:
        dataset_config = json.load(f)
    with open(os.path.join('configs', 'custom_settings.json'), 'r') as f:
        custom_settings = json.load(f)

    config.update(method_config)
    config.update(dataset_config)
    config.update(custom_settings)

    if 'SCAR' in config['data']:
        config['use_SCAR'] = True
    else:
        config['use_SCAR'] = False

    if config['use_original_paper_code']:
        config['mode'] = 'near_o'
    else:
        config['mode'] = 'near_y'

    # used by SAR-EM
    n_threads = multiprocessing.cpu_count()
    sem = threading.Semaphore(n_threads)
    threads = []

    for idx in range(config['start_exp_idx'], config['start_exp_idx'] + config['num_experiments']):
        for base_label_frequency in config['label_frequencies']:
            config['base_label_frequency'] = base_label_frequency

            np.random.seed(idx)
            torch.manual_seed(idx)
            tf.random.set_seed(idx)

            if config['device'] == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                config['device'] = device
            else:
                device = config['device']
            
            train_samples, val_samples, test_samples, label_frequency, pi_p, n_input = \
                get_dataset(config['data'], device, base_label_frequency, use_scar_labeling=config['use_SCAR'])
            vae_pu_data = \
                create_vae_pu_adapter(train_samples, val_samples, test_samples, device)

            config['label_frequency'] = label_frequency
            config['pi_p'] = pi_p
            config['n_input'] = n_input

            config['pi_pl'] = label_frequency * pi_p
            config['pi_pu'] = pi_p - config['pi_pl']
            config['pi_u'] = 1 - config['pi_pl']

            batch_size = config['batch_size_train']
            pl_batch_size = int(np.ceil(config['pi_pl'] * batch_size))
            u_batch_size = batch_size - pl_batch_size
            config['batch_size_l'], config['batch_size_u'] = (pl_batch_size, u_batch_size)
            config['batch_size_l_pn'], config['batch_size_u_pn'] = (pl_batch_size, u_batch_size)

            config['directory'] = os.path.join('result', config['data'], str(base_label_frequency), 'Exp' + str(idx))
            
            if config['training_mode'] == 'VAE-PU':
                trainer = VaePuOccTrainer(num_exp=idx, model_config=config, pretrain=True)
                trainer.train(vae_pu_data)
            else:
                np.random.seed(idx)
                torch.manual_seed(idx)
                tf.random.set_seed(idx)
                method_dir = os.path.join(config['directory'], 'external', config['training_mode'])

                if config['training_mode'] == 'SAR-EM':
                    exp_thread = SAREMThreadedExperiment(train_samples, test_samples, idx, base_label_frequency, config, method_dir, sem)
                    exp_thread.start()
                    threads.append(exp_thread)
                if config['training_mode'] == 'LBE':
                    log_prefix = f'Exp {idx}, c: {base_label_frequency} || '

                    lbe_training_start = time.perf_counter()
                    lbe = train_LBE(train_samples, val_samples, verbose=True, log_prefix=log_prefix)
                    lbe_training_time = time.perf_counter() - lbe_training_start

                    accuracy, precision, recall, f1 = eval_LBE(lbe, test_samples, verbose=True, log_prefix=log_prefix)
                    metric_values = {
                        'Method': 'LBE',
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1 score': f1,
                        'Time': lbe_training_time,
                    }
                    
                    os.makedirs(method_dir, exist_ok=True)
                    with open(os.path.join(method_dir, 'metric_values.json'), 'w') as f:
                        json.dump(metric_values, f)

    for t in threads:
        t.join()
