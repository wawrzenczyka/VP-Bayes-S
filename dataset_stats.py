# %%
from data_loading.vae_pu_dataloaders import get_dataset

stats = []
for dataset in [
        'MNIST 3v5', 
        'MNIST OvE',
        'CIFAR CarTruck', 
        'CIFAR MachineAnimal', 
        'STL MachineAnimal', 
        'Gas Concentrations'
    ]:
    train_samples, val_samples, test_samples, label_frequency, pi_p, n_input = \
        get_dataset(dataset, 'cpu', 0.1)
    n = len(train_samples[0]) + len(val_samples[0]) + len(test_samples[0])

    stats.append({'Name': dataset, 'Samples': n, 'Features': n_input, 'Class prior $\pi$': f'{pi_p:.2f}'})

# %%
import pandas as pd
df = pd.DataFrame.from_records(stats)
df

# %%
print(
    df.to_latex(
        index=False, 
        caption='Benchmark Dataset Statistics',
        label='tab:dataset-stats',
        escape=False,
        column_format='lrrr', 
        position='tbp'
    )
)

# %%
