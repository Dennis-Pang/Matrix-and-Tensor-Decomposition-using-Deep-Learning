import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime


def create_output_dir(model_name, dataset_name, rank, base_dir="outputs"):
    """
    
        Create an output directory with the naming format: model+data+rank+date(month and day)
        
        parameter:
        model_name: model name
        dataset_name: dataset name
        rank: rank value
        base_dir: base directory, default to "outputs"
        
        return:
        save_path: Complete save path
        dir_name: directory name
        
    """
    # Get the current date (month and day format)
    current_date = datetime.datetime.now().strftime('%m%d')
    
    # Create directory name
    dir_name = f"{model_name}_{dataset_name}_k{rank}_{current_date}"
    
    # Create a full path
    save_path = os.path.join(base_dir, dir_name)
    
    # Make sure the directory exists
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Created output directory: {save_path}")
    return save_path, dir_name


def log_experiment_details(details, path):
    """
    Record the detailed parameters of the experiment to a file.

    Parameters:
    details: A dictionary containing experimental parameters, which may include:
        - model: model name
        - dataset: dataset name
        - k/rank: rank value
        - base_channels: number of base channels
        - weight: weight parameter
        - temperature: temperature parameter
        - epochs: number of training epochs
        - optimizer: optimizer information
        - learning_rate: learning rate
        - batch_size: batch size
        - device: device information
        etc.
    path: save path
    """
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, 'experiment_details.txt')
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")
        f.write("EXPERIMENT PARAMETERS:\n")
        f.write("="*50 + "\n")
        
        # Record all experimental parameters
        for key, value in details.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("END OF EXPERIMENT DETAILS\n")
        f.write("="*50 + "\n")
    
    print(f"Experiment details saved to: {filepath}")



def plot_losses(losses_data, path):
    """
    Visualize training losses, compatible with two formats:
    1. K-loop format: k_losses = {k: {'mse': [], 'ortho': [], 'total': []}}
    2. Direct format: losses = {'mse': [], 'ortho': [], 'total': []}
    
    Parameters:
    losses_data: loss data, can be a k_losses dictionary or a losses dictionary
    path: save path
    """
    # Create a save directory
    os.makedirs(path, exist_ok=True)
    
    # Detect data format
    if isinstance(losses_data, dict) and isinstance(next(iter(losses_data.values())), dict):
        # K loop format: k_losses = {k: {'mse': [], 'ortho': [], 'total': []}}
        print("Plotting K-loop mode losses...")
        
        # Create Graph - Show loss curves for each k value
        plt.figure(figsize=(15, 10))
        
        # Create subgraphs for each loss type
        loss_types = ['total', 'mse', 'ortho']
        colors = plt.cm.tab10(range(len(losses_data)))
        
        for i, loss_type in enumerate(loss_types):
            plt.subplot(2, 2, i + 1)
            for j, k in enumerate(losses_data):
                epochs_offset = sum(len(losses_data[kk][loss_type]) for kk in sorted(losses_data.keys()) if kk < k)
                epochs = range(epochs_offset, epochs_offset + len(losses_data[k][loss_type]))
                plt.plot(epochs, losses_data[k][loss_type], color=colors[j], label=f'k={k}')
            
            plt.title(f'{loss_type.upper()} Loss by K')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.yscale('log')
            plt.grid(True)
            plt.legend()
        
        # Add a composite graph - the final loss of all k values
        plt.subplot(2, 2, 4)
        final_losses = {loss_type: [] for loss_type in loss_types}
        k_values = sorted(losses_data.keys())
        
        for k in k_values:
            for loss_type in loss_types:
                final_losses[loss_type].append(losses_data[k][loss_type][-1])
        
        for loss_type in loss_types:
            plt.plot(k_values, final_losses[loss_type], 'o-', label=f'{loss_type.upper()} Loss')
        
        plt.title('Final Loss vs K')
        plt.xlabel('K')
        plt.ylabel('Final Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
    else:
        # Direct format: losses = {'mse': [], 'ortho': [], 'total': []}
        print("Plotting direct mode losses...")
        plt.figure(figsize=(15, 5))
        
        # Total loss
        plt.subplot(131)
        plt.plot(losses_data['total'], color='blue', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        # MSE Loss
        plt.subplot(132)
        plt.plot(losses_data['mse'], color='red', label='MSE Loss')
        plt.title('MSE Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
        
        # Orthogonal loss
        plt.subplot(133)
        plt.plot(losses_data['ortho'], color='green', label='Orthogonality Loss')
        plt.title('Orthogonality Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Orthogonality')
        plt.yscale('log')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Save the image
    plt.savefig(os.path.join(path, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    print(f"Loss visualization saved to: {os.path.join(path, 'loss_curves.png')}")
    plt.close()

def save_loss_to_csv(losses_data, path):
    """
    Save loss data to a CSV file, compatible with two formats:
    1. K-loop format: k_losses = {k: {'mse': [], 'ortho': [], 'total': []}}
    2. Direct format: losses = {'mse': [], 'ortho': [], 'total': []}
    
    Parameters:
    losses_data: loss data, can be a k_losses dictionary or a losses dictionary
    path: save path
    """
    # Create a save directory
    os.makedirs(path, exist_ok=True)
    
    # Detect data format
    if isinstance(losses_data, dict) and isinstance(next(iter(losses_data.values())), dict):
        # K loop format: k_losses = {k: {'mse': [], 'ortho': [], 'total': []}}
        print("Saving K-loop mode losses...")
        all_losses = []
        for k in losses_data:
            df_k = pd.DataFrame({
                'k': k,
                'epoch': range(len(losses_data[k]['mse'])),
                'mse_loss': losses_data[k]['mse'],
                'ortho_loss': losses_data[k]['ortho'],
                'total_loss': losses_data[k]['total']
            })
            all_losses.append(df_k)
        
        # Merge all data
        df_all = pd.concat(all_losses, ignore_index=True)
        csv_filename = 'all_losses.csv'
        
    else:
        # Direct format: losses = {'mse': [], 'ortho': [], 'total': []}
        print("Saving direct mode losses...")
        df_all = pd.DataFrame({
            'epoch': range(len(losses_data['mse'])),
            'mse_loss': losses_data['mse'],
            'ortho_loss': losses_data['ortho'],
            'total_loss': losses_data['total']
        })
        csv_filename = 'losses.csv'
    
    # Save to CSV file
    csv_path = os.path.join(path, csv_filename)
    df_all.to_csv(csv_path, index=False)
    
    print(f"Losses saved to: {csv_path}")