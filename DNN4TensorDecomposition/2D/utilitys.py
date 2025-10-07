import os
import time
import torch
import numpy as np
import datetime
import pandas as pd

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

def save_loss_to_csv(losses_data, path):
    """
    
        Save lost data as a CSV file, compatible with two formats:
        1. K loop format: k_losses = {k: {'mse': [], 'ortho': [], 'total': []}}
        2. Direct format: losses = {'mse': [], 'ortho': [], 'total': []}}
        
        parameter:
        losses_data: Loss data, which can be a k_losses dictionary or a losses dictionary
        path: save the path
        
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
        
    else:
        # Direct format: losses = {'mse': [], 'ortho': [], 'total': []}
        print("Saving direct mode losses...")
        df_all = pd.DataFrame({
            'epoch': range(len(losses_data['mse'])),
            'mse_loss': losses_data['mse'],
            'ortho_loss': losses_data['ortho'],
            'total_loss': losses_data['total']
        })
    
    # Save to CSV file
    csv_filename = 'all_losses.csv' if 'k' in df_all.columns else 'losses.csv'
    df_all.to_csv(os.path.join(path, csv_filename), index=False)
    print(f"Losses saved to {csv_filename}")