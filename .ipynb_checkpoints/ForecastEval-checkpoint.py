import pandas as pd
import numpy as np
import torch
import process_edited as pce

from pypots.nn.functional import calc_mae, calc_mse
import torch.nn.functional as F

import os

torch.cuda.empty_cache()
torch.backends.cuda.preferred_linalg_library('magma')   # or 'cusolver'

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') # This is your global device
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Import all models you intend to use
from pypots.forecasting import (
    TimeLLM, MOMENT, TEFN, TimeMixer, GPT4TS, TimesNet, MICN, DLinear, FiLM, CSDI
    # Uncomment ModernTCN only IF you have fixed its TypeError!
    # from pypots.imputation import ModernTCN
)

# --- Suppress Warnings ---
import warnings
warnings.filterwarnings("ignore")

# --- Data Loading and Preprocessing ---
from dataprovider_pypots import ForePypots

real_df = pd.read_csv('./dataset/bike_sharing.csv')

#Metro_Traffic.csv
real_df["holiday"] = real_df["holiday"].fillna(0)

# Pollution Data.csv
# real_df["pm2.5"] = real_df["pm2.5"].fillna(0)

#bike_sharing.csv 
cols = real_df.columns.to_list()
cols[1] = "date" 
real_df.columns = cols
real_df = real_df.drop(columns=["instant", "yr", "mnth"])

print("Loading and preprocessing data...")
N_PRED_STEPS = 36

data = ForePypots(real_df,
                 train_ratio=0.8,
                 val_ratio=0.1,
                 test_ratio=0.1,
                 seq_len=48,
                 stride=1,
                 timewindow=12)

print("Data keys available:", data.keys())

# for forecasting task, we feed only N_STEPS data into the model and let it predict the following N_PRED_STEPS
FORECASTING_TRAIN_SET = {
    "X": data["train_X"][:, :-N_PRED_STEPS, :],
    "X_pred": data["train_X_ori"][:, -N_PRED_STEPS:, :],
}
train_set = FORECASTING_TRAIN_SET
FORECASTING_VAL_SET = {
    "X": data["val_X"][:, :-N_PRED_STEPS, :],
    "X_pred": data["val_X_ori"][:, -N_PRED_STEPS:, :],
}
val_set = FORECASTING_VAL_SET
FORECASTING_TEST_SET = {
    "X": data["test_X"][:, :-N_PRED_STEPS, :],
    "X_pred": data["test_X_ori"][:, -N_PRED_STEPS:, :],
}
test_set = FORECASTING_TEST_SET

# --- Imputation Evaluation Function ---
def ForecastingEvaluation(train_set, val_set, test_set, data, epochs, patience, verbose, num_mae_runs):
    # Global parameters derived from train_X
    n_steps = data["train_X_ori"].shape[1]-N_PRED_STEPS
    n_features = data["train_X_ori"].shape[2]

    # --- 1. Define Model Classes and Their Configurations ---
    #model_classes_to_run = ["TimeAutoDiff", "TEFN", "TimeMixer", "TimesNet", "MICN", "DLinear", "FiLM", "CSDI", "TimeLLM"]
    model_classes_to_run = ["CSDI"]
    
    # Dictionary of configurations for each model.
    # IMPORTANT: 'device' here will now correctly refer to the global 'device' variable.
    model_configs = {
        "TimeAutoDiff": {},
        
        "TimeLLM": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "term": "short", "llm_model_type": "GPT2",
            "n_layers": 1, "patch_size": 12, "patch_stride": 6, "d_llm": 768,
            "d_model": 256, "d_ffn": 512, "n_heads": 8, "dropout": 0.1,
            "domain_prompt_content": "Dataset has heterogeneous features both categorical and continuous.",
            "epochs": 500, "patience": 50, "verbose": verbose, "device": device, # Uses global device
        },

        "MOMENT": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "term": "short", "patch_size": 8, "patch_stride": 8,
            "transformer_backbone": 't5-small', "transformer_type": 'encoder_only',
            "n_layers": 1, "d_ffn": 512, "d_model": 512, "dropout": 0.1, "head_dropout": 0.1,
            "finetuning_mode": "linear-probing", "revin_affine": True,
            "add_positional_embedding": True, "value_embedding_bias": True, "orth_gain": 0.1,
            "epochs": 50, "patience": 20, "verbose": verbose, "device": 'cuda', # Uses global device
        },

        "TEFN": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "n_fod": 3, "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "TimeMixer": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "term": "short", "n_layers": 3, "d_model": 512, "d_ffn": 1024, "top_k": 5,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "GPT4TS": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "patch_size": 1, "patch_stride": 1, "n_layers": 2, "train_gpt_mlp": True, "d_ffn": 16, "dropout": 0.1,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "TimesNet": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "n_layers": 3, "top_k": 3, "d_model": 64, "d_ffn": 64, "n_kernels": 6,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "MICN": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features,  
            "n_layers": 2, "d_model": 512, "conv_kernel": [4, 8],
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "DLinear": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features,
            "moving_avg_window_size": 5, "individual": True, "d_model": 1024,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "FiLM": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "window_size": [2],
            "multiscale": [1, 2], "d_model": 1024,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": device, # Uses global device
        },

        "CSDI": {
            "n_steps": n_steps, "n_features": n_features, "n_pred_steps": N_PRED_STEPS, "n_pred_features": n_features, 
            "n_layers": 4, "n_heads": 8, "n_channels": 64, "d_time_embedding": 128, "d_feature_embedding": 16,
            "d_diffusion_embedding": 128,
            "epochs": 500, "patience": 50, "verbose": verbose, "device": device, # Uses global device
        },
    }

    all_models_mae_results = {}
    all_models_mse_results = {} 

    # --- 2. Loop Through Each Model Class ---
    for model_name in model_classes_to_run:
        print(f"\n--- Processing {model_name} ---")

        config = model_configs.get(model_name)
        if config is None:
            print(f"Error: No configuration found for {model_name}. Skipping.")
            continue

        current_model_mae_scores = []
        current_model_mse_scores = [] 

        try:
            if model_name == "TimeAutoDiff":
                import VAE as vae
                import DIFF as diff

                VAE_training = 30000; diff_training = 30000; lat_dim = 8
                real_df1 = real_df.drop(['date'], axis=1)

                # Construct `data_dict` for VAE and DIFF from your `data` object
                data_dict = data

                ############ Auto-encoder Training ############
                n_epochs = VAE_training; eps = 1e-5
                weight_decay = 1e-6; lr = 2e-4; hidden_size = 512; num_layers = 2; batch_size = 50
                channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8; threshold = 1
                ds = vae.train_autoencoder(real_df1, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, min_beta, max_beta, emb_dim, time_dim, lat_dim, 'cuda', data_dict)
                latent_features = ds[1]

                ############ Diffusion Training ############
                n_epochs = diff_training; hidden_dim = 512; num_layers = 2; diffusion_steps = 100
                num_classes = latent_features.shape[2]
                Diff_model = diff.train_diffusion(latent_features, real_df1, data_dict, hidden_dim, num_layers, diffusion_steps, n_epochs, num_classes)

                for i in range(num_mae_runs):
                    print(f"  {model_name}: Run {i+1}/{num_mae_runs} (Initializing & Training)...")
                    
                    ############ Sampling ############
                    target_mask_test=data_dict['target_mask_test']
                    target_test=data_dict['response_test']
                    cond_test=data_dict['cond_test']
                    time_info_test=data_dict['time_info_test']

                    # Sampling process
                    diffusion_steps = 100
                    Batch_size, Seq_len, _ = target_test.shape
                    Lat_dim = lat_dim
                    t_grid = torch.linspace(0, 1, Seq_len).view(1, -1, 1).to('cuda') # Use global device

                    samples = diff.sample(
                        t_grid.repeat(Batch_size, 1, 1),
                        Batch_size,
                        Seq_len,
                        Lat_dim,
                        diffusion_steps,
                        Diff_model,
                        time_info_test,
                        cond_test,
                        target_mask_test.float().to('cuda'), # Use global device
                    )

                    # Process the generated data
                    gen_output = ds[0].decoder(samples.to('cuda'), target_mask_test, cond_test) # Use global device
                    _synth_data = pce.convert_to_tensor(real_df1, gen_output, 1, Batch_size, Seq_len)

                    # Calculate MAE and MSE
                    # Ensure _synth_data and target_test are on CPU and are numpy arrays for calc_mae_mse
                    mae = calc_mae(_synth_data.to('cpu').numpy(), target_test.to('cpu').numpy(), target_mask_test.numpy())
                    mse = F.mse_loss(_synth_data.to('cpu'), target_test.to('cpu'), reduction='mean').numpy()
                    torch.cuda.empty_cache()
                    print(mae, mse)
                    
                    current_model_mae_scores.append(mae)
                    current_model_mse_scores.append(mse)
                
            else:
                ModelClass = globals()[model_name]
                # Initialize the model for each run
                model_instance = ModelClass(**config)

                # Train the model
                model_instance.fit(FORECASTING_TRAIN_SET, FORECASTING_VAL_SET)

                for i in range(num_mae_runs):
                    print(f"  {model_name}: Run {i+1}/{num_mae_runs} (Initializing & Training)...")
                    # Impute missing values
                    forecasting = model_instance.predict(FORECASTING_TEST_SET)["forecasting"]

                    # Calculate MAE and MSE
                    if model_name == "CSDI":
                        # CSDI typically returns multiple samples, take the mean for evaluation
                        forecasting = forecasting.mean(axis=1)
                    else:
                        forecasting = forecasting

                    mae = calc_mae(forecasting, FORECASTING_TEST_SET["X_pred"].numpy(), ~np.isnan(FORECASTING_TEST_SET["X_pred"].numpy()))
                    mse = calc_mse(forecasting, FORECASTING_TEST_SET["X_pred"].numpy(), ~np.isnan(FORECASTING_TEST_SET["X_pred"].numpy()))

                    print(mae, mse)
                    torch.cuda.empty_cache()

                    current_model_mae_scores.append(mae)
                    current_model_mse_scores.append(mse)

        except Exception as e:
            print(f"  An error occurred during run {i+1} for {model_name}: {e}")
            import traceback
            traceback.print_exc()

        all_models_mae_results[model_name] = current_model_mae_scores
        all_models_mse_results[model_name] = current_model_mse_scores

    return all_models_mae_results, all_models_mse_results

# --- Execute the Evaluation ---
if __name__ == "__main__":
    evaluation_epochs = 5000
    evaluation_patience = 100
    evaluation_verbose = False
    num_runs = 5 # Renamed for clarity since it applies to both MAE and MSE

    print("\n--- Starting Imputation Evaluation ---")
    results_maes, results_mses = ForecastingEvaluation(
        train_set, val_set, test_set, data,
        epochs=evaluation_epochs,
        patience=evaluation_patience,
        verbose=evaluation_verbose,
        num_mae_runs=num_runs # Keep the parameter name as num_mae_runs for compatibility
    )

    print("\n--- Summary of MAE Results Across All Models and Runs ---")
    for model_name, mae_list in results_maes.items():
        valid_scores = [s for s in mae_list if not np.isnan(s)]
        if valid_scores:
            print(f"\n{model_name}:")
            print(f"  All {len(valid_scores)} MAEs: {[f'{s:.4f}' for s in valid_scores]}")
            print(f"  Mean MAE: {np.mean(valid_scores):.4f}")
            print(f"  Std Dev MAE: {np.std(valid_scores):.4f}")
        else:
            print(f"\n{model_name}: No valid MAE scores collected (all runs failed or were skipped).")

    print("\n--- Summary of MSE Results Across All Models and Runs ---")
    for model_name, mse_list in results_mses.items():
        valid_scores = [s for s in mse_list if not np.isnan(s)]
        if valid_scores:
            print(f"\n{model_name}:")
            print(f"  All {len(valid_scores)} MSEs: {[f'{s:.4f}' for s in valid_scores]}")
            print(f"  Mean MSE: {np.mean(valid_scores):.4f}")
            print(f"  Std Dev MSE: {np.std(valid_scores):.4f}")
        else:
            print(f"\n{model_name}: No valid MSE scores collected (all runs failed or were skipped).")
