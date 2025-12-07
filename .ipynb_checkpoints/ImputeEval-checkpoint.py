import pandas as pd
import numpy as np
import torch
import process_edited as pce  # 'process_edited.py' 파일이 필요합니다.

from pypots.nn.functional import calc_mae
import torch.nn.functional as F
import os

torch.cuda.empty_cache()
torch.backends.cuda.preferred_linalg_library('magma')  # or 'cusolver'

# 'cuda:1'을 사용하도록 설정. 사용 가능하지 않으면 'cpu' 사용
# 이것이 '전역 device' 변수입니다.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# --- 모델 임포트 ---
from pypots.imputation import (
    iTransformer, PatchTST, TimeMixerPP, TimeLLM, MOMENT, TEFN, TimeMixer, GPT4TS, TimesNet, MICN, DLinear, FiLM, CSDI
)

# --- 경고 메시지 무시 ---
import warnings
warnings.filterwarnings("ignore")

# --- 데이터 로딩 클래스 임포트 ---
from dataprovider_pypots import ImpPypots, MultiImpPypots, ForePypots # 'dataprovider_pypots.py' 파일이 필요합니다.

# --- Imputation 평가 함수 정의 ---
# [수정 1] 맨 뒤에 main_device 매개변수 추가
def ImputeEvaluation(train_set, val_set, test_set, test_X_ori, indicating_mask, epochs, patience, verbose, num_mae_runs, main_device):
    # Global parameters derived from train_X
    n_steps = train_set["X"].shape[1]
    n_features = train_set["X"].shape[2]

    # --- 1. Define Model Classes and Their Configurations ---
    #model_classes_to_run = ["TimeMixerPP"]
    model_classes_to_run = ["PatchTST", "iTransformer", "TimeMixerPP"]
    #model_classes_to_run = ["TimeAutoDiff", "TimeLLM", "MOMENT", "TEFN", "TimeMixer", "TimesNet", "MICN", "DLinear", "FiLM", "CSDI", "PatchTST", "iTransformer", "TimeMixerPP"]
    
    # 모델 설정 딕셔너리
    # [수정 2] "device": device 를 "device": main_device 로 변경
    model_configs = {
        "TimeAutoDiff": {},
        
        "PatchTST": {
            "n_steps": n_steps,              # look-back window L
            "n_features": n_features,        # 데이터셋 feature 수
            "patch_size": 16,                # P
            "patch_stride": 8,               # S
            "n_layers": 3,                   # Transformer encoder layers
            "d_model": 128,                  # latent dim D
            "n_heads": 16,                   # attention heads
            "d_k": 8,                        # d_model / n_heads
            "d_v": 8,                        # 보통 d_k와 동일
            "d_ffn": 256,                    # FFN hidden dim F
            "dropout": 0.2,                  # encoder dropout
            "attn_dropout": 0.2,             # attention dropout
            "epochs": 1000,
            "patience": 200,
            "verbose": verbose,
            "device": main_device,           # main_device 사용
        },

        "iTransformer": {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": 3,
            "d_model": 128,
            "n_heads": 16,
            "d_k": 8,
            "d_v": 8,
            "d_ffn": 256,
            "dropout": 0.2,
            "attn_dropout": 0.2,
            "epochs": 1000,
            "patience": 200,
            "verbose": verbose,
            "device": main_device,           # main_device 사용
        },

        "TimeMixerPP": {
            "n_steps": n_steps, "n_features": n_features, "n_layers": 2,
            "d_model": 128, "d_ffn": 512, "top_k": 3, "n_heads": 4, "n_kernels": 1, "dropout": 0.1,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },
        
        "TimeLLM": {
            "n_steps": n_steps, "n_features": n_features, "llm_model_type": "GPT2",
            "n_layers": 1, "patch_size": 12, "patch_stride": 6, "d_llm": 768,
            "d_model": 256, "d_ffn": 512, "n_heads": 8, "dropout": 0.1,
            "domain_prompt_content": "Dataset has heterogeneous features both categorical and continuous.",
            "epochs": 50, "patience": 20, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "MOMENT": {
            "n_steps": n_steps, "n_features": n_features, "patch_size": 8, "patch_stride": 8,
            "transformer_backbone": 't5-small', "transformer_type": 'encoder_only',
            "n_layers": 1, "d_ffn": 512, "d_model": 512, "dropout": 0.1, "head_dropout": 0.1,
            "finetuning_mode": "linear-probing", "revin_affine": True,
            "add_positional_embedding": True, "value_embedding_bias": True, "orth_gain": 0.1,
            "epochs": 50, "patience": 20, "verbose": verbose, "device": 'cuda', # 하드코딩된 'cuda'는 그대로 둠
        },

        "TEFN": {
            "n_steps": n_steps, "n_features": n_features, "n_fod": 3,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "TimeMixer": {
            "n_steps": n_steps, "n_features": n_features, "n_layers": 3,
            "d_model": 512, "d_ffn": 1024, "top_k": 5,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "GPT4TS": {
            "n_steps": n_steps, "n_features": n_features, "patch_size": 1, "patch_stride": 1,
            "n_layers": 2, "train_gpt_mlp": True, "d_ffn": 16, "dropout": 0.1,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "TimesNet": {
            "n_steps": n_steps, "n_features": n_features, "n_layers": 3, "top_k": 3,
            "d_model": 64, "d_ffn": 64, "n_kernels": 6,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "MICN": {
            "n_steps": n_steps, "n_features": n_features, "n_layers": 2,
            "d_model": 512, "conv_kernel": [4, 8],
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "DLinear": {
            "n_steps": n_steps, "n_features": n_features, "moving_avg_window_size": 5,
            "individual": True, "d_model": 1024,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "FiLM": {
            "n_steps": n_steps, "n_features": n_features, "window_size": [2],
            "multiscale": [1, 2], "d_model": 1024,
            "epochs": epochs, "patience": patience, "verbose": verbose, "device": main_device, # main_device 사용
        },

        "CSDI": {
            "n_steps": n_steps, "n_features": n_features, "n_layers": 4, "n_heads": 8,
            "n_channels": 64, "d_time_embedding": 128, "d_feature_embedding": 16,
            "d_diffusion_embedding": 128,
            "epochs": 200, "patience": 25, "verbose": verbose, "device": main_device, # main_device 사용
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

        for i in range(num_mae_runs):
            print(f"  {model_name}: Run {i+1}/{num_mae_runs} (Initializing & Training)...")
            try:
                if model_name == "TimeAutoDiff":
                    # 이 블록 안의 'device = 'cuda'는 TimeAutoDiff만을 위한 '지역 변수'로 올바르게 작동합니다.
                    # 'main_device'와 충돌하지 않습니다.
                    device = 'cuda' 
                    import VAE as vae
                    import DIFF as diff

                    VAE_training = 3; diff_training = 3; lat_dim = 5
                    real_df1 = real_df.drop(['date'], axis=1) # 이 부분은 데이터셋에 'date'가 없을 경우 에러 발생 가능
                    
                    data_dict = data # data 객체를 data_dict로 사용

                    # VAE Training
                    n_epochs = VAE_training; eps = 1e-5
                    weight_decay = 1e-6; lr = 2e-4; hidden_size = 512; num_layers = 2; batch_size = 45;
                    channels = 64; min_beta = 1e-5; max_beta = 0.1; emb_dim = 128; time_dim = 8; threshold = 1; device = 'cuda'
                    ds = vae.train_autoencoder(real_df1, channels, hidden_size, num_layers, lr, weight_decay, n_epochs, batch_size, threshold, 
                                               min_beta, max_beta, emb_dim, time_dim, lat_dim, device, data_dict)
                    latent_features = ds[1]

                    # Diffusion Training
                    n_epochs = diff_training; hidden_dim = 512; num_layers = 2; diffusion_steps = 100; num_classes = len(latent_features)
                    Diff_model = diff.train_diffusion(latent_features, real_df1, data_dict, hidden_dim, num_layers, diffusion_steps, n_epochs, 
                                                num_classes)

                    # Sampling
                    target_mask_test=data_dict['target_mask_test']
                    target_test=data_dict['response_test']
                    cond_test=data_dict['cond_test']
                    time_info_test=data_dict['time_info_test']

                    diffusion_steps = 100
                    Batch_size, Seq_len, _ = target_test.shape
                    Lat_dim = lat_dim
                    t_grid = torch.linspace(0, 1, Seq_len).view(1, -1, 1).to(device)

                    samples = diff.sample(
                        t_grid.repeat(Batch_size, 1, 1),
                        Batch_size, Seq_len, Lat_dim, diffusion_steps, Diff_model,
                        time_info_test, cond_test,
                        target_mask_test.float().to(device),
                    )

                    gen_output = ds[0].decoder(samples.to(device), target_mask_test, cond_test)
                    _synth_data = pce.convert_to_tensor(real_df1, gen_output, 1, Batch_size, Seq_len)

                    mae = calc_mae(_synth_data.to('cpu').numpy(), target_test.to('cpu').numpy(), target_mask_test.numpy())
                    mse = F.mse_loss(_synth_data.to('cpu'), target_test.to('cpu'), reduction='mean').numpy()
                    torch.cuda.empty_cache()
                    print(mae, mse)

                else:
                    ModelClass = globals()[model_name]
                    model_instance = ModelClass(**config)
                    model_instance.fit(train_set, val_set)
                    imputation = model_instance.impute(test_set)

                    if model_name == "CSDI":
                        imputation_for_metrics = imputation.mean(axis=1)
                    else:
                        imputation_for_metrics = imputation

                    mae = calc_mae(imputation_for_metrics, test_X_ori.numpy(), indicating_mask.numpy())
                    mse = F.mse_loss(torch.as_tensor(imputation_for_metrics), torch.as_tensor(test_X_ori), reduction='mean').numpy()
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

# --- 스크립트 실행 메인 블록 ---
if __name__ == "__main__":
    
    # --- 실행할 데이터셋 목록 ---
    datasets_to_run = ["bike_sharing", "Metro_Traffic", "Pollution Data", "AirQuality"]
    #datasets_to_run = ["Pollution Data"]

    # --- 전역 실행 설정 ---
    evaluation_epochs = 2500
    evaluation_patience = 50
    evaluation_verbose = True
    num_runs = 1 # 각 모델 및 데이터셋 당 실행 횟수

    # --- 모든 결과를 저장할 마스터 딕셔너리 ---
    all_datasets_maes = {}
    all_datasets_mses = {}

    print(f"--- 🏃‍♂️ Starting Full Evaluation ---")
    print(f"Using device: {device}") # 전역 device 확인
    print(f"Datasets to process: {datasets_to_run}")
    print("----------------------------------------")

    # --- 데이터셋 루프 시작 ---
    for dataset in datasets_to_run:
        print(f"\n========================================")
        print(f" processing DATASET: {dataset} ")
        print(f"========================================")
        
        try:
            # --- 1. 데이터 로딩 및 전처리 ---
            print(f"Loading and preprocessing data for {dataset}...")
            real_df = pd.read_csv(f'./dataset/{dataset}.csv')
            
            # 데이터셋별 전처리
            if dataset == "bike_sharing":
                cols = real_df.columns.to_list()
                cols[1] = "date" 
                real_df.columns = cols
                real_df = real_df.drop(columns=["instant", "yr", "mnth"])
            if dataset == "Metro_Traffic":
                real_df["holiday"] = real_df["holiday"].fillna(0)
            if dataset == "Pollution Data":
                real_df["pm2.5"] = real_df["pm2.5"].fillna(0)
            # 'AirQuality'는 특별한 전처리 없음

            # ImpPypots를 사용한 데이터셋 생성
            #data = ImpPypots(real_df,
            #                 train_ratio=0.8,
            #                 val_ratio=0.1,
            #                 test_ratio=0.1,
            #                 seq_len=48,
            #                 stride=12,
            #                 rate=0.01,
            #                 pattern="block", #"point", "subseq", "block"
            #                 sub_seq_len=12,
            #                 block_len=3,
            #                 block_width=2)
            
            # 데이터셋 생성 for Forecasting
            data = ForePypots(real_df.iloc[0:5000],
                             train_ratio=0.8,
                             val_ratio=0.1,
                             test_ratio=0.1,
                             seq_len=48,
                             stride=1,
                             timewindow=36)
            
            print("Data keys available:", data.keys())

            # 데이터셋 분리
            train_X, val_X, test_X = data["train_X"], data["val_X"], data["test_X"]

            train_set = {"X": train_X}
            val_set = {
                "X": val_X,
                "X_ori": data["val_X_ori"],
            }
            test_set = {"X": test_X}
            test_X_ori = data["test_X_ori"]
            indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)

            print(f"Data for {dataset} prepared. n_steps: {train_X.shape[1]}, n_features: {train_X.shape[2]}")

            # --- 2. 해당 데이터셋으로 평가 실행 ---
            print(f"\n--- Starting Imputation Evaluation for {dataset} ---")
            
            # [수정 3] 함수 호출 시 'main_device=device' 추가
            results_maes, results_mses = ImputeEvaluation(
                train_set, val_set, test_set, test_X_ori, indicating_mask,
                epochs=evaluation_epochs,
                patience=evaluation_patience,
                verbose=evaluation_verbose,
                num_mae_runs=num_runs,
                main_device=device  # 전역 device 변수를 매개변수로 전달
            )
            
            # --- 3. 결과 저장 ---
            all_datasets_maes[dataset] = results_maes
            all_datasets_mses[dataset] = results_mses
            
            print(f"\n--- ✅ Completed Evaluation for {dataset} ---")

        except Exception as e:
            print(f"\n--- ❌ ERROR processing {dataset} ---")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            print("----------------------------------------")
            # 에러 발생 시 해당 데이터셋 결과는 비어있게 됨

    # --- 4. 모든 데이터셋 처리 후 최종 결과 요약 ---
    print("\n\n========================================")
    print(" 🏁 FINAL SUMMARY OF ALL RESULTS 🏁 ")
    print("========================================")

    print("\n--- 📊 Summary of MAE Results ---")
    for dataset_name, model_results in all_datasets_maes.items():
        print(f"\n=== DATASET: {dataset_name} ===")
        for model_name, mae_list in model_results.items():
            valid_scores = [s for s in mae_list if not np.isnan(s)]
            if valid_scores:
                print(f"  🔹 {model_name}:")
                print(f"     All {len(valid_scores)} MAEs: {[f'{s:.4f}' for s in valid_scores]}")
                print(f"     Mean MAE: {np.mean(valid_scores):.4f}")
                print(f"     Std Dev MAE: {np.std(valid_scores):.4f}")
            else:
                print(f"  🔹 {model_name}: No valid MAE scores collected.")
    
    print("\n\n--- 📊 Summary of MSE Results ---")
    for dataset_name, model_results in all_datasets_mses.items():
        print(f"\n=== DATASET: {dataset_name} ===")
        for model_name, mse_list in model_results.items():
            valid_scores = [s for s in mse_list if not np.isnan(s)]
            if valid_scores:
                print(f"  🔹 {model_name}:")
                print(f"     All {len(valid_scores)} MSEs: {[f'{s:.4f}' for s in valid_scores]}")
                print(f"     Mean MSE: {np.mean(valid_scores):.4f}")
                print(f"     Std Dev MSE: {np.std(valid_scores):.4f}")
            else:
                print(f"  🔹 {model_name}: No valid MSE scores collected.")

    print("\n--- All evaluations complete. ---")
