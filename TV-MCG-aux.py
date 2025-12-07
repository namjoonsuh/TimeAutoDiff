#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import torch
import random
import matplotlib.pyplot as plt

import dataprovider_pypots as dp
import process_edited as pce
import VAE as vae
import DIFF as diff_mod   # DIFF 모듈
import Evaluation.Metrics as mt
import Evaluation.predictive_metrics as pdm

from dataprovider_pypots import (
    add_continuous_metadata_noise_subset,
    flip_categorical_metadata_subset,
)

device = "cuda"


def main():
    # ------------------------
    # 1. 데이터셋 설정
    # ------------------------
    # time_col은 실제 데이터셋 column 이름에 맞게 확인해서 수정해줘야 함!
    dataset_configs = {
        #"Metro_Traffic": {
        #     "path": "dataset/Metro_Traffic.csv",
        #     "reader": pd.read_csv,
        #     "target_columns": ["weather_main", "temp"],
        #     "time_col": "date",       # 실제 time 컬럼명 확인 필요
        #},
        #"Pollution Data": {
        #     "path": "dataset/Pollution Data.csv",
        #     "reader": pd.read_csv,
        #     "target_columns": ["cbwd", "Iws"],
        #     "time_col": "date",
        #},
        #"Hurricane": {
        #    "path": "dataset/Hurricane.csv",
        #    "reader": pd.read_csv,
        #    "target_columns": ["year", "trend"],
        #    "time_col": "date",  # 실제 time 컬럼명에 맞게 수정
        #},
        "AirQuality": {
             "path": "dataset/AirQuality.csv",
             "reader": pd.read_csv,
             "target_columns": ["NOx(GT)", "NO2(GT)"],
             "time_col": "date",       # 예시: 실제 컬럼명 확인 필요
        },
        "ETTh1": {
             "path": "dataset/ETTh1.csv",
             "reader": pd.read_csv,
             "target_columns": ["LULL", "OT"],
             "time_col": "date",       # 실제 컬럼명에 맞게 수정
        },
        #"Energy": {
        #     "path": "dataset/energy_data.csv",
        #     "reader": pd.read_csv,
        #     "target_columns": ["lights", "rv1"],
        #     "time_col": "date",       # 실제 컬럼명에 맞게 수정
        #},
    }

    # ------------------------
    # 2. 공통 하이퍼파라미터
    # ------------------------
    VAE_training = 2000
    diff_training = 50000
    lat_dim = 6

    vae_n_epochs = VAE_training
    eps = 1e-5
    weight_decay = 1e-6
    lr = 2e-4
    hidden_size = 512
    num_layers_vae = 2
    batch_size = 100
    channels = 64
    min_beta = 1e-5
    max_beta = 0.1
    emb_dim = 128
    time_dim = 8
    threshold = 1

    diff_n_epochs = diff_training
    hidden_dim = 200
    num_layers_diff = 2
    diffusion_steps = 100

    # robustness 실험 설정
    levels = [0.0, 0.1, 0.2, 0.4, 0.6]
    iterations = 1000
    num_runs = 5

    seq_len = 24
    stride = 1
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # ------------------------
    # 3. 결과 저장용 딕셔너리
    # ------------------------
    disc_results = {}  # {dataset_name: {"mean": [...], "std": [...]} }
    temp_results = {}

    # ------------------------
    # 4. 데이터셋 루프
    # ------------------------
    for ds_name, cfg in dataset_configs.items():
        print("\n\n==============================")
        print(f"=== Dataset: {ds_name} ===")
        print("==============================")

        # 4-1) 데이터 로딩
        reader = cfg["reader"]
        path = cfg["path"]
        target_columns = cfg["target_columns"]
        time_col = cfg["time_col"]

        real_df = reader(path)

        # --- 데이터셋별 간단 전처리 (필요시 조건문 추가) ---
        if ds_name == "bike_sharing":
            cols = real_df.columns.to_list()
            cols[1] = "date"
            real_df.columns = cols
            real_df = real_df.drop(columns=["instant", "yr", "mnth"])

        if ds_name == "Metro_Traffic":
            if "holiday" in real_df.columns:
                real_df["holiday"] = real_df["holiday"].fillna(0)

        if ds_name == "Pollution Data":
            if "pm2.5" in real_df.columns:
                real_df["pm2.5"] = real_df["pm2.5"].fillna(0)

        # time_col이 실제로 존재하는지 체크 (없으면 에러)
        assert (
            time_col in real_df.columns
        ), f"{ds_name}: time_col '{time_col}' not found in columns!"

        # 4-2) TVMCG용 데이터 준비
        real_df1 = real_df.drop([time_col], axis=1)
        parser_tmp = pce.DataFrameParser().fit(real_df1, 1)
        column_name = parser_tmp.column_name()

        pdset = dp.TVMCG(
            real_df.iloc[0:2000, :],  # 필요시 길이 조정
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seq_len=seq_len,
            stride=stride,
            column_list=column_name,
            target_column=target_columns,
        )

        # 4-3) VAE 학습
        ds = vae.train_autoencoder_best_train(
            None,
            channels,
            hidden_size,
            num_layers_vae,
            lr,
            weight_decay,
            vae_n_epochs,
            batch_size,
            threshold,
            min_beta,
            max_beta,
            emb_dim,
            time_dim,
            lat_dim,
            device,
            pdset,
        )
        latent_features = ds[1]

        # 4-4) Diffusion 학습
        num_classes = len(latent_features)
        diff_model = diff_mod.train_diffusion(
            latent_features,
            None,
            pdset,
            hidden_dim,
            num_layers_diff,
            diffusion_steps,
            diff_n_epochs,
            num_classes,
        )

        # 4-5) Robustness 실험 준비
        Batch_size, Seq_len, _ = pdset["response_train"].shape
        Lat_dim = lat_dim

        base_target_mask = pdset["target_mask_train"]
        base_target = pdset["response_train"]
        base_cond = pdset["cond_train"]
        base_time_info = pdset["time_info_train"]

        target_idx = pdset["target_idx"]
        cont_meta_idx = pdset["cond_cont_idx"]
        cat_meta_idx = pdset["cond_cat_idx"]
        cat_num_classes = pdset["cat_num_classes_idx"]

        disc_mean = []
        disc_std = []
        tmp_mean = []
        tmp_std = []

        # 4-6) level 루프
        for level in levels:
            print(f"\n=== {ds_name} | Level {level} ===")
            disc_scores = []
            tmp_scores = []

            for run in range(num_runs):
                print(f"  Run {run+1}/{num_runs}...")

                # (1) metadata corruption
                cond_clean = base_cond.clone()
                cond_test_noisy = add_continuous_metadata_noise_subset(
                    cond=cond_clean,
                    cont_idx=cont_meta_idx,
                    level=level,
                )
                cond_test_flipped = flip_categorical_metadata_subset(
                    cond=cond_test_noisy,
                    cat_idx=cat_meta_idx,
                    p=level,
                    num_classes_dict=cat_num_classes,
                )

                # (2) 준비
                target_mask_test = base_target_mask.to(device).float()
                target_test = base_target.to(device)
                cond_test = cond_test_flipped.to(device)
                time_info_test = base_time_info.to(device)

                # (3) Sampling
                t_grid = torch.linspace(0, 1, Seq_len).view(1, -1, 1).to(device)
                samples = diff_mod.sample(
                    t_grid.repeat(Batch_size, 1, 1),
                    Batch_size,
                    Seq_len,
                    Lat_dim,
                    diffusion_steps,
                    diff_model,
                    time_info_test,
                    cond_test,
                    target_mask_test,
                )

                # (4) Decode
                parser = pdset["parser"]  # TVMCG에서 저장한 parser
                gen_output = ds[0].decoder(
                    samples.to(device), target_mask_test, cond_test
                )
                _synth_data = pce.convert_to_tensor(
                    parser, gen_output, 1, Batch_size, Seq_len
                )

                # (5) target feature만 사용
                target_test1 = target_test.detach().cpu()
                _synth_data1 = _synth_data.detach().cpu()

                # (6) Metrics
                disc_score = mt.discriminative_score_metrics(
                    target_test1, _synth_data1, iterations
                )
                tmp_score = mt.temp_disc_score(
                    target_test1, _synth_data1, iterations
                )

                if torch.is_tensor(disc_score):
                    disc_score = float(disc_score)
                if torch.is_tensor(tmp_score):
                    tmp_score = float(tmp_score)

                disc_scores.append(disc_score)
                tmp_scores.append(tmp_score)

            # level별 평균 / 표준편차
            disc_mean.append(np.mean(disc_scores))
            disc_std.append(np.std(disc_scores))
            tmp_mean.append(np.mean(tmp_scores))
            tmp_std.append(np.std(tmp_scores))

            print(
                f"→ {ds_name} | Disc mean ± std: {disc_mean[-1]:.4f} ± {disc_std[-1]:.4f}"
            )
            print(
                f"→ {ds_name} | Temp  mean ± std: {tmp_mean[-1]:.4f} ± {tmp_std[-1]:.4f}"
            )

        # 데이터셋별 결과 저장
        disc_results[ds_name] = {"mean": disc_mean, "std": disc_std}
        temp_results[ds_name] = {"mean": tmp_mean, "std": tmp_std}

    # ------------------------
    # 5. 최종 플롯 (데이터셋별 곡선 같이 보기)
    # ------------------------

    # Plot A: Discriminative Score
    plt.figure(figsize=(7, 5))
    for ds_name in dataset_configs.keys():
        dm = disc_results[ds_name]["mean"]
        ds_ = disc_results[ds_name]["std"]
        plt.errorbar(
            levels,
            dm,
            yerr=ds_,
            marker="o",
            capsize=3,
            label=ds_name,
        )

    plt.xlabel("Metadata corruption level (p)")
    plt.ylabel("Discriminative Score")
    plt.title(f"Discriminative Score vs Metadata Corruption\n(mean of {num_runs} runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot B: Temporal Score
    plt.figure(figsize=(7, 5))
    for ds_name in dataset_configs.keys():
        tm = temp_results[ds_name]["mean"]
        ts = temp_results[ds_name]["std"]
        plt.errorbar(
            levels,
            tm,
            yerr=ts,
            marker="s",
            capsize=3,
            label=ds_name,
        )

    plt.xlabel("Metadata corruption level (p)")
    plt.ylabel("Temporal Score")
    plt.title(f"Temporal Score vs Metadata Corruption\n(mean of {num_runs} runs)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # (선택) 숫자 테이블 출력
    print("\n=== Summary (Discriminative) ===")
    print("Dataset | Level | Mean | Std")
    for ds_name in dataset_configs.keys():
        for l, m, s in zip(
            levels,
            disc_results[ds_name]["mean"],
            disc_results[ds_name]["std"],
        ):
            print(f"{ds_name} | {l:.2f} | {m:.4f} | {s:.4f}")

    print("\n=== Summary (Temporal) ===")
    print("Dataset | Level | Mean | Std")
    for ds_name in dataset_configs.keys():
        for l, m, s in zip(
            levels,
            temp_results[ds_name]["mean"],
            temp_results[ds_name]["std"],
        ):
            print(f"{ds_name} | {l:.2f} | {m:.4f} | {s:.4f}")


if __name__ == "__main__":
    main()

