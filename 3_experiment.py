import os
import numpy as np
import pandas as pd
import json
from joblib import Parallel, delayed
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from scipy.stats import bootstrap
from models import (
    SVMModel, NaiveBayesModel, LogisticRegressionModel,
    KNNModel, XGBoostModel)

from features.dataset_construction import construct_enhanced_data


def run_experiment(filepath, model_type, algorithm, standardize=False, n_iterations=500, output_folder="output"):
    os.makedirs(output_folder, exist_ok=True)
    enhanced_data = construct_enhanced_data(filepath, model_type)

    features = enhanced_data.drop(columns=['season', 'date_game','team_1', 'team_2', 'game_result'])

    y = enhanced_data['game_result']

    results_baseline = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}
    results_enhanced = {'accuracy': [], 'f1': [], 'recall': [], 'precision': [], 'auroc': []}

    for _ in range(n_iterations):
        X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.2, stratify=y)
        
        baseline_features = ['avg_REB_Percent', 'avg_AS_Percent', 'avg_ST_Percent', 'avg_BS_Percent',
                             'avg_FOUL_Percent', 'avg_TO_Percent', 'avg_FG%_Percent', 'avg_eFG%_Percent',
                             'avg_TS%_Percent', 'avg_off_rtg', 'avg_drtg', 'team_1_elo', 'team_2_elo', 
                             'team_1_recent_elo_change', 'team_2_recent_elo_change']
        X_train_baseline = X_train.drop(columns=baseline_features)
        X_test_baseline = X_test.drop(columns=baseline_features)

        X_train_enhanced = X_train
        X_test_enhanced = X_test

        if standardize:
            scaler_baseline = StandardScaler()
            X_train_baseline = scaler_baseline.fit_transform(X_train_baseline)
            X_test_baseline = scaler_baseline.transform(X_test_baseline)
            
            scaler_enhanced = StandardScaler()
            X_train_enhanced = scaler_enhanced.fit_transform(X_train_enhanced)
            X_test_enhanced = scaler_enhanced.transform(X_test_enhanced)

        models = {
            "SVM": SVMModel(), "NaiveBayes": NaiveBayesModel(), "LogisticRegression": LogisticRegressionModel(),
            "KNN": KNNModel(), "XGBoost": XGBoostModel()
        }

        model_base = models[algorithm]
        model_base.train(X_train_baseline, y_train)
        y_pred_baseline = model_base.predict(X_test_baseline)
        y_prob_baseline = model_base.predict_proba(X_test_baseline)

        results_baseline['accuracy'].append(accuracy_score(y_test, y_pred_baseline))
        results_baseline['f1'].append(f1_score(y_test, y_pred_baseline))
        results_baseline['recall'].append(recall_score(y_test, y_pred_baseline))
        results_baseline['precision'].append(precision_score(y_test, y_pred_baseline))
        results_baseline['auroc'].append(roc_auc_score(y_test, y_prob_baseline))
        
        model_enhanced = models[algorithm]
        model_enhanced.train(X_train_enhanced, y_train)
        y_pred_enhanced = model_enhanced.predict(X_test_enhanced)
        y_prob_enhanced = model_enhanced.predict_proba(X_test_enhanced)

        results_enhanced['accuracy'].append(accuracy_score(y_test, y_pred_enhanced))
        results_enhanced['f1'].append(f1_score(y_test, y_pred_enhanced))
        results_enhanced['recall'].append(recall_score(y_test, y_pred_enhanced))
        results_enhanced['precision'].append(precision_score(y_test, y_pred_enhanced))
        results_enhanced['auroc'].append(roc_auc_score(y_test, y_prob_enhanced))

    avg_results_baseline = {metric: np.mean(scores) for metric, scores in results_baseline.items()}
    avg_results_enhanced = {metric: np.mean(scores) for metric, scores in results_enhanced.items()}

    results_summary = {
        "model_type": model_type,
        "algorithm": algorithm,
        "baseline": avg_results_baseline,
        "enhanced": avg_results_enhanced
    }

    output_path = os.path.join(output_folder, model_type, f"results_{model_type}_{algorithm}.json")
    with open(output_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    
    significance_summary = ""

    # print("\nSignificance Tests using Bootstrap:")
    significance_summary += "Significance Tests using Bootstrap:\n"
    n_bootstrap = 10000  
    for metric in ['accuracy', 'f1', 'recall', 'precision', 'auroc']:
        diff = np.array(results_enhanced[metric]) - np.array(results_baseline[metric])
        conf_int = bootstrap((diff,), np.mean, confidence_level=0.95, n_resamples=n_bootstrap, method='basic').confidence_interval
        is_significant = conf_int.low > 0  
        significance = "Significant" if is_significant else "Not Significant"
        # print(f"{metric.capitalize()} Significance (95% CI): {conf_int} - {significance}")
        significance_summary += f"{metric.capitalize()} Significance (95% CI): {conf_int} - {significance}\n"
    
    significance_summary += "\n" + "="*50 + "\n"
    
    output_path_significance = os.path.join(output_folder, model_type ,f"results_{model_type}_{algorithm}_significance.txt")
    with open(output_path_significance, "w") as f:
        f.write(significance_summary)
    
    print(f"{model_type} - {algorithm} finished!")


def run_parallel_experiment(df, model_type, algorithm):
    df_experiment = df.copy()
    run_experiment(df_experiment, model_type, algorithm, standardize=False, n_iterations=20, output_folder="output")

if __name__ == "__main__":
    df = pd.read_csv('./dados_jogos.csv')

    df = df[df['season']>2010]

    # Retirando os valores nulos das labels numéricas
    df.dropna(subset=['team_1_mp', 'team_1_fg',
       'team_1_fga', 'team_1_fg_pct', 'team_1_fg3', 'team_1_fg3a',
       'team_1_fg3_pct', 'team_1_ft', 'team_1_fta', 'team_1_ft_pct',
       'team_1_orb', 'team_1_drb', 'team_1_trb', 'team_1_ast', 'team_1_stl',
       'team_1_blk', 'team_1_tov', 'team_1_pf', 'team_1_pts', 'team_1_ts_pct',
       'team_1_efg_pct', 'team_1_fg3a_per_fga_pct', 'team_1_fta_per_fga_pct',
       'team_1_orb_pct', 'team_1_drb_pct', 'team_1_trb_pct', 'team_1_ast_pct',
       'team_1_stl_pct', 'team_1_blk_pct', 'team_1_tov_pct', 'team_1_usg_pct',
       'team_1_off_rtg', 'team_1_def_rtg', 'team_1_ws','team_2_mp',
       'team_2_fg', 'team_2_fga', 'team_2_fg_pct', 'team_2_fg3', 'team_2_fg3a',
       'team_2_fg3_pct', 'team_2_ft', 'team_2_fta', 'team_2_ft_pct',
       'team_2_orb', 'team_2_drb', 'team_2_trb', 'team_2_ast', 'team_2_stl',
       'team_2_blk', 'team_2_tov', 'team_2_pf', 'team_2_pts', 'team_2_ts_pct',
       'team_2_efg_pct', 'team_2_fg3a_per_fga_pct', 'team_2_fta_per_fga_pct',
       'team_2_orb_pct', 'team_2_drb_pct', 'team_2_trb_pct', 'team_2_ast_pct',
       'team_2_stl_pct', 'team_2_blk_pct', 'team_2_tov_pct', 'team_2_usg_pct',
       'team_2_off_rtg', 'team_2_def_rtg', 'team_2_ws'], inplace = True )
    
    # Fazendo com que as strings se tornem valores numéricos
    df['team_1_ws'] = df['team_1_ws'].apply(lambda x: int(x[1:].strip()) if x[0]=='W' else -int(x[1:].strip()))
    df['team_2_ws'] = df['team_2_ws'].apply(lambda x: int(x[1:].strip()) if x[0]=='W' else -int(x[1:].strip()))
    # Associando os resultados a labels numéricos
    df['game_result'] = df['game_result'].apply(lambda x: 1 if x.strip()=='W' else 0)

    # Criando as features relacionadas aos arremessos de 2 pontos 
    df['team_1_fg2'] = df['team_1_fg'] - df['team_1_fg3']
    df['team_1_fg2a'] = df['team_1_fga'] - df['team_1_fg3a']
    df['team_1_fg2_pct'] = df['team_1_fg2'] / df['team_1_fg2a']
    df['team_2_fg2'] = df['team_2_fg'] - df['team_2_fg3']
    df['team_2_fg2a'] = df['team_2_fga'] - df['team_2_fg3a']
    df['team_2_fg2_pct'] = df['team_2_fg2'] / df['team_2_fg2a']

    models_type = ['FourFactors', 'FourFactors_detailed', 'DefenseOfense', 'DefenseOfense_detailed']
    models_list =  ['LogisticRegression', 'XGBoost', 'KNN', 'NaiveBayes', 'SVM']
    
    combinations = list(itertools.product(models_type, models_list))

    
    Parallel(n_jobs=7)(delayed(run_parallel_experiment)(df, model_type, algorithm) for model_type, algorithm in combinations)