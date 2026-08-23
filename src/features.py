import time
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss, roc_auc_score
from codecarbon import EmissionsTracker
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold
from imblearn.under_sampling import NearMiss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def kfold_and_gridsearch(numero_folds, X, y, porcentagem_reducao):
    
    param_KNN = {
        'n_neighbors': [3, 5, 7, 15, 30, 50],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    param_LogisticRegression = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    }
    
    param_RandomForest = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 5, 10],
        'criterion': ['gini']
    }
    
    param_LightGBM = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 63]
    }
    param_XGBoost = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 6, 9],
    'subsample': [0.8, 1.0]
    }
    param_MLP = {
        'hidden_layer_sizes': [(64,), (128,)],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [200, 300]
    }
    
    kfold = KFold(n_splits=numero_folds, shuffle=True, random_state=42)

    metricas_KNN = []
    metricas_LR = []
    metricas_RF = []
    metricas_LightGBM = []
    metricas_MLP = []
    metricas_RNA = []
    metricas_XGBoost = []

    contador = 1

    for train_index, test_index in kfold.split(X):
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train_filtrado = remover_features_por_mi(X_train, y_train, porcentagem_reducao)
        
        features_selecionadas = X_train_filtrado.columns
        
        X_test_filtrado = X_test[features_selecionadas]

        X_train_divided, X_val, y_train_divided, y_val = train_test_split(X_train_filtrado, y_train, test_size=0.2, random_state=42, stratify=y_train)
        
        scaler = MinMaxScaler()
        X_train_divided = scaler.fit_transform(X_train_divided)
        X_val = scaler.transform(X_val)
        
        X_train_full = scaler.fit_transform(X_train_filtrado)
        X_test_full = scaler.transform(X_test_filtrado)

        metricas_val_KNN = []
        par_KNN = []

        print(f"Começando GRID KNN fold {contador}")
              
        for params in ParameterGrid(param_KNN):
            knn = KNeighborsClassifier(n_neighbors=params['n_neighbors'], metric=params['metric'])                       
            resultado_validacao_KNN = treinar_com_codecarbon(knn,X_train_divided, y_train_divided,X_val,y_val, porcentagem_reducao)

            resultado_validacao_KNN['fold'] = contador
            resultado_validacao_KNN['etapa'] = 'grid_search'
            resultado_validacao_KNN['params'] = params
            
            metricas_val_KNN.append(resultado_validacao_KNN)
            par_KNN.append(params)

        melhor_resultado = max(metricas_val_KNN, key=lambda x: x['f1'])
        melhores_params_knn = par_KNN[metricas_val_KNN.index(melhor_resultado)]

        print("Treinando KNN FINAL")
                                      
        knn_best = KNeighborsClassifier(metric=melhores_params_knn['metric'], n_neighbors=melhores_params_knn['n_neighbors'])
        resultado_teste_final = treinar_com_codecarbon(knn_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final['fold'] = contador
        resultado_teste_final['etapa'] = 'treino_final'
        resultado_teste_final['params'] = melhores_params_knn

        print(f"RESULTADO FINAL KNN (FOLD {resultado_teste_final['fold']})\n")
        print(f"ACC: {resultado_teste_final['acc']}")
        print(f"PRECISION: {resultado_teste_final['prec']}")
        print(f"RECALL: {resultado_teste_final['rec']}")
        print(f"F1: {resultado_teste_final['f1']}")
        
        resultado_teste_final['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_KNN)
        resultado_teste_final['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_KNN)
        
        metricas_KNN.append(resultado_teste_final)

        print(f"Começando GRID LR fold {contador}")
        metricas_val_LR = []
        par_LR = []

        for params in ParameterGrid(param_LogisticRegression):
            lr = LogisticRegression(
                C=params['C'], 
                penalty=params['penalty'], 
                solver=params['solver'], 
                max_iter=1000, 
                random_state=42
            )
            resultado_validacao = treinar_com_codecarbon(lr, X_train_divided, y_train_divided, X_val, y_val, porcentagem_reducao)

            resultado_validacao['fold'] = contador
            resultado_validacao['etapa'] = 'grid_search'
            resultado_validacao['params'] = params
            
            metricas_val_LR.append(resultado_validacao)
            par_LR.append(params)

        melhor_resultado_lr = max(metricas_val_LR, key=lambda x: x['f1'])
        melhores_params_lr = par_LR[metricas_val_LR.index(melhor_resultado_lr)]

        print("Treinando LR FINAL")
                                      
        lr_best = LogisticRegression(
            C=melhores_params_lr['C'], 
            penalty=melhores_params_lr['penalty'], 
            solver=melhores_params_lr['solver'], 
            max_iter=1000, 
            random_state=42
        )
        resultado_teste_final = treinar_com_codecarbon(lr_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final['fold'] = contador
        resultado_teste_final['etapa'] = 'treino_final'
        resultado_teste_final['params'] = melhores_params_lr

        print(f"RESULTADO FINAL LR (FOLD {resultado_teste_final['fold']})\n")
        print(f"ACC: {resultado_teste_final['acc']}")
        print(f"PRECISION: {resultado_teste_final['prec']}")
        print(f"RECALL: {resultado_teste_final['rec']}")
        print(f"F1: {resultado_teste_final['f1']}")
        
        resultado_teste_final['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_LR)
        resultado_teste_final['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_LR)
        
        metricas_LR.append(resultado_teste_final)

        print(f"Começando GRID RF fold {contador}")
        metricas_val_RF = []
        par_RF = []

        for params in ParameterGrid(param_RandomForest):
            rf = RandomForestClassifier(
                n_estimators=params['n_estimators'], 
                max_depth=params['max_depth'], 
                criterion=params['criterion'], 
                random_state=42,
                n_jobs=-1
            )
            resultado_validacao = treinar_com_codecarbon(rf, X_train_divided, y_train_divided, X_val, y_val, porcentagem_reducao)

            resultado_validacao['fold'] = contador
            resultado_validacao['etapa'] = 'grid_search'
            resultado_validacao['params'] = params
            
            metricas_val_RF.append(resultado_validacao)
            par_RF.append(params)

        melhor_resultado_rf = max(metricas_val_RF, key=lambda x: x['f1'])
        melhores_params_rf = par_RF[metricas_val_RF.index(melhor_resultado_rf)]

        print("Treinando RF FINAL")
                                      
        rf_best = RandomForestClassifier(
            n_estimators=melhores_params_rf['n_estimators'], 
            max_depth=melhores_params_rf['max_depth'], 
            criterion=melhores_params_rf['criterion'], 
            random_state=42,
            n_jobs=-1
        )
        resultado_teste_final = treinar_com_codecarbon(rf_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final['fold'] = contador
        resultado_teste_final['etapa'] = 'treino_final'
        resultado_teste_final['params'] = melhores_params_rf
        
        resultado_teste_final['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_RF)
        resultado_teste_final['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_RF)
        
        metricas_RF.append(resultado_teste_final)

        print(f"Começando GRID LightGBM fold {contador}")
        metricas_val_LightGBM = []
        par_LightGBM = []

        for params in ParameterGrid(param_LightGBM):
            lgb = LGBMClassifier(
                n_estimators=params['n_estimators'], 
                learning_rate=params['learning_rate'], 
                num_leaves=params['num_leaves'], 
                random_state=42,
                n_jobs=-1,
                verbosity=-1                                             
            )
            resultado_validacao = treinar_com_codecarbon(lgb, X_train_divided, y_train_divided, X_val, y_val, porcentagem_reducao)

            resultado_validacao['fold'] = contador
            resultado_validacao['etapa'] = 'grid_search'
            resultado_validacao['params'] = params
            
            metricas_val_LightGBM.append(resultado_validacao)
            par_LightGBM.append(params)

        melhor_resultado_lgb = max(metricas_val_LightGBM, key=lambda x: x['f1'])
        melhores_params_lgb = par_LightGBM[metricas_val_LightGBM.index(melhor_resultado_lgb)]

        print("Treinando LightGBM FINAL")
                                      
        lgb_best = LGBMClassifier(
            n_estimators=melhores_params_lgb['n_estimators'], 
            learning_rate=melhores_params_lgb['learning_rate'], 
            num_leaves=melhores_params_lgb['num_leaves'], 
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        resultado_teste_final = treinar_com_codecarbon(lgb_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final['fold'] = contador
        resultado_teste_final['etapa'] = 'treino_final'
        resultado_teste_final['params'] = melhores_params_lgb
        
        resultado_teste_final['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_LightGBM)
        resultado_teste_final['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_LightGBM)
        
        metricas_LightGBM.append(resultado_teste_final)

        print(f"Começando GRID XGBoost fold {contador}")
        metricas_val_XGB = []
        par_XGB = []

        for params in ParameterGrid(param_XGBoost):
            xgb = XGBClassifier(
                n_estimators=params['n_estimators'], 
                learning_rate=params['learning_rate'],
                max_depth=params['max_depth'],
                subsample=params['subsample'],
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            resultado_validacao_XGB = treinar_com_codecarbon(xgb, X_train_divided, y_train_divided, X_val, y_val, porcentagem_reducao)

            resultado_validacao_XGB['fold'] = contador
            resultado_validacao_XGB['etapa'] = 'grid_search'
            resultado_validacao_XGB['params'] = params
            
            metricas_val_XGB.append(resultado_validacao_XGB)
            par_XGB.append(params)

        melhor_resultado_xgb = max(metricas_val_XGB, key=lambda x: x['f1'])
        melhores_params_xgb = par_XGB[metricas_val_XGB.index(melhor_resultado_xgb)]

        print("Treinando XGBoost FINAL")
                                      
        xgb_best = XGBClassifier(
            n_estimators=melhores_params_xgb['n_estimators'], 
            learning_rate=melhores_params_xgb['learning_rate'],
            max_depth=melhores_params_xgb['max_depth'],
            subsample=melhores_params_xgb['subsample'],
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        resultado_teste_final_xgb = treinar_com_codecarbon(xgb_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final_xgb['fold'] = contador
        resultado_teste_final_xgb['etapa'] = 'treino_final'
        resultado_teste_final_xgb['params'] = melhores_params_xgb
        
        resultado_teste_final_xgb['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_XGB)
        resultado_teste_final_xgb['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_XGB)
        
        metricas_XGBoost.append(resultado_teste_final_xgb)

        print(f"Começando GRID MLP fold {contador}")
        metricas_val_MLP = []
        par_MLP = []

        for params in ParameterGrid(param_MLP):
            mlp = MLPClassifier(
                hidden_layer_sizes=params['hidden_layer_sizes'],
                alpha=params['alpha'],
                max_iter=params['max_iter'],
                random_state=42
            )
            resultado_validacao = treinar_com_codecarbon(mlp, X_train_divided, y_train_divided, X_val, y_val, porcentagem_reducao)

            resultado_validacao['fold'] = contador
            resultado_validacao['etapa'] = 'grid_search'
            resultado_validacao['params'] = params
            
            metricas_val_MLP.append(resultado_validacao)
            par_MLP.append(params)

        melhor_resultado_mlp = max(metricas_val_MLP, key=lambda x: x['f1'])
        melhores_params_mlp = par_MLP[metricas_val_MLP.index(melhor_resultado_mlp)]

        print("Treinando MLP FINAL")
                                      
        mlp_best = MLPClassifier(
            hidden_layer_sizes=melhores_params_mlp['hidden_layer_sizes'],
            alpha=melhores_params_mlp['alpha'],
            max_iter=melhores_params_mlp['max_iter'],
            random_state=42
        )
        resultado_teste_final = treinar_com_codecarbon(mlp_best, X_train_full, y_train, X_test_full, y_test, porcentagem_reducao)
        
        resultado_teste_final['fold'] = contador
        resultado_teste_final['etapa'] = 'treino_final'
        resultado_teste_final['params'] = melhores_params_mlp

        print(f"RESULTADO FINAL MLP (FOLD {resultado_teste_final['fold']})\n")
        print(f"ACC: {resultado_teste_final['acc']}")
        print(f"PRECISION: {resultado_teste_final['prec']}")
        print(f"RECALL: {resultado_teste_final['rec']}")
        print(f"F1: {resultado_teste_final['f1']}")
        
        resultado_teste_final['emissions_grid_total_g'] = sum(x['emissions_g'] for x in metricas_val_MLP)
        resultado_teste_final['energia_grid_total_kwh'] = sum(x['energia_kwh'] for x in metricas_val_MLP)
        
        metricas_MLP.append(resultado_teste_final)

        contador+=1

    return {
        "KNN": metricas_KNN,
        "LR": metricas_LR,
        "RF": metricas_RF,
        "LightGBM": metricas_LightGBM,
        "XGBoost": metricas_XGBoost,
        "MLP": metricas_MLP
    }

def treinar_com_codecarbon(modelo, X_train, y_train, X_test, y_test, porcentagem_reducao):

    tracker = EmissionsTracker(
        output_dir="../reports/codecarbon", 
        output_file=f"emissoes_detalhadas_codecarbon_{porcentagem_reducao}%.csv",
        log_level="error",
        allow_multiple_runs=True
    )
    start_time = time.time()

    tracker.start()
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
    finally:
        emissions_kg = tracker.stop()

    emissions_g = emissions_kg * 1000
    energia = tracker._total_energy.kWh
    tempo = time.time() - start_time

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    rga = emissions_g / acc
    
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "emissions_g": emissions_g, "energia_kwh": energia, "tempo_s": tempo, "rga": rga
    }

def remover_features_por_mi(X, y, porcentagem_remocao):
        
    if porcentagem_remocao == 0:
        return X.copy()
    
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=True)
    
    n_features_total = len(X.columns)
    n_remover = int(n_features_total * (porcentagem_remocao / 100.0))
    
    features_para_remover = mi_ranking.head(n_remover).index.tolist()
    
    X_filtrado = X.drop(columns=features_para_remover)
    
    print(f"--- Seleção de Features via MI Score ---")
    print(f"Features originais: {n_features_total}")
    print(f"Features removidas ({porcentagem_remocao}%): {len(features_para_remover)}")
    print(f"Features restantes: {len(X_filtrado.columns)}")
    
    return X_filtrado

def reducao_codecarbon(porcentagem_reducao, X, y, numero_folds):
    print(f"Redução de {porcentagem_reducao}%")

    kfold = kfold_and_gridsearch(numero_folds, X, y, porcentagem_reducao)

    return kfold

def aumento_codecarbon(porcentagem_aumento, X, y, numero_folds, dataset):
    print(f"Aumento de {porcentagem_aumento}%")

    if dataset == 'bondora':
        X_aumentado = aumento_features_bondora(X, porcentagem_aumento)
    elif dataset == 'german':
        X_aumentado = aumentar_features_german(X, porcentagem_aumento)
    else:
        X_aumentado = aumentar_features_xyz(X, porcentagem_aumento)

    kfold = kfold_and_gridsearch(numero_folds, X_aumentado, y, porcentagem_aumento)
    return kfold

def signed_log(series):
    """Retorna sign(x) * log1p(abs(x)) para lidar com negativos sem gerar NaN."""
    arr = series.to_numpy(dtype=float)
    return np.sign(arr) * np.log1p(np.abs(arr))

def safe_log1p_nonneg(series):
    """Aplica log1p em valores não-negativos; valores negativos viram log1p(0)=0."""
    arr = series.to_numpy(dtype=float)
    arr = np.where(arr > 0, arr, 0.0)
    return np.log1p(arr)

def aumento_features_bondora(df, percent_increase):
    df_new = df.copy()
    epsilon = 1e-6

    print(f"\n--- Processando Nível: {percent_increase}% ---")

    if percent_increase >= 10:
        print("Adicionando 11 features do Nível 10%...")
        df_new['feat_Amount_vs_Income'] = df_new['AppliedAmount'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Amount_vs_FreeCash'] = df_new['AppliedAmount'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Liabilities_vs_FreeCash'] = df_new['LiabilitiesTotal'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Liabilities_vs_Income'] = df_new['LiabilitiesTotal'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_EstimatedMonthlyPayment'] = df_new['AppliedAmount'] / (df_new['LoanDuration'] + epsilon)
        df_new['feat_Payment_vs_FreeCash_Ratio'] = df_new['feat_EstimatedMonthlyPayment'] / (df_new['FreeCash'] + epsilon)
        df_new['feat_Payment_vs_Income_Ratio'] = df_new['feat_EstimatedMonthlyPayment'] / (df_new['IncomeTotal'] + epsilon)
                                                                               
        for col in ['BidsPortfolioManager', 'BidsApi', 'BidsManual']:
            if col not in df_new.columns:
                df_new[col] = 0
        df_new['feat_TotalBids'] = df_new['BidsPortfolioManager'].fillna(0) + df_new['BidsApi'].fillna(0) + df_new['BidsManual'].fillna(0)
        df_new['feat_Api_Bids_Ratio'] = df_new['BidsApi'] / (df_new['feat_TotalBids'] + epsilon)
        df_new['feat_Duration_vs_Age'] = df_new['LoanDuration'] / (df_new['Age'] + epsilon)
        df_new['feat_EarlyRepayment_Rate'] = df_new['PreviousEarlyRepaymentsCountBeforeLoan'] / (df_new['NoOfPreviousLoansBeforeLoan'] + epsilon)

    if percent_increase >= 20:
        print("Adicionando 10 features do Nível 20%...")
                                                    
        income_cols = ['IncomeFromPension', 'IncomeFromFamilyAllowance', 'IncomeFromSocialWelfare',
                       'IncomeFromLeavePay', 'IncomeFromChildSupport', 'IncomeOther']
        for c in income_cols:
            if c not in df_new.columns:
                df_new[c] = 0

        df_new['feat_OtherIncome_Total'] = (df_new['IncomeFromPension'].fillna(0) + df_new['IncomeFromFamilyAllowance'].fillna(0) +
                                           df_new['IncomeFromSocialWelfare'].fillna(0) + df_new['IncomeFromLeavePay'].fillna(0) +
                                           df_new['IncomeFromChildSupport'].fillna(0) + df_new['IncomeOther'].fillna(0))
        df_new['feat_OtherIncome_Ratio'] = df_new['feat_OtherIncome_Total'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_PrincipalIncome_Ratio'] = df_new['IncomeFromPrincipalEmployer'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Pension_Ratio'] = df_new['IncomeFromPension'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_FamilyAllowance_Ratio'] = df_new['IncomeFromFamilyAllowance'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_SocialWelfare_Ratio'] = df_new['IncomeFromSocialWelfare'] / (df_new['IncomeTotal'] + epsilon)
        df_new['feat_Manual_Bids_Ratio'] = df_new['BidsManual'] / (df_new['feat_TotalBids'] + epsilon)
        df_new['feat_Portfolio_Bids_Ratio'] = df_new['BidsPortfolioManager'] / (df_new['feat_TotalBids'] + epsilon)
                                  
        for c in ['RefinanceLiabilities', 'LiabilitiesTotal', 'AppliedAmount']:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new['feat_Refinance_vs_Liabilities'] = df_new['RefinanceLiabilities'] / (df_new['LiabilitiesTotal'] + epsilon)
        df_new['feat_Refinance_vs_Amount'] = df_new['RefinanceLiabilities'] / (df_new['AppliedAmount'] + epsilon)

    if percent_increase >= 30:
        print("Adicionando 10 features do Nível 30% (log seguros)...")
                                                                          
        nonneg_cols = ['AppliedAmount', 'IncomeTotal', 'LiabilitiesTotal', 'FreeCash', 'Age',
                    'LoanDuration', 'Interest', 'ExistingLiabilities', 'NoOfPreviousLoansBeforeLoan', 'feat_TotalBids']
        for c in nonneg_cols:
            if c not in df_new.columns:
                df_new[c] = 0
                                                                
        df_new['feat_Log_AppliedAmount'] = safe_log1p_nonneg(df_new['AppliedAmount'])
        df_new['feat_Log_IncomeTotal'] = safe_log1p_nonneg(df_new['IncomeTotal'])
        df_new['feat_Log_LiabilitiesTotal'] = safe_log1p_nonneg(df_new['LiabilitiesTotal'])
                                                       
        df_new['feat_Log_FreeCash'] = signed_log(df_new['FreeCash'])
        df_new['feat_Log_Age'] = safe_log1p_nonneg(df_new['Age'])
        df_new['feat_Log_LoanDuration'] = safe_log1p_nonneg(df_new['LoanDuration'])
        df_new['feat_Log_Interest'] = signed_log(df_new['Interest'])                                          
        df_new['feat_Log_ExistingLiabilities'] = safe_log1p_nonneg(df_new['ExistingLiabilities'])
        df_new['feat_Log_NoOfPreviousLoans'] = safe_log1p_nonneg(df_new['NoOfPreviousLoansBeforeLoan'])
        df_new['feat_Log_TotalBids'] = safe_log1p_nonneg(df_new['feat_TotalBids'])

    if percent_increase >= 40:
        print("Adicionando 11 features do Nível 40%...")
        df_new['feat_Age_sq'] = df_new['Age']**2
        df_new['feat_Interest_sq'] = df_new['Interest']**2
        df_new['feat_LoanDuration_sq'] = df_new['LoanDuration']**2
        df_new['feat_Log_IncomeTotal_sq'] = df_new.get('feat_Log_IncomeTotal', 0)**2
        df_new['feat_Log_AppliedAmount_sq'] = df_new.get('feat_Log_AppliedAmount', 0)**2
        df_new['feat_Age_x_Interest'] = df_new['Age'] * df_new['Interest']
        df_new['feat_Age_x_Amount'] = df_new['Age'] * df_new['AppliedAmount']
        df_new['feat_Age_x_Income'] = df_new['Age'] * df_new['IncomeTotal']
        df_new['feat_Interest_x_Amount'] = df_new['Interest'] * df_new['AppliedAmount']
        df_new['feat_Interest_x_Duration'] = df_new['Interest'] * df_new['LoanDuration']
        df_new['feat_Interest_x_Income'] = df_new['Interest'] * df_new['IncomeTotal']

    if percent_increase >= 50:
        print("Adicionando 10 features do Nível 50%...")
        df_new['feat_Amount_x_Duration'] = df_new['AppliedAmount'] * df_new['LoanDuration']
        df_new['feat_DebtToIncome_x_Age'] = df_new['DebtToIncome'] * df_new['Age']
        df_new['feat_DebtToIncome_x_Interest'] = df_new['DebtToIncome'] * df_new['Interest']
        for c in ['ExistingLiabilities', 'AppliedAmount', 'IncomeTotal', 'FreeCash']:
            if c not in df_new.columns:
                df_new[c] = 0
        df_new['feat_Amount_vs_ExistingLiabilities'] = df_new['AppliedAmount'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_Income_vs_ExistingLiabilities'] = df_new['IncomeTotal'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_FreeCash_vs_ExistingLiabilities'] = df_new['FreeCash'] / (df_new['ExistingLiabilities'] + epsilon)
        df_new['feat_Income_minus_Liabilities'] = df_new['IncomeTotal'] - df_new['LiabilitiesTotal']
        df_new['feat_Liabilities_minus_Refinance'] = df_new['LiabilitiesTotal'] - df_new['RefinanceLiabilities'].fillna(0)
        df_new['feat_Age_cubed'] = df_new['Age']**3
        df_new['feat_Interest_cubed'] = df_new['Interest']**3

    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    nan_count = df_new.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaNs encontrados — preenchendo com 0.")
    df_new = df_new.fillna(0)

    print(f"Número final de features: {df_new.shape[1]}")
    return df_new

def aumentar_features_german(df, percent_increase):
    """
    Adiciona features cumulativas (10% a 50%) para o German Credit.
    Base atual: 14 colunas preditoras.
    Aumentos de ~1.4 (arredondado para 1 ou 2) features por nível.
    """
    df_new = df.copy()
    epsilon = 1e-6

    if percent_increase >= 10:
                                                                       
        df_new['feat_Credit_per_month'] = df_new['Credit amount'] / (df_new['Duration'] + epsilon)

    if percent_increase >= 20:
                                                                         
        df_new['feat_Age_young'] = (df_new['Age'] < 30).astype(int)

    if percent_increase >= 30:
                                                               
        df_new['feat_Job_Credit_Interaction'] = df_new['Job'] * df_new['Credit amount']
                                             
        df_new['feat_High_Amount'] = (df_new['Credit amount'] > df_new['Credit amount'].median()).astype(int)

    if percent_increase >= 40:

        df_new['feat_Credit_Age_Risk'] = (
            df_new['Credit amount'] /
            (df_new['Age'] + epsilon)
        )

        df_new['feat_Duration_Age_Ratio'] = (
            df_new['Duration'] /
            (df_new['Age'] + epsilon)
        )

    if percent_increase >= 50:

        risk_cols = [
            'Purpose_education',
            'Purpose_repairs',
            'Purpose_vacation/others'
        ]

        df_new['feat_High_Risk_Purpose'] = (
            df_new[risk_cols].sum(axis=1)
        )
                                             
        df_new = df_new.replace([np.inf, -np.inf], np.nan).fillna(0)

    return df_new

def aumentar_features_xyz(df, percent_increase):

    df_new = df.copy()
    epsilon = 1e-6

    print(f"\n--- Aumento de {percent_increase}% ---")

    if percent_increase >= 10:

        df_new['feat_loan_vs_income'] = df_new['loan_amnt'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_installment_vs_income'] = df_new['installment'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_revol_bal_vs_income'] = df_new['revol_bal'] / (df_new['annual_inc'] + epsilon)
        df_new['feat_totalacc_ratio'] = df_new['open_acc'] / (df_new['total_acc'] + epsilon)

    if percent_increase >= 20:

        df_new['feat_credit_pressure'] = df_new['dti'] * df_new['revol_util']
        df_new['feat_inq_intensity'] = df_new['inq_last_6mths'] + df_new['inq_last_12m']
        df_new['feat_delinq_risk'] = df_new['mths_since_last_delinq'] / (df_new['total_acc'] + epsilon)
        df_new['feat_pubrec_ratio'] = df_new['pub_rec'] / (df_new['total_acc'] + epsilon)

    if percent_increase >= 30:

        df_new['feat_log_loan'] = safe_log1p_nonneg(df_new['loan_amnt'])
        df_new['feat_log_income'] = safe_log1p_nonneg(df_new['annual_inc'])
        df_new['feat_log_revol_bal'] = safe_log1p_nonneg(df_new['revol_bal'])
        df_new['feat_log_tot_cur_bal'] = safe_log1p_nonneg(df_new['tot_cur_bal'])

    if percent_increase >= 40:

        df_new['feat_dti_x_loan'] = df_new['dti'] * df_new['loan_amnt']
        df_new['feat_int_x_term'] = df_new['int_rate'] * df_new['term']
        df_new['feat_income_x_grade'] = df_new['annual_inc'] * df_new['grade']
        df_new['feat_revol_x_util'] = df_new['revol_bal'] * df_new['revol_util']

    if percent_increase >= 50:

        df_new['feat_dti_sq'] = df_new['dti'] ** 2
        df_new['feat_int_sq'] = df_new['int_rate'] ** 2
        df_new['feat_loan_sq'] = df_new['loan_amnt'] ** 2
        df_new['feat_income_minus_debtproxy'] = df_new['annual_inc'] - df_new['loan_amnt']

    df_new = df_new.replace([np.inf, -np.inf], np.nan)
    df_new = df_new.fillna(0)

    print("Total de features:", df_new.shape[1])

    return df_new