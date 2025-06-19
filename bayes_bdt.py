import numpy as np
import pandas as pd
import pickle
import mplhep as hep
import matplotlib.pyplot as plt
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
#from xgboost import XGBClassifier
import torch
from tqdm import tqdm
import xgboost as xgb
import optuna
plt.style.use(hep.style.CMS)
xgb_model = xgb.Booster()
xgb_model.load_model("Run2022-20231030-1731-Event0.model")

keys = [
        'MuonEta',
        'Muon_disp', 'Muon_xydisp',
        'Muon_combinedQuality_updatedSta', 'Muon_combinedQuality_trkKink',
           'Muon_combinedQuality_glbKink', 'Muon_combinedQuality_trkRelChi2',
           'Muon_combinedQuality_staRelChi2',
           'Muon_combinedQuality_chi2LocalPosition',
           'Muon_combinedQuality_chi2LocalMomentum',
           'Muon_combinedQuality_localDistance',
           'Muon_combinedQuality_globalDeltaEtaPhi',
           'Muon_combinedQuality_tightMatch',
           'Muon_combinedQuality_glbTrackProbability',
           'Muon_combinedQuality_match1_dX', 'Muon_combinedQuality_match1_pullX',
           'Muon_combinedQuality_match1_pullDxDz',
           'Muon_combinedQuality_match1_dY', 'Muon_combinedQuality_match1_pullY',
           'Muon_combinedQuality_match1_pullDyDz',
           'Muon_combinedQuality_match2_dX', 'Muon_combinedQuality_match2_pullX',
           'Muon_combinedQuality_match2_pullDxDz',
           'Muon_combinedQuality_match2_dY', 'Muon_combinedQuality_match2_pullY',
           'Muon_combinedQuality_match2_pullDyDz', 'Muon_validMuonHitComb',
           'Muon_Numberofvalidtrackerhits', 'Muon_Numberofvalidpixelhits','Muon_GLhitPattern_numberOfValidMuonHits','Muon_numberOfMatchedStations','Muon_numberOfMatches',
           'Muon_innerTrack_ValidFraction', 'Muon_innerTrack_highPurity','Muon_innerTrack_normalizedChi2',
           'Muon_outerTrack_normalizedChi2', 'Muon_outerTrack_muonStationsWithValidHits'
    ]


def apply_hlt(df):
    idxs = []
    for i in tqdm(range(len(df))):
        event = df.iloc[i]
        if np.isnan(event['MuonPt']): continue
            
        names = event['Trigger_hltname']
        decisions = event['Trigger_hltdecision']
        for name, decision in zip(names, decisions):
            if 'dstau3mu' in name.lower() and decision==1:
                idxs.append(i)

    idxs = np.array(idxs)
    df = df.iloc[idxs]

    return df

def disp_cols(df):

    if 'x_bs' not in df.keys():
        df.rename({'BS_x': 'x_bs', 'BS_y': 'y_bs', 'BS_z': 'z_bs'})
    
    df['Muon_disp'] = np.sqrt((df['Muon_vx'] - df['x_bs'])**2 + (df['Muon_vy'] - df['y_bs'])**2 + (df['Muon_vz'] - df['z_bs'])**2)
    df['Muon_xydisp'] = np.sqrt((df['Muon_vx'] - df['x_bs'])**2 + (df['Muon_vy'] - df['y_bs'])**2)

    return df
    
def get_sf(pt, eta, sf, pt_bins, eta_bins):
    
    if pt > 25 or np.abs(eta) > 2.5:
        return 20.
    
    for i in range(len(pt_bins[:-1])):
        for j in range(len(eta_bins[:-1])):
            
            pt_lower = pt_bins[i]
            pt_upper = pt_bins[i+1]
            eta_lower = eta_bins[j]
            eta_upper = eta_bins[j+1]
            
            if pt > pt_lower and pt <= pt_upper:
                if eta > eta_lower and eta <= eta_upper:
                    return sf[i][j]
                    
def apply_sf(df, sf_table, bkg=True):
    sf_table, binx, biny = sf_table
    
    idxs = []

    for i in tqdm(range(len(df))):
        event = df.iloc[i]
    
        pt = event['MuonPt']
        eta = event['MuonEta']
        sf = get_sf(pt, eta, sf_table, binx, biny)
        x  = np.random.rand()
        if bkg:
            if not(sf > 1 and x > 1/sf):
                idxs.append(i)
        else:
            if not(sf < 1 and x > sf):
                idxs.append(i)

    
    idxs = np.array(idxs)
    return df.copy().iloc[idxs]

def apply_isGlobal(df, NotGlobal=False):

    if NotGlobal:
        return df[df['Muon_isGlobal']==0]
    else:
        return df[df['Muon_isGlobal']==1]

## BAYESIAN OPTIMIZATION

def objective(trial, train_data, train_labels, valid_data, valid_labels):
    ## optimize learning_rate, l2_regularization

    lr = trial.suggest_float('learning_rate', 0.0, 1)
    l2_regularization = trial.suggest_float('l2_regularization', 0,1)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2,512)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 2,512)
    max_depth = trial.suggest_int('max_depth', 2,512)
    
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=max_depth, verbose=0, random_state=717, early_stopping=True,tol=1e-7,n_iter_no_change=10, validation_fraction=None, learning_rate=lr, l2_regularization=l2_regularization, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf)

    clf.fit(train_data, train_labels)

    staged_probs = clf.staged_predict_proba(valid_data)
    
    aucs = []
    for i, probs in enumerate(staged_probs):
        auc = roc_auc_score(valid_labels, probs[:,1])
        aucs.append(auc)


    return np.max(aucs) #, bd_auc

def process(isGlobal=False, KPi=False, NotGlobal=False, plotting=False):
    xgb_keys = [
            'Muon_innerTrack_ValidFraction',
            'Muon_combinedQuality_glbTrackProbability',
            "Muon_innerTrack_nLostHitsInner", #muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS)
            "Muon_innerTrack_nLostHitsOuter", #muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS)
            "Muon_combinedQuality_trkKink",
            "Muon_combinedQuality_chi2LocalPosition",
            "Muon_combinedQuality_match2_dX",
            "Muon_combinedQuality_match2_pullX",
            "Muon_combinedQuality_match1_dX",
            "Muon_combinedQuality_match1_pullX",
            "Muon_innerTrack_nPixels",  #muon.innerTrack()->hitPattern().numberOfValidPixelHits())
            "Muon_innerTrack_nValidHits", #muon.innerTrack()->hitPattern().numberOfValidTrackerHits()
            "Muon_innerTrack_nLostHitsOn", #muon.innerTrack()->hitPattern().numberOfLostTrackerHits(reco::HitPattern::TRACK_HITS)
            "Muon_combinedQuality_match2_dY",
            "MuonEta",
            "Muon_combinedQuality_match1_dY",
            "Muon_combinedQuality_match2_pullY",
            "Muon_combinedQuality_match1_pullY",
            "Muon_combinedQuality_match2_pullDyDz",
            "Muon_combinedQuality_match1_pullDyDz",
            "Muon_combinedQuality_match2_pullDxDz",
            "Muon_combinedQuality_match1_pullDxDz",
            "MuonPt",
        ]
       
    with open('InclusiveDilepton_Run3Compare_WithCuts_DF.pkl', 'rb') as handle:
        bkg_df = pickle.load(handle)

    with open('MCDsTau3Mu_DF.pkl', 'rb') as handle:
        ds_df = pickle.load(handle)
    with open('MCBdTau3Mu_DF.pkl', 'rb') as handle:
        bd_df = pickle.load(handle)
    
    sig_df = pd.concat((ds_df, bd_df))
    if KPi:
        bkg_df = bkg_df[np.where(np.isin(np.abs(bkg_df['Muon_MotherPdgId']), [321, 221]), True, False)]
    else:
        bkg_df = bkg_df[np.where(np.isin(np.abs(bkg_df['Muon_MotherPdgId']), [511,521,531,321,411,421,431,443, 221]), True, False)]

    if isGlobal or NotGlobal:
        bkg_df = apply_isGlobal(bkg_df, NotGlobal=NotGlobal)
        sig_df = apply_isGlobal(sig_df, NotGlobal=NotGlobal)
    
    bkg_df = apply_hlt(bkg_df)
    bkg_df = disp_cols(bkg_df)

    sig_df = apply_hlt(sig_df)
    sig_df = disp_cols(sig_df)

    if plotting:
        return sig_df, bkg_df
        
    h_sig, binx, biny, _ = plt.hist2d(sig_df['MuonPt'], sig_df['MuonEta'], range=[[2,25], [-2.4, 2.4]], bins=[60,60], density=True)
    h_bkg, _, _, _ = plt.hist2d(bkg_df['MuonPt'], bkg_df['MuonEta'], range=[[2,25], [-2.4, 2.4]], bins=[binx,biny], density=True)
    #sf_table = h_sig/h_bkg
    sf_table = (h_bkg/h_sig, binx, biny)
    plt.clf()

    sig = apply_sf(sig_df, sf_table, bkg=False).sample(frac=1, random_state=717)
    bkg = apply_sf(bkg_df, sf_table, bkg=True)

    xgb_sig = sig.copy()
    xgb_bkg = bkg_df.copy()

    xgb_sig = xgb_sig[xgb_keys]
    xgb_bkg = xgb_bkg[xgb_keys]

    sig_mother_ids = np.abs(np.array(sig['Muon_simMotherPdgId']))
    bkg_mother_ids = np.abs(np.array(bkg['Muon_MotherPdgId']))

    sig = sig[keys]
    bkg = bkg[keys]
    

    print(len(sig))
    print(len(xgb_sig))
    print(len(bkg))
    print(len(xgb_bkg))
    frac_train = .9
    num_sig_train = int(len(sig)*frac_train)
    num_bkg_train = int(len(bkg)*frac_train)

    train_bkg = resample(bkg.iloc[:num_bkg_train][keys].to_numpy(), n_samples=num_sig_train)
    sig_trainkeys = sig[keys]
    bkg_trainkeys = bkg[keys]

    train_data = np.concatenate( (sig_trainkeys.iloc[:num_sig_train].to_numpy(), train_bkg) )
    train_labels = np.concatenate( (np.ones(num_sig_train), np.zeros(num_sig_train)) )
    
    valid_data = np.concatenate( (sig_trainkeys.iloc[num_sig_train:].to_numpy(), bkg_trainkeys.iloc[num_bkg_train:].to_numpy()) )
    valid_labels = np.concatenate( (np.ones(len(sig)-num_sig_train), np.zeros(len(bkg)-num_bkg_train)) )

    #train_mvas = np.abs(np.concatenate( (sig_mvas[:num_sig_train],bkg_mvas[:num_bkg_train]) ))
    #valid_mvas = np.abs(np.concatenate( (sig_mvas[num_sig_train:],bkg_mvas[num_bkg_train:]) ))
    
    valid_mother_ids = np.abs(np.concatenate( (sig_mother_ids[num_sig_train:],bkg_mother_ids[num_bkg_train:]) ))

    
    xgb_train_data = np.concatenate( (xgb_sig.iloc[:num_sig_train].to_numpy(), xgb_bkg.iloc[:num_bkg_train].to_numpy()) )
    xgb_valid_data = np.concatenate( (xgb_sig.iloc[num_sig_train:].to_numpy(), xgb_bkg.iloc[num_bkg_train:].to_numpy()) )

    
    np.save('train_data.npy', train_data)
    np.save('xgb_train_data.npy', xgb_train_data)
    np.save('train_labels.npy', train_labels)
    
    np.save('valid_data.npy', valid_data)
    np.save('xgb_valid_data.npy', xgb_valid_data)
    np.save('valid_labels.npy', valid_labels)
    np.save('valid_mother_ids.npy', valid_mother_ids)

    return sig, bkg

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization of Tau3MuMVA')
    parser.add_argument('--isGlobal', type=bool, help='If specified, only trains on global muons')
    parser.add_argument('--NotGlobal', type=bool, help='If specified, only trains on NOT global muons')
    parser.add_argument('--KPi', type=bool, help='If specified, only trains on K/Pi background')
    parser.add_argument('--n_trials', type=int, help='Number of trials')
    parser.add_argument('--reprocess', type=bool, help='If specified, forces reprocessing of dataframes')
    parser.add_argument('--name', type=str, help='If specified, saves optimized MVA as <name>.pkl, else saves as test.pkl', default='test')
    
    args = parser.parse_args()
    
    isGlobal = bool(args.isGlobal)
    NotGlobal = bool(args.NotGlobal)
    
    if isGlobal or NotGlobal:
        assert isGlobal != NotGlobal, 'isGlobal and NotGlobal tags are mutually exclusive!'
    
    KPi = bool(args.KPi)
    n_trials = int(args.n_trials)
    reprocess = bool(args.reprocess)
    name = str(args.name)
    
    
    if reprocess == False:
        try:
            train_data = np.load('train_data.npy')
            train_labels = np.load('train_labels.npy')
            xbg_train_data = np.load('xgb_train_data.npy')
            
            
            valid_data = np.load('valid_data.npy')
            valid_labels = np.load('valid_labels.npy')
            valid_mother_ids= np.load('valid_mother_ids.npy')
            xgb_valid_data = np.load('xgb_valid_data.npy')
        
        except:
            print('Numpy arrays not found. Processing with the following options:')
            print(f'\t isGlobal={isGlobal}')
            print(f'\t NotGlobal={NotGlobal}')
            print(f'\t KPi={KPi}')
            process(isGlobal=isGlobal, KPi=KPi, NotGlobal=NotGlobal)
    
            train_data = np.load('train_data.npy')
            train_labels = np.load('train_labels.npy')
            xbg_train_data = np.load('xgb_train_data.npy')
            
            
            valid_data = np.load('valid_data.npy')
            valid_labels = np.load('valid_labels.npy')
            valid_mother_ids= np.load('valid_mother_ids.npy')
            xgb_valid_data = np.load('xgb_valid_data.npy')
    
            print('Done')
    
    else:
        print('Reprocessing with the following options:')
        print(f'\t isGlobal={isGlobal}')
        print(f'\t KPi={KPi}')
        print(f'\t NotGlobal={NotGlobal}')
        process(isGlobal=isGlobal, KPi=KPi, NotGlobal=NotGlobal)
    
        train_data = np.load('train_data.npy')
        train_labels = np.load('train_labels.npy')
        train_mvas = np.load('train_mvas.npy')
        train_mother_ids= np.load('train_mother_ids.npy')
        xbg_train_data = np.load('xgb_train_data.npy')
        
        
        valid_data = np.load('valid_data.npy')
        valid_labels = np.load('valid_labels.npy')
        valid_mvas = np.load('valid_mvas.npy')
        valid_mother_ids= np.load('valid_mother_ids.npy')
        xgb_valid_data = np.load('xgb_valid_data.npy')
    
    
    study = optuna.create_study(directions=['maximize'])
    study.optimize(lambda trail: objective(trail, train_data,train_labels, valid_data, valid_labels), n_trials=n_trials)
    '''
    print('Best Trials')
    for trial in study.best_trials:
        print(f"Trial number: {trial.number}")
        print(f"Values: {trial.values}")
        print(f"Params: {trial.params}")
        print("")
        
    if len(study.best_trials) == 1:
        best = study.best_trials[0]
    else:
        values = []
        for trial in study.best_trials:
            values.append(trial.values[1])
    
        idx = np.argmax(values)
        best = study.best_trials[idx]
    '''
    
    params = study.best_params
    
    clf = HistGradientBoostingClassifier(max_iter=500, max_depth=100, verbose=0, random_state=717, early_stopping=True,tol=1e-7,n_iter_no_change=10, validation_fraction=None)
    clf.fit(train_data, train_labels)
    
    staged_probs = clf.staged_predict_proba(valid_data)
    
    aucs = []
    for i, probs in enumerate(staged_probs):
        auc = roc_auc_score(valid_labels, probs[:,1])
        aucs.append(auc)
    
    plt.plot(range(len(aucs)), aucs)
    plt.show()
    
    max_iter = np.argmax(aucs)+1
    print('Best Iteration: ', max_iter)
    print('Best AUC: ', np.max(aucs))
    
    clf = HistGradientBoostingClassifier(max_iter=max_iter, max_depth=100, verbose=0, random_state=717, validation_fraction=None)
    clf.fit(train_data, train_labels)
    
    with open(f'{name}.pkl', 'wb') as handle:
        pickle.dump(clf, handle)
    
    print('Saved')
    
    valid_clf_probs = clf.predict_proba(valid_data)[:, 1]
    mother_cats_dict = {'Tau/Photon': [15,22], 'B':[511,521,531,541], 'K+-/Pi+-':[321,221], 'D': [411,421,431], 'J/Psi':[443], 'Rho/Omega':[113,223]}
    
    sig_probs = valid_clf_probs[valid_labels==1]
    bkg_probs = valid_clf_probs[valid_labels==0]
    bkg_ids = valid_mother_ids[valid_labels==0]
    
    ## Overall
    auc = roc_auc_score(valid_labels, valid_clf_probs)
    fpr, tpr, _ = roc_curve(valid_labels, valid_clf_probs)
    
    if KPi: label=f'KPi AUC: {auc:.4f}'
    else: label=f'Overall AUC: {auc:.4f}'
    
    plt.plot(fpr, tpr, label=label, linewidth=5)
    plt.plot([0,1],[0,1], linestyle='dashed', c='black',alpha=.9)
    
    xgb_valid_probs = xgb_model.predict(xgb.DMatrix(xgb_valid_data))
    auc = roc_auc_score(valid_labels, xgb_valid_probs)
    fpr, tpr, _ = roc_curve(valid_labels, xgb_valid_probs)
    plt.plot(fpr, tpr, label=f'Run3MVA: {auc:.4f}', linewidth=5)
    plt.plot([0,1],[0,1], linestyle='dashed', c='black',alpha=.9)
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Validation')
    plt.legend(title='Bkg Source')
    plt.grid()
    plt.savefig(f'{name}_OverallROC.png')
    
    if not KPi:
        ## Per Bkg Source
        for key, ids in mother_cats_dict.items():
        
            try:
                probs = np.concatenate( (sig_probs, bkg_probs[ np.isin(np.abs(bkg_ids),ids) ]) )
                labels = np.concatenate( (np.ones(len(sig_probs)), np.zeros(np.sum(np.isin(np.abs(bkg_ids),ids)))) )
                auc = roc_auc_score(labels, probs)
                fpr, tpr, _ = roc_curve(labels, probs)
                plt.plot(fpr, tpr, label=f'{key} AUC: {auc:.4f}')
            except:
                pass
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Validation')
    plt.legend(title='Bkg Source')
    plt.grid()
    plt.savefig(f'{name}_PerBkgROC.png')
    
    
    
    print('Calculating Permutation Importance:')
    from sklearn.inspection import permutation_importance
    
    perm_importance = permutation_importance(clf, valid_data, valid_labels)
    feature_importances = perm_importance.importances_mean
    indices = np.argsort(feature_importances)[::-1]
    for i, idx in enumerate(indices):
        print(f'{i+1}: {keys[idx]} {feature_importances[idx]:.5e}')

if __name__ == "__main__":
    main()

