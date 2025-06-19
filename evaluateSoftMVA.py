import sys, os, subprocess, uproot, joblib, argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

branches_MVA =[
    "Etamu", "dispMu", "xydispMu",
    "cQ_uS_", "cQ_tK_", "cQ_gK_", "cQ_tRChi2_",
    "cQ_sRChi2_", "cQ_Chi2LP_", "cQ_Chi2LM_", "cQ_lD_", "cQ_gDEP_", "cQ_tM_", "cQ_gTP_", 
    "match1_dX_", "match1_pullX_", "match1_pullDxDz_", "match1_dY_", "match1_pullY_", "match1_pullDyDz_", 
    "match2_dX_", "match2_pullX_", "match2_pullDxDz_", "match2_dY_", "match2_pullY_", "match2_pullDyDz_", 
    "validMuonHitComb", "nValidTrackerHits",
    "nValidPixelHits", "GL_nValidMuHits", "nStMu", "nMatchesMu", 
    "innerTrk_ValidFraction_", "innerTrk_highPurity_", 
    "innerTrk_normChi2_", "outerTrk_normChi2_", "outerTrk_muStValidHits_"   
]

def disp_cols(df):

    if 'x_bs' not in df.keys():
        df.rename({'BS_x': 'x_bs', 'BS_y': 'y_bs', 'BS_z': 'z_bs'})
    
    df['dispMu1'] = np.sqrt((df['Vx1'] - df['x_bs'])**2 + (df['Vy1'] - df['y_bs'])**2 + (df['Vz1'] - df['z_bs'])**2)
    df['xydispMu1'] = np.sqrt((df['Vx1'] - df['x_bs'])**2 + (df['Vy1'] - df['y_bs'])**2)
    df['dispMu2'] = np.sqrt((df['Vx2'] - df['x_bs'])**2 + (df['Vy2'] - df['y_bs'])**2 + (df['Vz2'] - df['z_bs'])**2)
    df['xydispMu2'] = np.sqrt((df['Vx2'] - df['x_bs'])**2 + (df['Vy2'] - df['y_bs'])**2)
    df['dispMu3'] = np.sqrt((df['Vx3'] - df['x_bs'])**2 + (df['Vy3'] - df['y_bs'])**2 + (df['Vz3'] - df['z_bs'])**2)
    df['xydispMu3'] = np.sqrt((df['Vx3'] - df['x_bs'])**2 + (df['Vy3'] - df['y_bs'])**2)



    return df

def load_data(file_names):
    """Load ROOT data and turn tree into a pd dataframe"""
    trees = []
    for file in file_names:
        print("Loading data from", file)
        f = uproot.open(file)
        tree = f["FinalTree"]
        trees.append(tree.arrays(library="pd"))
    data = pd.concat(trees)
    data = disp_cols(data)
    return data

def save_data(data, fileName):
    data.to_csv(fileName+".csv", index=False)
    print("File CSV saved!")
    del data
    rdf = RDF.FromCSV(fileName+".csv")
    rdf.Snapshot("FinalTree", fileName+".root")
    print("File ROOT saved!")

def predict(data, index, model):
    print("Start prediction label: ", index)
    branches = [var + str(index) for var in branches_MVA]
    X = data[branches]
    X = X.values
    #predictionsID = model.predict(X)
    predictions = model.predict_proba(X)
    #data["privateMVAID_mu"+str(index)] = predictionsID
    data["privateMVA_mu"+str(index)] = predictions[:,1]
    del predictions
    del predictionsID
    del X
    #print("Done!")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set tau3mu or control")
    parser.add_argument("--type", type=str, help="tau3mu or control")
    args = parser.parse_args()
    type = args.type
    if type=="tau3mu":
        type = "Data"
        maxN = 4
    elif type=="control":
        type = "Control"
        maxN = 3
    
    filePath = '/depot/cms/users/schul105/Tau3Mu/analysis/CMSSW_15_0_6_patch1/src/Analysis/rootFiles/'

    files = [filePath+'AnalysedTree_MC_Ds_tau3mu_2023.root']
    model = joblib.load('MVA_KPiGlobalNoPt.pkl')
    data = load_data(files)
    for i in range(1,maxN):
        data = predict(data, i, model)
    save_data(data, "ROOTFiles/All"+type+"_plusMVA")

