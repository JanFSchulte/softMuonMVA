import uproot
import matplotlib.pyplot as plt
import numpy as np
import mplhep as hep
import pandas as pd
import awkward as ak
from tqdm import tqdm
from scipy.stats import ks_2samp
import multiprocessing
from time import time
import pickle
import awkward as ak

exclude_keys = {'evt', 'run', 'lumi', 'nPileUpInt', 'GenParticle_PdgId', 'GenParticle_Pt', 'GenParticle_Eta', 'GenParticle_Phi', 'GenParticle_isDs', 'GenParticle_isB', 'GenParticle_isBdecay', 'GenParticle_MotherPdgId', 'MuonCollectionSize'}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process dilepton sample and convert to pandas dataframe')
    parser.add_argument('--file', type=str, help='Root file to process')
    parser.add_argument('--outfile', type=str, help='Path to processed file, including .pkl extension')
    parser.add_argument('--n_jobs', type=int, help='Number of jobs')
    
    args = parser.parse_args()
    
    file = args.file
    n_jobs = int(args.n_jobs)
    outfile = args.outfile
    
    file = uproot.open(file)
    
    for key in file.keys():
        if 'ntuple' in key:
            d, _ = key.split('/')
            break
    
    ntuple = file[d]['ntuple']
    
    ntuple_keys = ntuple.keys()
    '''
    for key in ntuple_keys:
        if 'L1' in key or 'HLT' in key or 'cal' in key or 'Triplet' in key or 'Ref' in key or 'IP' in key or 'Vtx' in key:
            ntuple_keys.remove(key)

    for key in exclude_keys:
        try:
            ntuple_keys.remove(key)
        except:
            pass
    '''
    
    ntuple_keys = ['Trigger_hltname','Trigger_hltdecision', 'Muon_isGlobal',
        'MuonEta', 'MuonPt', 'Muon_GLhitPattern_numberOfValidMuonHits',
       'Muon_GLnormChi2', 'Muon_Numberofvalidpixelhits',
       'Muon_Numberofvalidtrackerhits',
       'Muon_combinedQuality_chi2LocalMomentum',
       'Muon_combinedQuality_chi2LocalPosition',
       'Muon_combinedQuality_glbKink',
       'Muon_combinedQuality_glbTrackProbability',
       'Muon_combinedQuality_globalDeltaEtaPhi',
       'Muon_combinedQuality_localDistance',
       'Muon_combinedQuality_match1_dX', 'Muon_combinedQuality_match1_dY',
       'Muon_combinedQuality_match1_pullDxDz',
       'Muon_combinedQuality_match1_pullDyDz',
       'Muon_combinedQuality_match1_pullX',
       'Muon_combinedQuality_match1_pullY',
       'Muon_combinedQuality_match2_dX', 'Muon_combinedQuality_match2_dY',
       'Muon_combinedQuality_match2_pullDxDz',
       'Muon_combinedQuality_match2_pullDyDz',
       'Muon_combinedQuality_match2_pullX',
       'Muon_combinedQuality_match2_pullY',
       'Muon_combinedQuality_staRelChi2',
       'Muon_combinedQuality_tightMatch', 'Muon_combinedQuality_trkKink',
       'Muon_combinedQuality_trkRelChi2',
       'Muon_combinedQuality_updatedSta', 'Muon_innerTrack_ValidFraction',
       'Muon_innerTrack_highPurity', 'Muon_innerTrack_normalizedChi2',
       'Muon_numberOfMatchedStations', 'Muon_numberOfMatches',
       'Muon_outerTrack_muonStationsWithValidHits',
       'Muon_outerTrack_normalizedChi2', 'Muon_segmentCompatibility',
       'Muon_trackerLayersWithMeasurement', 'Muon_validMuonHitComb',
       'Muon_vx', 'Muon_vy', 'Muon_vz', 'Muon_SoftMVA_Val']

    try:
        ntuple['x_bs']
        ntuple_keys += ['x_bs', 'y_bs', 'z_bs']
    except:
        ntuple_keys += ['BS_x', 'BS_y', 'BS_z']
    
    
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
        "Muon_innerTrack_nValidHits",# muon.innerTrack()->hitPattern().numberOfValidTrackerHits()
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
    
    ntuple_keys = list( set(ntuple_keys)|set(xgb_keys) )
    
    if 'Muon_simMotherPdgId' in ntuple.keys():
        ntuple_keys += ['Muon_simMotherPdgId']
    else:
        ntuple_keys += ['Muon_MotherPdgId']
        
    df = ntuple2pandas(ntuple, ntuple_keys, n_jobs=n_jobs)
    
    df.to_pickle(outfile)
    
    return


def get_soft_idx(evt):
    return np.argmin(evt['MuonPt'])


def select_tau3mu(evt):
    if np.sum(np.abs(evt['Muon_simMotherPdgId'])==15)>=1:
        return True
    else:
        return False

def select_dilep(evt):
    return evt['MuonPt'][get_soft_idx(evt)] < 10

def collect(idx, chunk):
    
    arr = []
    for evt in chunk:
        key = list(evt.to_list().keys())[0]
        if 'hlt' not in key and 'bs' not in key.lower():
            arr.extend(*list(evt.to_list().values()))
        else:
            arr.append(*list(evt.to_list().values()))

    arr = np.array(arr)
    return idx, arr

def callback_fcn(x):
    for idx,chunk in x:
        idxs.append(idx)
        arr.append(chunk)
    

def run_var(branch, key, n_jobs=32):

    global arr
    global idxs

    
    arr = []
    idxs = []
    args_agg = []
    stop = False
    array = branch.arrays()

    n_hits_array = []
    
    chunk_size = int(len(array)/n_jobs)
    
    while chunk_size < 1:
        n_jobs -= 1
        chunk_size = int(len(array)/n_jobs)

    p = multiprocessing.Pool(n_jobs)
    
    chunks = [(i,array[i*chunk_size:(i+1)*chunk_size]) for i in range(n_jobs)]

    p.starmap_async(collect, chunks, callback=callback_fcn, error_callback=lambda x: print(x))
    p.close()
    p.join()  

    
    final_arr = []

    for i in idxs:
        final_arr.append(arr[i])

    final_arr = np.concatenate(final_arr)

    return (key, final_arr)
    
def ntuple2pandas(ntuple, keys, signal=False, n_jobs=32):

    df = {key: [] for key in keys}
    arrs_agg = []
    
    args_agg = []
    
    #run_var(ntuple['MuonPt'], 'MuonPt', num_muons)
    
    print(f'Max Jobs: {n_jobs}')
    pbar = tqdm(keys)
    for key in pbar:
        
        pbar.set_description(f'Running {key}')
        if 'hlt' in key:
            hlt_col = hlt_assignment(ntuple[key], ntuple['MuonPt'])    
            df[key] = hlt_col
            continue

        if 'bs' in key.lower():
            bs_col = bs_assignment(ntuple[key], ntuple['MuonPt'])    
            df[key] = bs_col
            continue
        
        arrs_agg.append(run_var(ntuple[key], key, n_jobs=n_jobs))

    for pair in arrs_agg:
        k,v = pair
        df[k] = v
    
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in df.items()]))

    return df

def hlt_assignment(hlt_branch, ref_branch):
    hlt_col = []
    hlt_branch = hlt_branch.arrays().to_list()

    for i, evt in enumerate(ref_branch.arrays().to_list()):
        num_muons = len(list(evt.values())[0])
        for j in range(num_muons):
            hlt_col.append(list(hlt_branch[i].values())[0])

    return hlt_col

def bs_assignment(bs_branch, ref_branch):
    bs_col = []
    bs_branch = bs_branch.arrays().to_list()

    for i, evt in enumerate(ref_branch.arrays().to_list()):

        num_muons = len(list(evt.values())[0])
        for j in range(num_muons):
            bs_col.append(list(bs_branch[i].values())[0])

    return bs_col

if __name__ == '__main__':
    main()
