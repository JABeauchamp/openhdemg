"""
This module contains all the functions used to quantify and analyze MU persistent inward currents.
Currently includes delta F. 
"""

import pandas as pd
import numpy as np
from openhdemg.library.analysis import compute_SVR
from itertools import combinations

def compute_deltaF(
        emgfile,
        average_method_ = "test_unit_average",
        normalization_ = "False",
        recruitment_difference_cutoff_ = 1.0,
        corr_cutoff_ = 0.7,
        controlunitmodulation_cutoff_= 0.5,
        clean_ = "True"
        ):

    """
    Conducts a paired motor unit analysis, quantifying delta F bettwen the supplied collection of motor units. 
    Origional framework for deltaF provided in Gorassini et. al., 2002 : https://journals.physiology.org/doi/full/10.1152/jn.00024.2001
    James (Drew) Beauchamp 

    Parameters
    ----------
    emgfile : dict
        The dictionary containing the emgfile.
    average_method_ : str,  default "test_unit_average"
        The method for test MU deltaF value. More to be added TODO

        "test_unit_average"
            The average across all possible control units

    normalization_ : str,  {"False", "ctrl_max_desc"}, default "False"
        
        "ctrl_max_desc"
            Whether to normalize deltaF values to control unit descending range during test unit firing
            See Skarabot et. al., 2023 : https://www.biorxiv.org/content/10.1101/2023.10.16.562612v1

    recruitment_difference_cutoff_ : float, default 1
        An exlusion criteria corresponding to the necessary difference between control and test MU recruitement in seconds 
    corr_cutoff_ : float, (0,1), default 0.7
        An exclusion criteria corresponding to the correlation between control and test unit discharge rate. 
    controlunitmodulation_cutoff_ : float, default 0.5
        An exclusion criteria corresponding to the necessary modulation of control unit discharge rate during test unit firing in Hz.  
    clean : str, default "True"
        To remove values that do not meet exclusion criteria

        
    Returns
    -------
    dF : pd.DataFrame
        A pd.DataFrame containing deltaF values and corresponding MU number

    Warnings
    --------
    TODO

    Example
    -----
    import openhdemg.library as emg
    # Load the sample file
    emgfile = emg.emg_from_samplefile()

    # Sort MUs based on recruitment order
    emgfile = emg.sort_mus(emgfile=emgfile)

    # quantify delta F
    dF= compute_deltaF(emgfile=emgfile)

    dF

             dF   MU
    0       NaN  0.0
    1       NaN  1.0
    2       NaN  2.0
    3  1.838382  3.0
    4  2.709522  4.0


    """
    svrfits = compute_SVR(emgfile) # use smooth svr fits

    dFret_ret = []
    MUcombo_ret =[]
    if emgfile["NUMBER_OF_MUS"]<2: # if less than 2 MUs, can not quantify deltaF
        dFret_ret = np.nan
        MUcombo_ret = np.nan*np.ones(1,2)
    else:
        combs = combinations(range(emgfile["NUMBER_OF_MUS"]), 2) # combindations of MUs in file
        # TODO if units are nonconsecutive

        # init
        R = []
        dFret = []
        testMU = []
        ctrl_mod = []
        MUcombo_ =  []
        rcrt_diff = []
        controlMU = [] 
        for mucomb in list(combs): # for all possible combinations of MUs
            
            MU1_id, MU2_id = mucomb[0], mucomb[1] # assign MUs
            MUcombo_.append((MU1_id,MU2_id)) # track current MU combination

            # first MU firings, recruitment, and decrecruitment 
            MU1_times = np.where(emgfile["BINARY_MUS_FIRING"][MU1_id]==1)[0]
            MU1_rcrt,MU1_drcrt = MU1_times[1],MU1_times[-1] #skip first since idr is defined on second

            # second MU firings, recruitment, and decrecruitment 
            MU2_times = np.where(emgfile["BINARY_MUS_FIRING"][MU2_id]==1)[0]
            MU2_rcrt,MU2_drcrt = MU2_times[1],MU2_times[-1] #skip first since idr is defined on second

            # region of MU overlap
            MUoverlap = range(max(MU1_rcrt, MU2_rcrt), min(MU1_drcrt, MU2_drcrt))

            # if MUs do not overlapt by more than two or more samples 
            if len(MUoverlap)<2:
                dFret = np.append(dFret,np.nan)
                R = np.append(R,np.nan)
                rcrt_diff = np.append(rcrt_diff,np.nan)
                ctrl_mod = np.append(ctrl_mod,np.nan)
                continue #TODO test

            # corr between units - not always necessary, can be set to 0 when desired 
            r = pd.DataFrame(zip(svrfits.genSVR[MU1_id][MUoverlap],svrfits.genSVR[MU2_id][MUoverlap])).corr()
            R = np.append(R,r[0][1])

            # recruitment diff, necessary to ensure PICs are activated in control unit
            rcrt_diff = np.append(rcrt_diff,np.abs(MU1_rcrt-MU2_rcrt)/emgfile["FSAMP"])
            if MU1_rcrt < MU2_rcrt:
                controlU = 1 # MU 1 is control unit, 2 is test unit

                if MU1_drcrt < MU2_drcrt: # if control (reporter) unit is not on for entirety of test unit, set last firing to control unit
                    MU2_drcrt = MU1_drcrt
                    # this may understimate PICs, other methods can be employed 
                
                #delta F: change in control MU discharge rate between test unit recruitment and derecruitment
                df = svrfits.genSVR[MU1_id][MU2_rcrt]-svrfits.genSVR[MU1_id][MU2_drcrt]  

                # control unit discharge rate modulation while test unit is firing
                ctrl_mod = np.append(ctrl_mod,np.nanmax(svrfits.genSVR[MU1_id][range(MU2_rcrt,MU2_drcrt)])-np.nanmin(svrfits.genSVR[MU1_id][range(MU2_rcrt,MU2_drcrt)]))

                if normalization_ == "False":
                     dFret = np.append(dFret,df)
                elif normalization_ == "ctrl_max_desc": # normalize deltaF values to control unit descending range during test unit firing
                     k = svrfits.genSVR[MU1_id][MU2_rcrt]-svrfits.genSVR[MU1_id][MU1_drcrt]
                     dFret = np.append(dFret,df/k)

            elif MU1_rcrt > MU2_rcrt: 
                controlU = 2 # MU 2 is control unit, 1 is test unit
                
                if MU1_drcrt > MU2_drcrt: # if control (reporter) unit is not on for entirety of test unit, set last firing to control unit
                    MU1_drcrt = MU2_drcrt
                    # this may understimate PICs, other methods can be employed 
                
                #delta F: change in control MU discharge rate between test unit recruitment and derecruitment
                df = svrfits.genSVR[MU2_id][MU1_rcrt]-svrfits.genSVR[MU2_id][MU1_drcrt]

                # control unit discharge rate modulation while test unit is firing
                ctrl_mod = np.append(ctrl_mod,np.nanmax(svrfits.genSVR[MU2_id][range(MU1_rcrt,MU1_drcrt)])-np.nanmin(svrfits.genSVR[MU2_id][range(MU1_rcrt,MU1_drcrt)]))

                if normalization_ == "False":
                     dFret = np.append(dFret,df)
                elif normalization_ == "ctrl_max_desc": # normalize deltaF values to control unit descending range during test unit firing
                     k = svrfits.genSVR[MU2_id][MU1_rcrt]-svrfits.genSVR[MU2_id][MU2_drcrt]
                     dFret = np.append(dFret,df/k)

            elif MU1_rcrt == MU2_rcrt:
                if MU1_drcrt > MU2_drcrt:
                    controlU = 1 # MU 1 is control unit, 2 is test unit
               
                    #delta F: change in control MU discharge rate between test unit recruitment and derecruitment
                    df = svrfits.genSVR[MU1_id][MU2_rcrt]-svrfits.genSVR[MU1_id][MU2_drcrt]

                    # control unit discharge rate modulation while test unit is firing
                    ctrl_mod = np.append(ctrl_mod,np.nanmax(svrfits.genSVR[MU1_id][range(MU2_rcrt,MU2_drcrt)])-np.nanmin(svrfits.genSVR[MU1_id][range(MU2_rcrt,MU2_drcrt)]))

                    if normalization_ == "False":
                        dFret = np.append(dFret,df)
                    elif normalization_ == "ctrl_max_desc": # normalize deltaF values to control unit descending range during test unit firing
                        k = svrfits.genSVR[MU1_id][MU2_rcrt]-svrfits.genSVR[MU1_id][MU1_drcrt]
                        dFret = np.append(dFret,df/k)
                else:
                    controlU = 2 # MU 2 is control unit, 1 is test unit
                
                    #delta F: change in control MU discharge rate between test unit recruitment and derecruitment
                    df = svrfits.genSVR[MU2_id][MU1_rcrt]-svrfits.genSVR[MU2_id][MU1_drcrt]

                    # control unit discharge rate modulation while test unit is firing
                    ctrl_mod = np.append(ctrl_mod,np.nanmax(svrfits.genSVR[MU2_id][range(MU1_rcrt,MU1_drcrt)])-np.nanmin(svrfits.genSVR[MU2_id][range(MU1_rcrt,MU1_drcrt)]))

                    if normalization_ == "False":
                        dFret = np.append(dFret,df)
                    elif normalization_ == "ctrl_max_desc": # normalize deltaF values to control unit descending range during test unit firing
                        k = svrfits.genSVR[MU2_id][MU1_rcrt]-svrfits.genSVR[MU2_id][MU2_drcrt]
                        dFret = np.append(dFret,df/k)   

            # collect which MUs were control vs test
            controlMU.append(MUcombo_[-1][controlU-1])
            testMU.append(MUcombo_[-1][1-controlU//2])

        if clean_ == "True": # remove values that dont meet exclusion criteria
             rcrt_diff_bin = rcrt_diff>recruitment_difference_cutoff_
             corr_bin = R>corr_cutoff_
             ctrl_mod_bin = ctrl_mod>controlunitmodulation_cutoff_
             clns = np.asarray([rcrt_diff_bin&corr_bin&ctrl_mod_bin])
             dFret[~clns[0]] = np.nan

        if average_method_ == "test_unit_average": # average across all control units
            for ii in range(emgfile["NUMBER_OF_MUS"]):
                clean_indices  = [index for (index, item) in enumerate(testMU) if item == ii]
                if np.isnan(dFret[clean_indices]).all():
                    dFret_ret = np.append(dFret_ret,np.nan)
                else:
                    dFret_ret = np.append(dFret_ret, np.nanmean(dFret[clean_indices]))
                MUcombo_ret = np.append(MUcombo_ret,ii)
        else: # return all values and corresponding combinations
            dFret_ret = dFret
            MUcombo_ret = MUcombo_

    dF = pd.DataFrame({'dF':dFret_ret,'MU':MUcombo_ret})
    return dF