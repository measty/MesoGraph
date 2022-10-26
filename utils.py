#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:15:25 2020

@author: u1876024
"""
import numpy as np
import torch
import pickle
import math

"""Various utility functions for working with MesoGraph models"""

USE_CUDA = torch.cuda.is_available()
device = {True: "cuda", False: "cpu"}[USE_CUDA]


def cuda(v):
    if USE_CUDA:
        return v.cuda()
    return v


def toTensor(v, dtype=torch.float, requires_grad=True):
    return cuda(torch.from_numpy(np.array(v)).type(dtype).requires_grad_(requires_grad))


def toNumpy(v):
    if type(v) is not torch.Tensor:
        return np.asarray(v)
    if USE_CUDA:
        return v.detach().cpu().numpy()
    return v.detach().numpy()


def pickleLoad(ifile):
    with open(ifile, "rb") as f:
        return pickle.load(f)


def pickleSave(ofile, obj):
    with open(ofile, "wb") as f:
        pickle.dump(obj, f)


def add_missranked(df):
    """for each core, calculates over all the cores with a different label, the
    proportion of those cores which are ranked by score opposite to that which
    would be expecteed by label."""
    df.set_index("core")
    miss = []
    for core in df.index:
        lab, s = df.loc[core, "y"], df.loc[core, "y_pred"]
        miss.append(
            (
                df[np.logical_and(df["y"].values < lab, df["y_pred"].values > s)].shape[
                    0
                ]
                + df[
                    np.logical_and(df["y"].values > lab, df["y_pred"].values < s)
                ].shape[0]
            )
            / sum(df["y"].values != lab)
        )
    df["miss"] = miss
    return df


def map_ind(ind, dataset="mesobank"):
    # maps core ind to slide
    if isinstance(ind, str):
        ind = int(ind.split("-")[0])
    if dataset == "mesobank":
        if ind < 3 or ind > 49:
            raise ValueError("ind is not a valid core")
        if ind < 26:
            slide = 0
        else:
            slide = 1
        return slide

    slide_offsets = [3, 13, 25, 37]
    if ind < 3 or ind > 44:
        raise ValueError("ind is not a valid core")
    if ind < 13:
        slide = 0
    elif ind < 25:
        slide = 1
    elif ind < 37:
        slide = 2
    else:
        slide = 3
    return slide


def get_short_names(used_feats):
    short_names = []
    for f in used_feats:
        if "Smoothed" in f:
            short_names.append(": ".join(["Smoothed"] + f.split(": ")[2:]))
        elif "Circle: Diameter" in f:
            if "Haralick" in f:
                short_names.append(
                    ": ".join(["Haralick"] + [f.split(": ")[3]] + [f[-5:]])
                )
            else:
                app = f[-5:]
                if app == ".dev.":
                    app = "Std.dev."
                short_names.append(": ".join([f.split(": ")[3]] + [app]))
        elif "ROI" in f:
            short_names.append(": ".join(["ROI"] + [f.split(": ")[2]] + [f[-5:]]))
        else:
            short_names.append(f)
        if len(short_names[-1]) > 28:
            split_n = short_names[-1].split(": ")
            short_names[-1] = (
                ": ".join(split_n[0 : math.ceil(len(split_n) / 2)])
                + "\n"
                + ": ".join(split_n[math.ceil(len(split_n) / 2) :])
            )
    return short_names
