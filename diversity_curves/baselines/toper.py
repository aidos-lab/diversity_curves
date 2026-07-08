import toper
import numpy as np


def toper_emb(graph_list, filtrations, sub=True, num_segm=501):
    all_sub = []
    for filt in filtrations:
        # , "weight"]:
        F = toper.main(graph_list, [filt], num_segm=num_segm)
        if filt == "fricci":
            filt = "forricci"
        elif filt == "oricci":
            filt = "olricci"

        if sub:
            array = np.array(F[f"{filt}_sub"])
        else:
            array = np.array(F[f"{filt}_super"])
        all_sub.append(array)
    all_sub = np.concatenate(all_sub, axis=1)
    return all_sub
