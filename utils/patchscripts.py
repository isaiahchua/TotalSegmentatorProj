import numpy as np
import torch
import torch.nn.functional as F
from utils import printProgressBarRatio

def RunModelOnPatches(model, inp, out_chns, patch_size, step_size,
                   device=torch.device("cpu")):
    """
    model: 3D segmentation model
    inp: 5D tensor (N, C, H, W, D)
    out_chns: number of output classes from model
    patch_size: int or tuple of ints of the patch dimensions
    step_size: int or tuple of ints of the step size between each patch
    device: the torch device to run the patch through the model with
    """
    if isinstance(patch_size, int):
        xp, yp, zp = patch_size, patch_size, patch_size
    elif isinstance(patch_size, tuple):
        assert len(patch_size) == 3
        xp, yp, zp = patch_size
    else:
        raise Exception("Invalid patch_size given")
    if isinstance(step_size, int):
        xs, ys, zs = step_size, step_size, step_size
    elif isinstance(step_size, tuple):
        assert len(step_size) == 3
        xs, ys, zs = step_size
    else:
        raise Exception("Invalid step_size given")
    sh = np.asarray(inp.shape[2:5])
    bbox = np.asarray((xp, yp, zp))
    mask = sh < bbox
    if mask.any():
        diff = np.expand_dims(bbox[mask] - sh[mask], 1)
        padding = np.zeros((3, 2), dtype=np.int16)
        padding[mask] = np.concatenate((diff//2, diff//2 + diff%2), axis=1)
        inp = F.pad(inp, padding[::-1].flatten().tolist())
    sh = np.asarray(inp.shape[2:5])
    out = torch.zeros(inp.shape[0], out_chns, *sh, dtype=torch.float32)
    x_nsteps = (sh[0] + xs - xp) // xs
    x_rem = (sh[0] + xs - xp) % xs
    x_steps = [(i*xs, i*xs + xp) for i in range(x_nsteps)]
    if x_rem > 0:
        x_lstep = (x_steps[-1][0] + x_rem, x_steps[-1][1] + x_rem)
        x_steps.append(x_lstep)
    y_nsteps = (sh[1] + ys - yp) // ys
    y_rem = (sh[1] + ys - yp) % ys
    y_steps = [(i*ys, i*ys + yp) for i in range(y_nsteps)]
    if y_rem > 0:
        y_lstep = (y_steps[-1][0] + y_rem, y_steps[-1][1] + y_rem)
        y_steps.append(y_lstep)
    z_nsteps = (sh[2] + zs - zp) // zs
    z_rem = (sh[2] + zs - zp) % zs
    z_steps = [(i*zs, i*zs + zp) for i in range(z_nsteps)]
    if z_rem > 0:
        z_lstep = (z_steps[-1][0] + z_rem, z_steps[-1][1] + z_rem)
        z_steps.append(z_lstep)
    model.eval()
    i = 1
    total_steps = len(x_steps) * len(y_steps) * len(z_steps)
    with torch.no_grad():
        for x1, x2 in x_steps:
            for y1, y2 in y_steps:
                for z1, z2 in z_steps:
                    patch = inp[:, :, x1:x2, y1:y2, z1:z2].to(device)
                    preds = model(patch).cpu()
                    out[:, :, x1:x2, y1:y2, z1:z2] += preds
                    printProgressBarRatio(i, total_steps, prefix="Patch")
                    i += 1
    return out
