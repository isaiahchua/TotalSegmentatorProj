import torch

def PatchSampler3D(model, inp, out_chns, patch_size, step_size,
                   device=torch.device("cpu")):
    shape = inp.shape
    out = torch.zeros(shape[0], out_chns, *shape[2:5], dtype=torch.float32)
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
    if shape[0] >= xp:
        x_nsteps = (shape[0] + xs - xp) // xs
        x_rem = (shape[0] + xs - xp) % xs
        x_steps = [(i*xs, i*xs + xp) for i in range(x_nsteps)]
        if x_rem > 0:
            x_lstep = (x_steps[-1][0] + x_rem, x_steps[-1][1] + x_rem)
            x_steps.append(x_lstep)
    else:
        x_steps = [(0, shape[0])]
    if shape[1] >= yp:
        y_nsteps = (shape[0] + ys - yp) // ys
        y_rem = (shape[0] + ys - yp) % ys
        y_steps = [(i*ys, i*ys + yp) for i in range(y_nsteps)]
        if y_rem > 0:
            y_lstep = (y_steps[-1][0] + y_rem, y_steps[-1][1] + y_rem)
            y_steps.append(y_lstep)
    else:
        y_steps = [(0, shape[1])]
    if shape[2] >= zp:
        z_nsteps = (shape[0] + zs - zp) // zs
        z_rem = (shape[0] + zs - zp) % zs
        z_steps = [(i*zs, i*zs + zp) for i in range(z_nsteps)]
        if z_rem > 0:
            z_lstep = (z_steps[-1][0] + z_rem, z_steps[-1][1] + z_rem)
            z_steps.append(z_lstep)
    else:
        z_steps = [(0, shape[2])]
    model.eval()
    with torch.no_grad():
        for x1, x2 in x_steps:
            for y1, y2 in y_steps:
                for z1, z2 in z_steps:
                    patch = inp[x1:x2, y1:y2, z1:z2].to(device)
                    preds = model(patch).cpu()
                    out[:, :, x1:x2, y1:y2, z1:z2] += preds
    return out
