def set_seed(seed=42):
    import random, numpy, torch
    random.seed(seed)
    numpy.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def unroller(e):
    return tuple(zip(*e))