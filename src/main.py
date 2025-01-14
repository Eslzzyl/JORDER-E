import torch

import data
import loss
import model
import utility
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)

    # pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(pytorch_total_params)

    loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader, model, loss, checkpoint)

    while not t.terminate():
        t.train()
        t.test()

    checkpoint.done()
