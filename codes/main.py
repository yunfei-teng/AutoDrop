from __future__ import print_function
import time
import argparse
import numpy
import random
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import time
import data
import models
import options

def main():
    args = options.Options().parse()
    device = torch.device("cuda" if args.use_cuda else "cpu")
    print('model will be trained on', device)
    torch.manual_seed(args.seed)
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    
    if args.dataset == 'cifar10':
        num_class = 10
    elif args.dataset == 'cifar100':
        num_class = 100
    if args.model == 'resnet18':
        model = models.ResNet18(num_class).to(device)
    elif args.model == 'resnet34':
        model = models.ResNet34(num_class).to(device)
    elif args.model == 'wrn28x10':
        model = models.Wide_ResNet(28, 10, args.dropout_rate, num_class).to(device)
    elif args.model == 'wrn40x10':
        model = models.Wide_ResNet(40, 10, args.dropout_rate, num_class).to(device)
    elif args.model == 'resnet18i':
        model = models.ResNet18i().to(device)

    model.initialize()
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    train_loader, test_loader = data.get_data_loader(args)
    args.train_loader_len = len(train_loader)
    print('The length of training data loader is %d'%args.train_loader_len)

    if args.optimizer == 'sgd':
        if args.use_momentum:
            if args.no_nesterov:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_value, nesterov=False)
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum_value, nesterov=True)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer.param_groups[0]['weight_decay'] = args.wd

    if args.is_auto_ld:
        lr_scheduler = None
    elif args.is_preset_ld:
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=args.ld_factor)
        if args.dataset == 'imagenet':
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=args.ld_factor)
    else:
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=args.ld_period, gamma=args.ld_factor, last_epoch=-1)

    epoch_begin_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_error = model.train_one_epoch(args, device, train_loader, optimizer, epoch)
        test_loss, test_error = model.test_one_epoch(device, test_loader)
        model.train_losses += [train_loss]
        model.train_errors += [train_error]
        model.test_losses  += [test_loss]
        model.test_errors  += [test_error]
        model.optim_lrs += [optimizer.param_groups[0]['lr']]
        epoch_period = (time.time() - epoch_begin_time)/(epoch* 3600)
        print('--(epoch %d, %.2fh/%.2fh)'%(epoch, epoch_period, epoch_period*(args.epochs-epoch-1)),
                'Learning rate is', model.optim_lrs[-1], '\n')
        if len(model.angle_velocities) > 0:
            print(epoch, 'my angle velocity is', model.angle_velocities[-1])
        if not args.is_auto_ld:
            lr_scheduler.step()
        
        if args.save_model and epoch % 5 == 0:
            torch.save(model.state_dict(), "%s/%s_%s_%d.pt"%(args.expr_dir, args.dataset, args.model, epoch))
        if args.save_model:
            torch.save({
                'lrs': model.optim_lrs,
                'train_losses': model.train_losses,
                'train_errors': model.train_errors,
                'test_losses': model.test_losses,
                'test_errors': model.test_errors,
                'param_distances':model.param_distances,
                'param_distances_acc':model.param_distances_acc,
                'angle_velocities':model.angle_velocities,
                'avg_momentum_angles': model.avg_momentum_angles,
                'avg_grad_productions': model.avg_grad_productions,
                'avg_grad_velocities': model.avg_grad_velocities,
                'avg_grad_norms': model.avg_grad_norms,
                'thetas': model.thetas,
                'p_dots': model.p_dots,
            }, '%s/%s_%s_stat.pt'%(args.expr_dir, args.dataset, args.model))

if __name__ == '__main__':
    main()