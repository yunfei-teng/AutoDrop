import os
import argparse
import datetime
import warnings

import torch
import torch.backends.cudnn as cudnn

import tools

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ''' set up arguments '''
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--test_batch_size', type=int, default=1000)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--extra_epochs', type=int, default=10)
        parser.add_argument('--lr', type=float, default=0.01)
        parser.add_argument('--p' , type=float, default=0.05)
        parser.add_argument('--lower_lr', type=float, default=-1)
        parser.add_argument('--ld_period', type=int, default=4)
        parser.add_argument('--ld_factor', type=float, default=1.0)
        parser.add_argument('--is_preset_ld', action='store_true', default=False)
        parser.add_argument('--is_auto_ld', action='store_true', default=False)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument('--no_cuda', action='store_true', default=False)
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--wd', type=float, default=0.0)
        parser.add_argument('--theta', type=float, default=0.5)
        parser.add_argument('--delta', type=float, default=0.5)
        parser.add_argument('--upper_theta', type=float, default=-1)
        parser.add_argument('--auto_theta', action='store_true', default=False)
        parser.add_argument('--grad_prod_stop', type=float, default=-1)
        parser.add_argument('--use_momentum', action='store_true', default=False)
        parser.add_argument('--no_nesterov', action='store_true', default=False)
        parser.add_argument('--momentum_value', type=float, default=0.9)
        parser.add_argument('--dropout_rate', type=float, default=0.3)
        parser.add_argument('--var_rd', action='store_true', default=False)
        parser.add_argument('--dry_run', action='store_true', default=False)
        parser.add_argument('--seed', type=int, default=1)
        parser.add_argument('--log_interval', type=int, default=100)
        parser.add_argument('--ins_interval', type=int, default=200)
        parser.add_argument('--save_model', action='store_true', default=True)
        parser.add_argument('--exp_name', type = str, default='pca')
        parser.add_argument('--dataset', type=str, default='mnist')
        parser.add_argument('--model', type=str, default='lenet')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument('--suffix', default='', type=str) # additional parameters
        parser.add_argument('--resume', default=None, type=str)
        parser.add_argument('--seed_loader', action='store_true', default=False)
        parser.add_argument('--parallelized', action='store_true', default=False)
        self.initialized = True
        return parser

    def get_argsions(self):
        ''' get argsions from parser '''
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(description='adaptive momentum SGD method')
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_argsions(self, args):
        ''' Print and save argsions
            It will print both current argsions and default values(if different).
            It will save argsions into a text file / [checkpoints_dir] / args.txt
        '''
        message = str(datetime.datetime.now())
        message += '\n----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)
        
        tools.mkdirs(args.expr_dir) # first remove existing directory and then create a new one
        file_name = os.path.join(args.expr_dir, 'args.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
    
    def parse(self):
        ''' Parse our argsions, create checkpoints directory suffix, and set up gpu device. '''
        args = self.get_argsions()

        # process args.suffix
        if args.suffix:
            suffix = ('_' + args.suffix.format(**vars(args))) if args.suffix != '' else ''
            args.exp_name = args.exp_name + suffix
        args.expr_dir = os.path.join(args.checkpoints_dir, args.exp_name)
        args.use_cuda = not args.no_cuda and torch.cuda.is_available()
        args.use_momentum = args.use_momentum or args.optimizer == 'adam'
        assert not (args.is_auto_ld and args.is_preset_ld)
        self.print_argsions(args)
        self.args = args
        return self.args