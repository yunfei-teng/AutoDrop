import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import tools
    
class Base_Model(nn.Module):
    def __init__(self):
        super(Base_Model, self).__init__()
        self.train_losses = []
        self.train_errors = []        
        self.test_losses = []
        self.test_errors = []
        self.optim_lrs = []
        self.step_counter = 0

    def load_model_params(self, flat_tensor):
        current_index = 0
        for parameter in self.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            parameter.data.copy_(flat_tensor[current_index:current_index+numel].view(size))
            current_index += numel

    def pull_model_params(self, flat_tensor, p=0.05):
        current_index = 0
        for parameter in self.parameters():
            numel = parameter.data.numel()
            size = parameter.data.size()
            leader = flat_tensor[current_index:current_index+numel].view(size)
            parameter.data.mul_(1-p).add_(p* leader)
            current_index += numel 

    def _forward_one_iteration(self, args, device, batch_idx, data, target):
        with torch.no_grad():
            data, target = data.to(device), target.to(device)
            output = self.forward(data)
            loss = F.cross_entropy(output, target)
        return loss.item()

    def _train_one_iteration(self, args, device, batch_idx, data, target, optimizer):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = self.forward(data)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()  
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        return loss, correct
        
    def train_one_epoch(self, args, device, train_loader, optimizer, epoch):
        self.train()
        acc_loss = 0
        acc_correct = 0
        acc_data_points = 0
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            loss, correct = self.train_one_iteration(args, device, batch_idx, data, target, optimizer)
            acc_loss += loss.item()* len(data)
            acc_correct += correct
            acc_data_points += len(data)
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break
        return acc_loss/acc_data_points, 100 - 100* acc_correct/acc_data_points

    def test_one_epoch(self, device, test_loader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.forward(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            test_accuracy))
        return test_loss, (100 - test_accuracy)


class Average_Model(Base_Model):
    def __init__(self):
        super(Average_Model, self).__init__()
        self.temp_positions1 = []
        self.optim_positions1 = []
        self.temp_positions2 = []
        self.optim_positions2 = []
        self.temp_positions3 = []
        self.temp_positions4 = []

        self.iter_counter = 0
        self.avg_counter = 0
        self.window_size = 1
        
        self.angle_velocities = []
        self.lr_decay_enabled = False
        self.lr_decay_flag = False

        self.max_momentum_angles = []
        self.min_momentum_angles = []
        self.avg_momentum_angles = []

        self.grad_velocities = []
        self.grad_productions = []
        self.grad_norms = []
        self.avg_grad_velocities = []
        self.avg_grad_productions = []
        self.avg_grad_norms = []

        self.param_distances = []
        self.param_distances_acc = []
        self.grad_variances = []
        self.p1_norms = []
        self.p2_norms = []
        self.p_dots = []

        self.thetas = []
        self.num_lr_decays = 0
        self.grad_decay_locked = False

        self.extra_counter = 0

    def train_one_iteration(self, args, device, batch_idx, data, target, optimizer):
        self.step_counter = self.step_counter + 1
        '''
        if args.use_momentum:
            if args.optimizer == 'sgd':
                momentum_name = 'momentum_buffer'
            num_params = 0
            mom_vector = torch.zeros(self.num_params)
            for p in optimizer.param_groups[0]['params']:
                param_state = optimizer.state[p]
                if momentum_name in param_state:
                    cur_state = param_state[momentum_name]
                    mom_vector[num_params:num_params+cur_state.numel()].copy_(cur_state.view(-1))
                    num_params += cur_state.numel()
                    first_pass_flag = False
                else:
                    first_pass_flag = True
        '''
        loss = self._train_one_iteration(args, device, batch_idx, data, target, optimizer)

        # 1. save model parameters between intervals
        '''
        if self.iter_counter % args.ins_interval == 0 and self.avg_counter < self.window_size:
            self.avg_counter += 1
            self.add_param_vector(c=1)
        elif self.iter_counter % args.ins_interval == 0 and self.avg_counter == self.window_size:
            self.avg_counter = 0
            self.iter_counter += 1
            average_position = torch.stack(self.temp_positions1).mean(dim=0)
            self.optim_positions1 += [average_position]
            self.optim_positions1 = self.optim_positions1[-3:]
            self.temp_positions1 = []
        else:
            self.iter_counter += 1
        '''

        # 2. save model parameters from the beginning iterations
        if batch_idx < (self.window_size+1):
            self.add_param_vector(c=2)
        elif batch_idx == (self.window_size+1):
            average_position = torch.stack(self.temp_positions2).mean(dim=0)
            self.optim_positions2 += [average_position]
            self.optim_positions2 = self.optim_positions2[-3:]
            self.thetas += [args.theta]

            variance = 0
            for i in range(1, len(self.temp_positions2)):
                variance += (self.temp_positions2[i]-self.temp_positions2[i-1]).norm().item()
            self.grad_variances += [variance/len(self.temp_positions2)]
            if len(self.optim_positions2) == 1:
                self.init_position = self.optim_positions2[0]
            if len(self.optim_positions2) > 1:
                self.param_distances += [(self.optim_positions2[-1]-self.optim_positions2[-2]).norm().item()]
                self.param_distances_acc += [(self.optim_positions2[-1]-self.init_position).norm().item()]
            self.temp_positions2 = []
            
            if self.get_angle_velocity(batch_idx) is not None:
                p1, p2, angle_velocity = self.get_angle_velocity(batch_idx)
                self.p1_norms += [p1.norm().item()]
                self.p2_norms += [p2.norm().item()]
                self.p_dots   += [torch.sum(p1* p2).item()]
                self.angle_velocities += [angle_velocity]

                # method 1
                '''
                if args.is_auto_ld:
                    if args.model == 'lenet':
                        angle_velocity_th = 90.0 
                    elif args.model == 'resnet18':
                        angle_velocity_th = 95.0 
                    if angle_velocity >= angle_velocity_th and self.lr_decay_enabled:
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']* args.ld_factor
                        self.lr_decay_enabled = False
                        self.lr_decay_flag = True
                    else:
                        self.lr_decay_flag = False
                    if not self.lr_decay_enabled and not self.lr_decay_flag:
                        self.lr_decay_enabled = True
                '''

                # method 2
                if not self.grad_decay_locked:
                    if len(self.avg_grad_productions) >= 1:
                        if self.avg_grad_productions[-1] < args.grad_prod_stop:
                            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']* args.ld_factor
                            self.grad_decay_locked = True
                
                if not self.grad_decay_locked:
                    if args.is_auto_ld and len(self.angle_velocities) >= 2:
                        angle_diff = self.angle_velocities[-1]-self.angle_velocities[-2]
                        lr_decay_flag = angle_diff > -args.theta and angle_diff < args.theta # and angle_velocity >= 90
                        self.lr_decay_enabled = self.lr_decay_enabled | lr_decay_flag
                        if self.lr_decay_enabled:
                            self.extra_counter += 1
                            if args.var_rd:
                                self.add_param_vector(c=3,p=self.extra_counter)
                            # self.add_param_vector(c=4,p=1,device='cuda')
                            if self.extra_counter == args.extra_epochs:
                                if args.auto_theta:
                                    args.theta = args.theta / args.ld_factor
                                    if args.upper_theta > 0:
                                        args.theta = min(args.upper_theta, args.theta)
                                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']* args.ld_factor
                                if args.lower_lr > 0:
                                    optimizer.param_groups[0]['lr'] = max(args.lower_lr, optimizer.param_groups[0]['lr'])
                                if args.var_rd:
                                    denominator = (1+self.extra_counter)*self.extra_counter/2
                                    self.load_model_params(torch.stack(self.temp_positions3).sum(dim=0)/denominator)
                                self.temp_positions3 = []
                                self.temp_positions4 = []
                                self.lr_decay_enabled = False
                                self.extra_counter = 0

        # (a) play with gradients
        '''
        num_params = 0
        grad_vector = torch.zeros(self.num_params)
        for n, l in enumerate(self.parameters()):
            cur_data, cur_grad = l.data.cpu(), l.grad.data.cpu()
            grad_vector[num_params:num_params+cur_grad.numel()].copy_(cur_grad.view(-1))
            num_params += cur_data.numel()
        
        if batch_idx > 1:
            self.grad_velocities  += [tools.calculate_angle(self.last_grad_vector, grad_vector)]
            self.grad_productions += [(self.last_grad_vector* grad_vector).norm().item()]
            self.grad_norms += [grad_vector.norm().item()]
        if batch_idx == args.train_loader_len:
            self.avg_grad_velocities  += [sum(self.grad_velocities)/len(self.grad_velocities)]
            self.avg_grad_productions += [sum(self.grad_productions)/len(self.grad_productions)]
            self.avg_grad_norms += [sum(self.grad_norms)/len(self.grad_norms)]
            self.grad_velocities  = []
            self.grad_productions = []
            self.grad_norms = []
        self.last_grad_vector = grad_vector
        '''
        self.avg_grad_velocities  += [0]
        self.avg_grad_productions += [0]
        self.avg_grad_norms += [0]

        # (b) play with momentum
        '''
        if args.use_momentum:
            if batch_idx == 1:
                self.momentum_angles = []
            if not first_pass_flag:
                self.momentum_angles += [tools.calculate_angle(mom_vector, grad_vector)]
            if batch_idx == args.train_loader_len:
                self.max_momentum_angles += [max(self.momentum_angles)]
                self.min_momentum_angles += [min(self.momentum_angles)]
                self.avg_momentum_angles += [sum(self.momentum_angles)/len(self.momentum_angles)]
                self.momentum_angles = []
        '''
        self.max_momentum_angles += [0]
        self.min_momentum_angles += [0]
        self.avg_momentum_angles += [0]
        return loss
    
    def get_angle_velocity(self, batch_idx):
        if len(self.optim_positions2) > 2:
            assert batch_idx == self.window_size + 1
            p1 = self.optim_positions2[-1] - self.optim_positions2[-2]
            p2 = self.optim_positions2[-2] - self.optim_positions2[-3]
            return p1, p2, tools.calculate_angle(p1, p2)
        else:
            return None

    def add_param_vector(self, c = 1, p = 1.0, device='cpu'):
        param_vector = torch.zeros(self.num_params, device=device)
        num_params = 0
        for n, l in enumerate(self.parameters()):
            cur_data = l.data.cpu()
            param_vector[num_params:num_params+cur_data.numel()].copy_(cur_data.view(-1))
            num_params += cur_data.numel()
        if c == 1:
            self.temp_positions1 += [param_vector*p]
        elif c == 2:
            self.temp_positions2 += [param_vector*p]
        elif c == 3:
            self.temp_positions3 += [param_vector*p]

# ResNets
class _BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(_BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(Average_Model):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def initialize(self):
        self.num_params = 0
        for n, l in enumerate(self.parameters()):
            self.num_params += l.data.numel()

        num_params = 0
        self.init_param = torch.zeros(self.num_params, requires_grad=False)
        for n, l in enumerate(self.parameters()):
            self.init_param[num_params:num_params+l.data.numel()].copy_(l.data.view(-1))
            num_params += l.data.numel()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes):
    return ResNet(_BasicBlock, [2, 2, 2, 2], num_classes)

def ResNet34(num_classes):
    return ResNet(_BasicBlock, [3, 4, 6, 3], num_classes)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out

class Wide_ResNet(Average_Model):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor
        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]
        self.conv1 = conv3x3(3,nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def initialize(self):
        self.num_params = 0
        for n, l in enumerate(self.parameters()):
            self.num_params += l.data.numel()

        num_params = 0
        self.init_param = torch.zeros(self.num_params, requires_grad=False)
        for n, l in enumerate(self.parameters()):
            self.init_param[num_params:num_params+l.data.numel()].copy_(l.data.view(-1))
            num_params += l.data.numel()

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNet18i(Average_Model):
    def __init__(self):
        super(ResNet18i, self).__init__()
        self.resnet18 = torchvision.models.resnet18(pretrained=False)

    def initialize(self):
        self.num_params = 0
        for n, l in enumerate(self.parameters()):
            self.num_params += l.data.numel()
        # self.resnet18 = nn.DataParallel(self.resnet18)

    def forward(self, x):
        return self.resnet18(x)