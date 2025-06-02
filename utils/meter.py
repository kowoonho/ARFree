class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99, save_seq=True):
        self.momentum = momentum
        self.save_seq = save_seq
        if self.save_seq:
            self.vals, self.steps = [], []
        self.reset()

    def ckpt(self):
        return {'vals':self.vals,'avg':self.avg,'steps':self.steps}

    def load(self,dict_ckpt):
        self.vals = dict_ckpt['vals']
        if len(self.vals)>0:
            self.val = self.vals[-1]
        self.avg = dict_ckpt['avg']
        self.steps = dict_ckpt['steps']

    def reset(self):
        self.val, self.avg = None, 0

    def update(self, val, step=None):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)

    def synchronize_and_update(self, accelerator, val, step=None):
        """
        Warning: does not synchronize the deque!
        """
        val = accelerator.reduce(val, reduction = 'mean')
        val = val.item()
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        if self.save_seq:
            self.vals.append(val)
            if step is not None:
                self.steps.append(step)
        return val