class JointProbability(object):
    def __init__(self, log_pdf, dlog_pdf):
        self.log_pdf = log_pdf
        self.dlog_pdf = dlog_pdf


class MCMCTransition(object):
    def __init__(self, joint_pdf, kernel_fns):
        self.joint_pdf = joint_pdf
        self.kernel_fns = kernel_fns

    def step(self, x_old):
        x_new = x_old
        for fn in self.kernel_fns:
            x_new = fn(x_old, x_new, self.joint_pdf)
        return x_new

    def sample(self, init, steps):
        x = init
        for _ in range(steps):
            x = self.step(x)
            yield x
