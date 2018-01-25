import numpy as np
import numpy.random as nr


class ProposalDistribution(object):
    def __init__(self, rand, log_pdf):
        self.rand = rand
        self.log_pdf = log_pdf


def metropolis_hastings_proposal(proposal_distribution):

    def kernel_function(x_old, x_new, joint_pdf):
        return proposal_distribution.rand(x_old)

    return kernel_function


def metropolis_hastings_acceptance(proposal_distribution):

    def kernel_function(x_old, x_new, joint_pdf):
        level = (joint_pdf.log_pdf(x_new) - joint_pdf.log_pdf(x_old) +
                 proposal_distribution.log_pdf(x_old, given=x_new) -
                 proposal_distribution.log_pdf(x_new, given=x_old))
        if np.log(nr.rand()) <= level:
            return x_new
        else:
            return x_old

    return kernel_function
