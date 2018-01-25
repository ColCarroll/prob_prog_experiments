import scipy.stats as st

from base import MCMCTransition
from metropolis import ProposalDistribution, metropolis_hastings_proposal, metropolis_hastings_acceptance


def random_walk_metropolis(joint_pdf, scale=0.1):
    proposal = ProposalDistribution(
        rand=lambda x: st.norm(loc=x, scale=scale).rvs(),
        log_pdf=lambda x, given: st.norm(loc=given, scale=scale).logpdf(x)
    )

    return MCMCTransition(
        joint_pdf=joint_pdf,
        kernel_fns=[
            metropolis_hastings_proposal(proposal),
            metropolis_hastings_acceptance(proposal)
        ]
    )
