from .contrastive_loss_with_RKD import ContrastiveLossWithRKD
from .proposal_loss_with_DTW import ProposalLossWithDTW
from .universal_logit_distillation import UniversalLogitDistillation
from .propose_with_proj import ProposalLossWithProj
from .emo_loss import EMOLoss
from .em_kd import EMKDLoss

criterion_list = {
    "contrastive_rkd": ContrastiveLossWithRKD,
    "proposal_dtw": ProposalLossWithDTW,
    "universal_logit": UniversalLogitDistillation,
    "proposal_proj": ProposalLossWithProj,
    "emo_loss": EMOLoss,
    "em_kd": EMKDLoss,
}

def build_criterion(args):
    if args.kd_loss_type not in criterion_list.keys():
        raise ValueError(f"Criterion {args.kd_loss_type} not found.")
    return criterion_list[args.kd_loss_type](args)