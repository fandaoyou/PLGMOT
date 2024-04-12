_base_ = './base.py'
model = dict(
    type='PLGMOT',
    up_thr=0.6
)
workflow = [('train', 1), ('update', 1)]
data = dict(
    train=dict(
        type='SeqDataset',
        box_refine=True,
        upper_thr=0.6,
        update_w=0.5,
        match_thr=0.4,
        seq_len=3,
        version=3
    )
)