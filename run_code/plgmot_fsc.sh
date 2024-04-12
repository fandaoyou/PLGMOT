python train.py --config configs/plgmot_fsc.py --work-dir ./work_dirs/plgmot_fsc
python detect.py --exp_id plgmot_fsc --config work_dirs/plgmot_fsc/plgmot_fsc.py --checkpoint work_dirs/plgmot_fsc/latest.pth
python hmaa_track.py --exp_id plgmot_fsc
python sort.py --exp_id plgmot_fsc
python ./tools/show_metric.py