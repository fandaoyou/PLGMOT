python train.py --config configs/plgmot_fsc.py --work-dir ./work_dirs/plgmot_fsc1
python detect.py --exp_id plgmot_fsc1 --config work_dirs/plgmot_fsc1/plgmot_fsc.py --checkpoint work_dirs/plgmot_fsc1/latest.pth
python hmaa_track.py --exp_id plgmot_fsc1
python sort.py --exp_id plgmot_fsc1

python train.py --config configs/plgmot_fsc.py --work-dir ./work_dirs/plgmot_fsc2
python detect.py --exp_id plgmot_fsc2 --config work_dirs/plgmot_fsc2/plgmot_fsc.py --checkpoint work_dirs/plgmot_fsc2/latest.pth
python hmaa_track.py --exp_id plgmot_fsc2
python sort.py --exp_id plgmot_fsc2