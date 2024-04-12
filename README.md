A Pytorch Implementation of Prototype Learning based Generic Multiple Object Tracking

## Installation

```
conda create -n PLGMOT
conda activate PLGMOT
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch -y

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .

pip install -r requirments.txt
```

## Data preparation

The FSC147 and GMOT40 datasets can be downloaded from [FSC147](https://github.com/cvlab-stonybrook/LearningToCountEverything) and [GMOT40](https://spritea.github.io/GMOT40/download.html).
After downloading, you should prepare the data in the following structure:

```
data
   |——————FSC147
   |        └——————FSC147_384_V2
   |                └——————images_384_VarV2
   |                └——————annotation_FSC147_384.json
   |                ...
   └——————GMOT40
   |         └——————GenericMOT_JPEG_Sequence
   |         └——————template_box_by_seq_for_global_track
   |         └——————track_label
configs
...
```

Generate initial pseudo bounding boxes:

```
python ./tools/gen_pseudo_label.py
```

## Train

To train PLGMOT on FSC(with 3270 images):

```
python train.py --config configs/plgmot_fsc.py --work-dir ./work_dirs/plgmot_fsc
```

To train PLGMOT on FSC+(with 5588 images):

```
python train.py --config configs/plgmot_fsc+.py --work-dir ./work_dirs/plgmot_fsc+
```

## Evaluation

Run detect.py and hmaa_track.py to get the detection and tracking results.

e.g.

```
python detect.py --exp_id plgmot_fsc --config work_dirs/plgmot_fsc/plgmot_fsc.py --checkpoint work_dirs/plgmot_fsc/latest.pth
python hmaa_track.py --exp_id plgmot_fsc
```

The results will be saved in `./work_dirs/{exp_id}/det` and `./track_results/{exp_id}` respectively.

Run `python ./tools/show_metric.py` to print the metrics.

Or you can just run `sh ./run_code/plgmot_fsc.sh`.

