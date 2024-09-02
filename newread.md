
```bash
conda deactivate
conda env remove -n ovad -y
conda create -n ovad python=3.8 -y
conda activate ovad
python -m pip install -U pip
conda install -y pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 \
  cudatoolkit=11.3 -c pytorch -c conda-forge
pip install pycocotools
pip install -U -r newreq.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install openai-clip open-clip-torch scikit-learn

# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install dill GPUtil detectron2 ipdb json lvis matplotlib nltk numpy Pillow pycocotools tabulate

ln -s ~/dataset_symlinks/coco/ ~/repos/ovad-benchmark-code/datasets


# ll ~/dataset_symlinks/coco/
# total 0
# lrwxrwxrwx 1 gings lmb-mit 48 Sep  2 12:15 annotations -> /misc/lmbraid21/gings/datasets/coco/annotations/
# lrwxrwxrwx 1 gings lmb-mit 52 Sep  2 12:15 train2017 -> /misc/lmbraid21/gings/datasets/coco/images/train2017
# lrwxrwxrwx 1 gings lmb-mit 50 Sep  2 12:15 val2017 -> /misc/lmbraid21/gings/datasets/coco/images/val2017

# below generates two new annotation json files 
# $datasets/coco/annotations/instances_train2017_base.json 
# training file with only base class (48) annotations 
# $datasets/coco/annotations/instances_val2017_base_novel17.json 
# validation file with base (48) and novel17 (17) class annotations
python tools/make_ovd_json.py --base_novel base \
  --json_path datasets/coco/annotations/instances_train2017.json
python tools/make_ovd_json.py --base_novel base_novel17 \
  --json_path datasets/coco/annotations/instances_val2017.json

# extract the OVAD-box instances from the annotations for the box oracle evaluation
# should generate datasets/ovad_box_instances
python tools/extract_obj_boxes.py

# precompute text feat
python tools/dump_attribute_features.py --out_dir datasets/text_representations   \
  --save_obj_categories --save_att_categories --fix_space \
  --prompt none \
  --avg_synonyms --not_use_object --prompt_att none 

# compute numbers
python ovamc/ova_open_clip.py --model_arch "ViT-B-32" -bs 50  --pretrained laion2b_e16 --prompt a
# ALL mAP 16.98
# - HEAD: 44.3
# - MEDIUM: 18.45
# - TAIL: 5.47

```


