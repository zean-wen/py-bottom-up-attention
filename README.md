# Bottom-up Attention with Detectron2 

The detectron2 system with **exact same model and weight** as the Caffe VG Faster R-CNN provided in [bottom-up-attetion](https://github.com/peteanderson80/bottom-up-attention).

The features extracted from this repo is compatible with LXMERT code and pre-trained models [here](https://github.com/airsplay/lxmert).
The original [bottom-up-attetion] is implemented based on [Caffe](https://github.com/BVLC/caffe), which is not easy to install and is inconsistent with the training code in PyTorch.
Our project thus transfers the weights and models to [detectron2](https://github.com/facebookresearch/detectron2) that could be few-line installed and has PyTorch front-end.

## Installation
```
git clone https://github.com/airsplay/py-bottom-up-attention.git
cd py-bottom-up-attention

# Install python libraries
pip install -r requirements.txt
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# Install detectron2
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

# or, as an alternative to `setup.py`, do
# pip install [--editable] .
```

## Demos

### Object Detection
[demo vg detection](demo/demo_vg_detection.ipynb)

### Feature Extraction 
1. Single image: [demo extraction](demo/demo_feature_extraction.ipynb)
2. Batchwise extraction: [demo batchwise extraction](demo/demo_batchwise_feature_extraction.ipynb)

## Feature Extraction Scripts for LXMERT
1. For MS COCO (VQA): [vqa script](demo/detectron2_mscoco_proposal_maxnms.py)


## References

Detectron2:
```BibTeX
@misc{wu2019detectron2,
  author =       {Yuxin Wu and Alexander Kirillov and Francisco Massa and
                  Wan-Yen Lo and Ross Girshick},
  title =        {Detectron2},
  howpublished = {\url{https://github.com/facebookresearch/detectron2}},
  year =         {2019}
}
```

Bottom-up Attention:
```BibTeX
@inproceedings{Anderson2017up-down,
  author = {Peter Anderson and Xiaodong He and Chris Buehler and Damien Teney and Mark Johnson and Stephen Gould and Lei Zhang},
  title = {Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering},
  booktitle={CVPR},
  year = {2018}
}
```

LXMERT:
```BibTeX
@inproceedings{tan2019lxmert,
  title={LXMERT: Learning Cross-Modality Encoder Representations from Transformers},
  author={Tan, Hao and Bansal, Mohit},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing},
  year={2019}
}
```

