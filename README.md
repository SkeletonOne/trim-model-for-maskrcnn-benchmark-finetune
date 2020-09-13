# trim-model-for-maskrcnn-benchmark-finetune
Trim pre-trained model for finetuning on custom dataset based on maskrcnn-benchmark (.pth format)

This code is heavily based on [trim_detectron_model](https://gist.github.com/wangg12/aea194aa6ab6a4de088f14ee193fd968) by wangg12. But this code support pretrained models which are in .pth format. 

For example, download a pretrained model in maskrcnn-benchmark [MODEL_ZOO](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/MODEL_ZOO.md), then modify the three arguments in the trim_model.py, run it and you will get a trimed model(without last layers). If the scheduler are also need to be removed, run the code that are commented.

Details can be found in [#15](https://github.com/facebookresearch/maskrcnn-benchmark/issues/15) in maskrcnn-benchmark.
