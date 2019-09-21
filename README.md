# fastmri-2019-challenge

This my submission to the fastMRI MRI reconstruction challenge as a representative of the Mayo Digial Health Sciences team: https://fastmri.org/submission_guidelines 

We developed a feature loss function using an autoencoder trained on the ground truth images. The loss function was able to improve on all performance benchmarks (NMSE,SSIM,PSNR) used in the contest for all the different models we tested on. We believe that this straightforward methodology is valuable for any researcher working on the challenging MRI construction problem. Due to time constraints we only trained/tested on the single-coil track of the challenge. It would be interesting to know how the feature loss performs on the multi-coil problem so please feel free to contact me if you were able to get some results!

As you can see most of the code for data loading/preprocessing comes from the fastMRI library: https://github.com/facebookresearch/fastMRI. As such much of the instructions on how to use this repo are the same as well.

On top of the requirements listed in `requirements.txt`, you will need to download the Fastai library: https://docs.fast.ai/ if you wish to try out the modified U-nets we used as well.

The one-page abstract will be linked here as soon as the contest is over!

You can contact me (Wei-Yin Ko) at ko.weiyin at mayo.edu if you have any questions about this project.

# Examples:

training the autoencoder:
```
python train_auto.py --challenge singlecoil --data-path ./fastmri --data-parallel --num-chans 64
```
training the model:
```
python train_model.py --challenge singlecoil --data-path ./fastmri --data-parallel --num-chans 64
```

running the model:
```
python run_model.py --data-path ./fastmri --data-split val  --checkpoint <your checkpoint path> --challenge singlecoil --mask-kspace --out-dir reconstructions_val
```