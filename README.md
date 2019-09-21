# fastmri-2019-challenge

This my submission to the fastMRI MRI reconstruction challenge as a representative of the Mayo Digial Health Sciences team: https://fastmri.org/submission_guidelines 

We developed a feature loss function using an autoencoder trained on the ground truth images. The loss function was able to improve on all performance benchmarks (NMSE,SSIM,PSNR) used in the contest for all the different models we tested on. We believe that this straightforward methodology is valuable for any researcher working on the challenging MRI construction problem.

As you can see much of the code for data loading/preprocessing comes from the fastMRI library: https://github.com/facebookresearch/fastMRI 

On top of the requirements listed in `requirements.txt`, you will need to download the Fastai library: https://docs.fast.ai/ if you wish to try out the modified U-nets we used as well.

The one-page abstract will be linked here as soon as the contest is over!

You can contact me (Wei-Yin Ko) at ko.weiyin at mayo.edu if you have any questions about this project.

# Examples:

running the model:
