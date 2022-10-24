# Parity-Odd-4PCF

Analysis pipeline for probing parity-violation with the 4-point correlation function of BOSS CMASS galaxies. This is described in detail in [Philcox 2022](https://arxiv.org/abs/2206.04227), and makes use of the [encore](https://github.com/oliverphilcox/encore) NPCF algorithm described in [Philcox et al. 2021](https://arxiv.org/abs/2105.08722) and the Gaussian NPCF covariance matrices from [Hou et al. 2021](https://arxiv.org/abs/2108.01714).

Note that the main analysis was performed blindly, *i.e.* the analysis pipeline and plotting routines were all created before the data was revealed. This includes the Chern-Simons 4PCF template, computed [here](compute_cs_4pcf.py). The main routines in the associated [notebook](BOSS%20Odd-Parity%204PCF%20(CS%20template).ipynb) were not modified after unblinding, except for replacing fake data with the truth and implementing various cosmetic improvements. We additionally include a [notebook](Nseries%20Odd-Parity%204PCF.ipynb) containing the analysis of Nseries mock catalogs. For posterity, a copy of the paper draft pre-unblinding can be found [here](paper/blinded_draft.pdf). 

The BOSS data can also be used to constrain inflationary models of parity-violation, as discussed in [Cabass, Ivanov, \& Philcox (2022)](IN-PREP). To this end, we provide code implementing the inflationary templates, and notebooks computing the corresponding amplitude constraints. The analysis notebooks can be found [here](BOSS%20Odd-Parity%204PCF%20(ghost%20template).ipynb) and [here](BOSS%20Odd-Parity%204PCF%20(collider%20template).ipynb), with the models provided in the (templates)[templates/] directory.

To run the main analysis notebook, the simulation data will be required. Almost all of this is contained in the ```data/``` directories, except for two large files ```all_patchy2048_fourpcf.npz``` and ```all_nseries-patchy2048_fourpcf.npz```. These can be downloaded from Dropbox ([file 1](https://www.dropbox.com/s/594iol702s7gk86/all_patchy2048_fourpcf.npz?dl=0) and [file 2](https://www.dropbox.com/s/r5ezfez15ou93ws/all_nseries-patchy2048_fourpcf.npz?dl=0)) and should be placed in the ```data/``` directory.

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Princeton / Institute for Advanced Study / Columbia / Simons Foundation)
