# Parity-Odd-4PCF

Analysis pipeline for probing parity-violation with the 4-point correlation function of BOSS CMASS galaxies. This is described in detail in Philcox (in prep.), and makes use of the [encore](https://github.com/oliverphilcox/encore) NPCF algorithm described in [Philcox et al. 2021](https://arxiv.org/abs/2105.08722) and the Gaussian NPCF covariance matrices from Hou et al. (in prep.).

Note that the main analysis was performed blindly, *i.e.* the analysis pipeline and plotting routines were all created before the data was revealed. This includes the Chern-Simons 4PCF template, computed [here](compute_cs_4pcf.py). None of the routines in the associated [notebook](BOSS%20Odd-Parity%204PCF.ipynb) were modified after unblinding; the only change was that that fake data was replaced by the truth. For posterity, a copy of the paper draft pre-unblinding can be found [here](paper/blinded_draft.pdf). 

### Authors
- [Oliver Philcox](mailto:ohep2@cantab.ac.uk) (Princeton / Institute for Advanced Study)
