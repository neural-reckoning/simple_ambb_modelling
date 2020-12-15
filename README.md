# General neural mechanismscan account for rising slope preference in localization of ambiguous sounds 

This repository is an interactive version of the paper *General neural mechanismscan account for rising slope preference in localization of ambiguous sounds*
by Jean-Hugues Lestang and Dan Goodman. We show that we can re-use standard neural mechanisms to account for the data
on revereberant sound localisation of [Dietz et al., 2013](https://www.pnas.org/content/110/37/15151).

To run this code interactively online, click the launch binder link below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neural-reckoning/simple_ambb_modelling/master?filepath=index.ipynb)

## Abstract

Sound localization in reverberant environments is a difficult task that human listeners perform effortlessly. Many neural mechanisms have been proposed to account for this behavior. Generally they rely on emphasizing localization information at the onset of the incoming sound while discarding localization cues that arrive later. We modelled several of these mechanisms using neural circuits commonly found in the brain and tested their performance in the context of experiments showing that, in the dominant frequency region for sound localisation, we have a preference for auditory cues arriving during the rising slope of the sound energy [(Dietz et al. 2013)](https://www.pnas.org/content/110/37/15151). We found that both single cell mechanisms (onset and adaptation) and population mechanisms (lateral inhibition) were easily able to reproduce the results across a very wide range of parameter settings. This suggests that sound localization in reverberant environments may not require specialised mechanisms specific to that task, but may instead rely on common neural circuits in the brain. This is in line with the theory that the brain consists of functionally overlapping general purpose mechanisms rather than a collection of mechanisms each highly specialised to specific tasks. This research is fully reproducible, and we made our code available to edit and run online via interactive live notebooks. 

![Basic mechanisms of the population model](fig_basic_mech.png?raw=true "Basic mechanisms of the population model")
