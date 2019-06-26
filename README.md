# Generic neural circuits account for reverberant sound localization

This repository is an interactive version of the paper *Generic neural circuits account for reverberant sound localization*
by Jean-Hugues Lestang and Dan Goodman. We show that we can re-use standard neural mechanisms to account for the data
on revereberant sound localisation of [Dietz et al., 2013](https://www.pnas.org/content/110/37/15151).

To run this code interactively online, click the launch binder link below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/neural-reckoning/simple_ambb_modelling/master)

## Abstract

Sound localization in the presence of reverberations is a difficult task that human listeners perform effortlessly.
Many neural mechanisms have been proposed to account for this behavior. Generally they rely on emphasizing localization
information at the onset of the incoming sound while discarding later arriving localization cues. We modelled several of
these mechanisms using neural circuits commonly found in the brain and tested their performance in the context of the
experiment described in [Dietz et al. (2013)](https://www.pnas.org/content/110/37/15151). We found that both single cell
mechanisms (onset response and adaptation) and population mechanisms (mutual inhibition) were able to reproduce the
results across a very wide range of parameter settings. This suggests that sound localization in reverberant
environments may not require specialised mechanisms specific to that task, but may instead be the result of
compositions of common neural circuits in the brain. This is in line with the theory that the brain consists of
functionally overlapping general purpose mechanisms rather than a collection of mechanisms each highly specialised to
specific tasks. Additionally, interactive live code notebooks comprehensively covering the results described in this
paper were made accessible online.

![Basic mechanisms of the population model](fig_basic_mech.png?raw=true "Basic mechanisms of the population model")
