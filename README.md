# active-target-localization
Codebase for the paper titled "Active Localization of Multiple Targets from Noisy Relative Measurements" (WAFR 2020)

Check out our [project page](https://ksengin.github.io/active-target-localization/)!

#### Platform and dependencies
* Ubuntu 16.04 
* Python 3.6
* Pytorch 1.4.0
* OpenAI gym 0.17.2 (https://github.com/openai/gym)
* Visdom 0.1.8


### Training

For training a session, run:

```shell
python -m target_localization.train --sess atl --num_targets 2 
```

Add the flag `--image_representation` to use the encoding of the the belief-map image as part of the observation function. Refer to the paper for more details.

### Testing

For evaluating the agent policy, run:

```shell
python -m target_localization.eval --sess atl --num_targets 2 
```
