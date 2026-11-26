"""Class weights and sampler probabilities derived ONLY from training counts."""
import torch

def class_weights(counts,mode='none',cap=5.0):
    counts=torch.as_tensor(counts,dtype=torch.float64)
    if (counts<=0).any():raise ValueError('Each training class needs at least one record')
    if cap<1:raise ValueError('weight_cap must be >=1')
    powers={'none':0.,'inverse_sqrt':.5,'inverse':1.}
    if mode not in powers:raise ValueError('Unknown class weighting mode')
    weights=counts.pow(-powers[mode]);weights/=weights.mean()
    return weights.clamp(max=cap).float()

def sampling_weights(labels,counts,power=1.0):
    if power<0:raise ValueError('sampling_power must be nonnegative')
    counts=torch.as_tensor(counts,dtype=torch.float64)
    if (counts<=0).any():raise ValueError('Missing training class')
    return counts[torch.as_tensor(labels,dtype=torch.long)].pow(-power)

def balance_state(labels,num_classes,config):
    counts=torch.bincount(torch.tensor(labels),minlength=num_classes)
    weights=class_weights(counts,config.class_weighting,config.weight_cap)
    probabilities=sampling_weights(labels,counts,config.sampling_power)
    probabilities/=probabilities.sum()
    per_class=torch.zeros(num_classes,dtype=torch.float64)
    per_class.scatter_add_(0,torch.tensor(labels),probabilities)
    return weights,{
        'training_class_counts':counts.tolist(),'class_weighting':config.class_weighting,
        'weight_cap':config.weight_cap,'class_weights':weights.tolist(),
        'sampling':config.sampling,'sampling_power':config.sampling_power,
        'draws_per_epoch':config.epoch_samples or len(labels),
        'replacement':config.sampling=='weighted',
        'expected_class_probability':per_class.tolist() if config.sampling=='weighted'
            else (counts/counts.sum()).tolist(),
        'note':'Both switches are independent. Enabling both compounds minority emphasis.'}
