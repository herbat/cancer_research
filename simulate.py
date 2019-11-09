import numpy as np

GENES    = 100
PROFILES = 3
SAMPLES  = 1


def gen_profiles(num_profiles):
    profiles = []
    for p in range(num_profiles):
        profiles.append(np.random.rand(GENES))
    return profiles


def gen_samples(profiles, num_samples):
    samples = []
    for s in range(num_samples):
        proportions = np.random.rand(len(profiles))
        res = np.zeros(GENES)
        for i, p in enumerate(profiles):
            res += proportions[i] * p
        samples.append(res)
    return samples


data = gen_samples(gen_profiles(PROFILES), SAMPLES)
print(data)
