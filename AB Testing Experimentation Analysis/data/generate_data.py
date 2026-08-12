"""
Synthetic data generator for the A/B testing project.

A synthetic experiment is generated simulating a checkout page redesign
test on an e-commerce site. Users are randomly assigned to a control or
treatment group, with a modest, realistic lift in conversion rate and
average order value built into the treatment group, and device type
included as a segment that affects baseline conversion but not the
treatment effect. A fixed random seed is used so the dataset is
reproducible.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 123
rng = np.random.default_rng(RANDOM_SEED)

N_USERS = 30000

DEVICES = ["Desktop", "Mobile", "Tablet"]
DEVICE_WEIGHTS = [0.42, 0.48, 0.10]

# Baseline (control) conversion rate by device, and the treatment lift applied
# uniformly on top of the device baseline
BASE_CONVERSION = {"Desktop": 0.130, "Mobile": 0.095, "Tablet": 0.110}
TREATMENT_LIFT = 0.013  # absolute percentage point lift applied in the treatment group

ORDER_VALUE_MEAN_CONTROL = 42.00
ORDER_VALUE_MEAN_TREATMENT = 44.50
ORDER_VALUE_SIGMA = 0.35  # lognormal shape parameter, kept equal across groups

user_id = np.arange(1, N_USERS + 1)
group = rng.choice(["control", "treatment"], size=N_USERS, p=[0.5, 0.5])
device = rng.choice(DEVICES, size=N_USERS, p=DEVICE_WEIGHTS)

conversion_prob = np.array([
    BASE_CONVERSION[d] + (TREATMENT_LIFT if g == "treatment" else 0.0)
    for d, g in zip(device, group)
])
converted = rng.binomial(1, conversion_prob)

order_value = np.full(N_USERS, np.nan)
converted_mask = converted == 1
n_converted = converted_mask.sum()

mean_for_converted = np.where(
    group[converted_mask] == "treatment", ORDER_VALUE_MEAN_TREATMENT, ORDER_VALUE_MEAN_CONTROL
)
mu = np.log(mean_for_converted) - (ORDER_VALUE_SIGMA ** 2) / 2
order_value[converted_mask] = rng.lognormal(mean=mu, sigma=ORDER_VALUE_SIGMA)

df = pd.DataFrame({
    "user_id": user_id,
    "group": group,
    "device": device,
    "converted": converted,
    "order_value": np.round(order_value, 2),
})

df.to_csv("checkout_ab_test.csv", index=False)
print(f"Generated {len(df):,} user records")
print(df.groupby("group")["converted"].agg(["count", "mean"]).round(4))
