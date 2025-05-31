#!/bin/bash

# Documents for Env
wget 'https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EZMQ_BlSckNJjkw_cXeCQ9wB9128trey9enEd7OjBfwdjg?e=xdyKw8&download=1' -O util_models.tar.gz  && tar -xzvf util_models.tar.gz  && \

# Pretrained models for NoiseAR
wget 'https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/EZAhgsnTKG9Dih1yhhRo1bsBUf4PwhvfbEMdJ06ok6oCSA?e=mWsCiF&download=1' -O pretrained_models.tar.gz && tar -xzvf pretrained_models.tar.gz  && \

# Eval Datasets
wget 'https://hkustconnect-my.sharepoint.com/:u:/g/personal/xliufz_connect_ust_hk/Ec4fdDg9tSRApMN7EaYurgQBT9K90egqDQEpQ5ZO6MkQZg?e=bgpCmT&download=1' -O data.tar.gz  && tar -xzvf data.tar.gz  && \

echo "Download completed successfully."
