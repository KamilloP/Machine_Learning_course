# Report for experiments

## General setup
- The experiments were conducted using 5, 10, 15, and 20 virtual machines (VMs). Both weak scaling and strong scaling were evaluated.
- Strong scaling with 20 VMs corresponds to the same experiment as weak scaling with 20 VMs.
- Infrastructure management and inter-node communication were implemented using Terraform and Ansible. For convenience, an NFS shared filesystem was deployed. The general setup was similar to Laboratory 2, but with several important differences.
- A custom VM image was first created, based on `ubuntu-minimal-2404-lts-amd64` with 20 GB of disk space. All required dependencies were preinstalled on this image, including Python 3.12, which is natively supported by this operating system. This approach eliminated the need to install packages via Ansible on every VM.
- The final cluster consisted of VMs deployed across multiple zones: `us-west1-b`, `us-east5-b`, `us-east1-b`, and `us-central1-b`.
- Due to DNS resolution issues between zones, the configuration files `nfs.yml` and `parse-tf-state.py` were modified to use internal IP addresses instead of hostnames.
- The random seed used during training was 42. The number of trees in each Random Forest model was set to 5.

## Training run time

Training time depending on the number of virtual machines and scaling mode.

\begin{table}[h!]
\centering
\caption{Training runtime depending of weak/strong scale and number of VM machines}
\label{tab:times}
\begin{tabular}{|c|c|c|}
\hline
\textbf{Number of VM/Scale} & \textbf{Weak} & \textbf{Strong} \\
\hline
5  & 1m44s (104s) & 19m3s (1143s) \\
10 & 1m43s (103s) & 7m8s (428s) \\
15 & 2m42s (162) & 2m51s (171s) \\
20 & 2m30s (150s) & 2m30s (150s) \\
\hline
\end{tabular}
\end{table}


## Conclusions
The weak scaling results show a non-trivial dependence on the number of VMs. Although each VM processes a shard of the same size, increasing the number of VMs leads to higher communication overhead, particularly during the aggregation and vocabulary-building phases. As a result, the total runtime does not remain constant and even slightly increases. Based on the observed trend, it is expected that experiments with 50 VMs would take approximately 3 minutes, and with 100 VMs around 3 minutes and 30 seconds.

Strong scaling behavior is clearly visible in the results. The runtime decreases approximately inversely with the number of VMs, which is consistent with the expectation that the workload per worker is reduced. However, some parts of the algorithm (notably vocabulary construction and synchronization) require global communication, which imposes a lower bound on the achievable runtime. Additionally, NFS communication overhead becomes more significant as the number of VMs increases. The large difference between 10 and 15 VMs and the much smaller difference between 15 and 20 VMs suggest that the system approaches this lower bound. Therefore, it is estimated that with 50 VMs the runtime would be close to 2 minutes, and with 100 VMs approximately 1 minute and 30 seconds.

## Appendix

- The `models` directory contains all trained models.

- The dataset was split using `split.py` (no permutation, seed = 42).

- Tests are provided in `test.py` and the tests directory (e.g., `animal_human`).

- The `results` directory contains runtime screenshots and some classification outputs.
Results of prediction of classification experiment defined in `results\classify_strong_5_and_10.png` are in `results/results` folder. Note that results are almost only `6` class id for each observation. Models are actually different trees, but scheme of random forest fails in this setup - usually all words in trees are absent in observation, so the most common class has been set. Test `animanl_human` is better for evaluation.

- VM provisioning and configuration files are included in `starting_vms`.
To create image I have created one VM first, based on `single.tf` file. I have set python virtualenv in `~/` folder and added command `source ~/venv/bin/activate` to `~/.bashrc` file. After that I have saved image, destroyed this VM and created other VMs. I have used `nfs5.yml` / `nfs6.yml` and `parse-tf-state5.py` / `parse-tf-state6.py` files respectively for this purpose (I had problems with NFS for other zones, so I tried multiple time/debug on older version; version 6 should work).  
