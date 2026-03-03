#!/usr/bin/env python3
import json, os

IPs = []        # public
private_IPs = []  # private

with open('terraform.tfstate') as f:
    data = json.load(f)

for res in data.get('resources', []):
    if res.get('type') == 'google_compute_instance':
        for vm in res.get('instances', []):
            attrs = vm['attributes']
            nic = attrs['network_interface'][0]

            private_IPs.append(nic['network_ip'])
            public_ip = nic.get('access_config', [{}])[0].get('nat_ip')
            if public_ip:
                IPs.append(public_ip)

print("Public IPs:", IPs)
print("Private IPs:", private_IPs)

# ========== ANSIBLE HOSTS (PUBLIC IP) ==========
with open('hosts', 'w') as f:
    f.write('[key_node]\n' + IPs[0] + '\n\n')

    f.write('[mpi_nodes]\n')
    for ip in IPs:
        f.write(ip + '\n')
    f.write('\n')

    f.write('[nfs_server]\n' + IPs[0] + '\n\n')

    f.write('[nfs_clients]\n')
    for ip in IPs[1:]:
        f.write(ip + '\n')
    f.write('\n')

    f.write('[all:vars]\n')
    f.write(f"ansible_ssh_user={os.environ['GCP_userID']}\n")
    f.write(f"ansible_ssh_private_key_file={os.environ['GCP_privateKeyFile']}\n")
    f.write("ansible_ssh_common_args='-o StrictHostKeyChecking=no'\n")
    f.write(f"nfs_server={private_IPs[0]}\n")

# ========== MPI HOSTFILE (PRIVATE IP) ==========
with open('hostfile_mpi', 'w') as f:
    for ip in private_IPs:
        f.write(ip + '\n')
