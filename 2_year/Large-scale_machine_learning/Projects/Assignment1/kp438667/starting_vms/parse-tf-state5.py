#!/usr/bin/env python3
import json
import os

hosts = []

with open('terraform.tfstate') as f:
    data = json.load(f)

for res in data["resources"]:
    if res["type"] == "google_compute_instance":
        for inst in res["instances"]:
            attr = inst["attributes"]
            name = attr["name"]
            nic = attr["network_interface"][0]

            private_ip = nic["network_ip"]
            public_ip = nic.get("access_config", [{}])[0].get("nat_ip", None)

            hosts.append((name, public_ip, private_ip))

# ===== ansible hosts =====

with open("hosts", "w") as f:
    f.write("[key_node]\n")
    f.write(hosts[0][1] + "\n\n")

    f.write("[mpi_nodes]\n")
    for _, pub, _ in hosts:
        f.write(pub + "\n")
    f.write("\n")

    f.write("[nfs_server]\n")
    f.write(hosts[0][1] + "\n\n")

    f.write("[nfs_clients]\n")
    for _, pub, _ in hosts[1:]:
        f.write(pub + "\n")
    f.write("\n")

    f.write("[all:vars]\n")
    f.write(f"ansible_ssh_user={os.environ['GCP_userID']}\n")
    f.write(f"ansible_ssh_private_key_file={os.environ['GCP_privateKeyFile']}\n")
    f.write("ansible_ssh_common_args='-o StrictHostKeyChecking=no'\n")
    f.write(f"nfs_server_private_ip={hosts[0][2]}\n")

# ===== hostfile for MPI =====

with open("hostfile_mpi", "w") as f:
    for name, _, _ in hosts:
        f.write(name + "\n")

print("Generated: hosts, hostfile_mpi")
print("NFS server private IP:", hosts[0][2])
