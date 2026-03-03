variable "project" {
  type = string
}

provider "google" {
  credentials = file("bml-lab.json")
  project     = var.project
  region      = "us-central1"
  zone        = "us-central1-c"
}

resource "google_compute_instance" "vm_instance" {
  count        = 1
  name         = "rf-base-vm-${count.index}"

  machine_type = "e2-medium"

  boot_disk {
    initialize_params {
      image = "ubuntu-minimal-2404-lts-amd64"
      size  = 20
    }
  }

  network_interface {
    # A default network is created for all GCP projects
    network       = "default"
    access_config {
    }
  }
}

// A variable for extracting the external ip of the first instance
output "ip" {
 value = "${google_compute_instance.vm_instance[0].network_interface.0.access_config.0.nat_ip}"
}
