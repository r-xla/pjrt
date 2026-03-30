import modal

app = modal.App("cuda-r-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-base-ubuntu24.04")
    .apt_install(
        "software-properties-common", "apt-utils", "tzdata", "locales",
        "libpng-dev", "pkg-config", "wget", "curl", "unzip",
        "libprotobuf-dev", "protobuf-compiler",
        "libopenblas-dev", "gfortran",
        "libcurl4-openssl-dev", "libssl-dev", "libxml2-dev",
        "sudo",
    )
    .run_commands(
        "wget -qO- https://cloud.r-project.org/bin/linux/ubuntu/marutter_pubkey.asc | tee /etc/apt/trusted.gpg.d/cran_ubuntu_key.asc",
        "add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu noble-cran40/'",
        "apt-get update -y",
        "apt-get install -y r-base r-base-dev",
        'Rscript -e \'install.packages("pak", repos = sprintf("https://r-lib.github.io/p/pak/stable/%s/%s/%s", .Platform$pkgType, R.Version()$os, R.Version()$arch))\'',
        'Rscript -e \'pak::pkg_install("mlverse/cudatoolkit/cuda12.8")\'',
    )
    .add_local_dir(".", "/root/pjrt", copy=True)
    .run_commands(
        'cd /root/pjrt && Rscript -e \'pak::local_install()\'',
    )
)


@app.function(image=image, gpu="any", timeout=3600)
def shell():
    """Use with: uvx modal shell tools/test_gpu.py::shell"""
    pass
