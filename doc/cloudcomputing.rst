..
  SPDX-FileCopyrightText: 2020 Maximilian Parzen and Emmanuel Paez

  SPDX-License-Identifier: CC-BY-4.0


.. _cloudcomputing:

###########################
Cloud Computing
###########################

Google Cloud Platform (GCP)
===========================

.. note::
    This set of instructions is partially Windows specific.
    We are happy to take pull requests explaining where the procedures deviate from the descriptions below for other operating systems.
    Likewise, tutorials for other cloud computing solutions are also highly welcome.

The Google Cloud Platform (GCP) is a cloud computing service you can use to run PyPSA-Eur calculations, especially if

- you do not have immediate access to high-performance computating facilities,
- you have problems with the Windows operating system and want a quick run on a linux-based system,
- you want to model whole of Europe in sufficient spatial and temporal resolution,
- you need quick results (trial version includes 32 vCPU cores and up to 800 GB of memory).

With the Google Cloud Platform you set up a virtual machine in the cloud which can store and operate data.
Like on your local computer, you have to install all software and solvers, and create paths on the virtual machine to set up the required environment.
The 300$ free trial budget is offered which equals roughly 10-20 simulations with 180 nodes at hourly basis.
The following steps are required:

- `Google Cloud Platform registration <https://console.cloud.google.com>`_, to receive 300$ free budget.
- `Creating an Virtual Machine (VM) instance <https://www.ibm.com/products/ilog-cplex-optimization-studio>`_, which is practically a virtual computer with Linux as OS.
- `Installation of Cloud SDK <https://cloud.google.com/sdk/>`_, to create a communication channel between your computer and the cloud virtual machine (VM).
- `Installation of WinSCP (Windows) <https://winscp.net/eng/download.php>`_ (or alternative), to handle or transfer files between the VM and you local computer.

Step 1 - Google Cloud Platform registration
-------------------------------------------

First, register at the `Google Cloud Platform <https://console.cloud.google.com>`_ (GCP).
Ann active bank account is required, which will not be charged unless you exceed the trial budget.

Step 2 - Create your Virtual Machine instance
---------------------------------------------

With the following steps we create a Virtual Machine (VM) on Google Cloud.

- Click on the `GCP Dashboard <https://console.cloud.google.com/home/dashboard>`_.
- Click on the "COMPUTE" header, on the "Compute Engine" and then on the "VM instance".
- Click on create.
- Click on new VM instance.

Now a window with the machine details will open. You have to configure the following things:

- Name: Set a name for your VM. Cannot be changed after creation.
- Region: You can keep the default us-central1 (Iowa), since it is a cheap computational region. Sometimes your machine is limited in a specific region. Just pick another region in that case.
- Machine configuration: The machine configuration sets how powerful your VM is.
  You can start with a 1 vCPU and 3.75 GB memory, N1 series machine as every operating second cost money.
  You can edit your machine configuration later. So use a cheap machine type configuration to transfer data and
  only when everything is ready and tested, your expensive machine type, for instance a custom 8 vCPU with 160 GB memory.
  Solvers do not parallelise well, so we recommend not to choose more than 8 vCPU.
  Check ``snakemake -n -j 1 solve_all_networks`` as a dry run to see how much memory is required.
  The memory requirements will vary depending on the spatial and temporal resoulution of your optimisation.
  Example: for an hourly, 181 node full European network, set 8 vCPU and 150 GB memory since the dry-run calculated a 135 GB memory requirement.)
- Boot disk: As default, your VM is created with 10 GB. Depending on how much you want to handle on one VM you should increase the disk size.
  We recommend a disk size of 100 GB for a safe start (cost roughly 8$ per month), the disk can be resized at any later stage with an additional disk.
- Click on create and celebrate your first VM on GCP.

Step 3 - Installation of Cloud SDK
----------------------------------

- Download Google Cloud SDK `SDK <https://cloud.google.com/sdk>`_. Check that you are logged in in your Google account. The link should lead you to the Windows installation of Google Cloud SDK.
- Follow the "Quickstart for Windows - Before you begin" steps.
- After the successfull installation and initialization, close the Google Cloud SDK reopen it again. Type the following command into the "Google Cloud SDK Shell":

    .. code:: bash

        gcloud compute ssh <your VM instance name> -- -L 8888:localhost:8888

- This command above will open a PuTTy command window that is connected to your Virtual Machine. Time to celebrate if it works!
- Now install all necessary tools. As little help, the first steps:
    .. code:: bash

        sudo apt-get update
        sudo apt-get install bzip2 libxml2-dev
        sudo apt-get install wget
        wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
        ls  (to see what anaconda file to bash)
        bash Anaconda3-2020.07-Linux-x86_64.sh
        source ~/.bashrc

- Close and reopen the PuTTy file (-> open Google Cloud SDK -> initialize again with the command above to open the PuTTY command window). Now ``conda`` can be listed with ``conda list``.
  Noq you can follow the standard installation instructions to finalize your machine (don't forget the solvers - for bigger simulations use commercial solvers such as Gurobi).

Step 4 - Installation of WinSCP
-------------------------------

For smooth data exchange between the VM and your local computer you may use WinSCP on Windows.
Make sure that your instance is operating for the next steps.

- Download `WinSCP <https://winscp.net/eng/download.php>`_ and follow the default installation steps.
- Open WinSCP after the installation. A login window will open.
- Keep SFTP as file protocol.
- As host name insert the External IP of your VM (click in your internet browser on your GCP VM instance to see the external IP)
- Set the User name in WinSCP to the name you see in your PuTTy window (check step 3 - for instance [username]@[VM-name]:~$)
- Click on the advanced setting. SSH -> Authentication.
- Option 1. Click on the Tools button and "Install Public Key into Server..". Somewhere in your folder structure must be a public key. I found it with the following folder syntax on my local windows computer -> :\Users\...\.ssh (there should be a PKK file).
- Option 2. Click on the Tools button and "Generate new key pair...". Save the private key at a folder you remember and add it to the "private key file" field in WinSCP. Upload the public key to the metadeta of your instance.
- Click ok and save. Then click Login. If successfull WinSCP will open on the left side your local computer folder structure and on the right side the folder strucutre of your VM. (If you followed Option 2 and its not initially working. Stop your instance, refresh the website, reopen the WinSCP field. Afterwards your your Login should be successfull)

If you had struggle with the above steps, you could also try `this video <https://www.youtube.com/watch?v=lYx1oQkEF0E>`_.

.. note::
    Double check the External IP of your VM before you try to login with WinSCP. It's often a cause for an error.

Step 5 - Extra. Copying your instance with all its data and paths included
--------------------------------------------------------------------------

Especially if you think about operating several instance for quicker simulations, you can create a so called `"image" <https://console.cloud.google.com/compute/images?authuser=1&project=exalted-country-284917>`_ of the virtual machine.
The "image" includes all the data and software set-ups from your VM. Afterwards you can create a VM from an image and avoid all the installation steps above.

Important points when to solve networks in PyPSA
------------------------------------------------

If you use the GCP with the default PyPSA-Eur settings, your budget will be used up very quickly. The following tips should help you to make the most of your budget:

- Always test using low resolution networks; i.e. a single country at 5 nodes and 24h resolution for 2 month of weather data.
- Adjust your solver in the ``config.yaml`` file. Set ``solving: skip_iterations: true``.
  This will lead to a single solver iteration which is often precise enough.
