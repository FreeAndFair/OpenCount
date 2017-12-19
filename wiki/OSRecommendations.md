# Summary #

OpenCount has to date been tested only under Ubuntu Linux.
Therefore, we recommend installing Ubuntu Linux.
There are two options: you can install Ubuntu directly on your
machine (perhaps dual-boot, with another OS), or you can run Ubuntu
inside a virtual machine (VM).

# Bare Metal Installs #

You can install Ubuntu Linux on your machine.
If your machine has more than 4 cores, use a 64-bit
flavor of Ubuntu (desktop-amd64);
the 32-bit versions of Ubuntu only support up to 4 cores.
Install the Desktop Edition of Ubuntu
(not a Server Edition), to ensure you get the graphical user interface packages.

# Virtual Machine Installs #

Install Ubuntu Linux.
We recommend the Desktop Edition.

# Filesystem #

If you will be using OpenCount repeatedly for multiple
large elections,
when partitioning your hard drive and creating the filesystem,
you may want to manually tweak the configuration of the filesystem
to increase the maximum number of inodes.
OpenCount creates many small files, which after extended use
can exceed the number of inodes available on default filesystem configurations.
For instance, if creating the filesystem with `mke2fs` or `mkfs`,
you can use the `-i 4096` option.