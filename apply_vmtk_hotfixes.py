from os import system, path
from sys import platform


def apply_vmtk_hotfixes(username, anaconda_version="miniconda3", conda_environment="vmtk"):
    """
    A workaround to fix common issues with the currently distributed version of VMTK (1.4) on with Anaconda.
    Python3 bugs occur in the following files: vmtkcenterlines.py, vmtksurfacecurvature.py, and vmtkmeshwriter.py.

    Args:
        username (str): Machine user name
        anaconda_version (str): Distribution of Anaconda, either miniconda or anaconda
        conda_environment (str): Name of conda environment where VMTK is installed
    """
    if platform == "darwin":
        install_path = "/Users/{}/{}3".format(username, anaconda_version)
    elif platform == "linux" or platform == "linux2":
        install_path = "/home/{}/{}3".format(username, anaconda_version)
    elif platform == "win32":
        install_path = "C:\\Users\\{}\\{}3".format(username, anaconda_version.capitalize())

    if platform == "win32":
        vmtk_path = "envs\\{}\\Lib\\site-packages\\vmtk"
    else:
        vmtk_path = "envs/{}/lib/python3.6/site-packages/vmtk"

    centerlines_path = path.join(vmtk_path.format(conda_environment), "vmtkcenterlines.py")
    curvature_path = path.join(vmtk_path.format(conda_environment), "vmtksurfacecurvature.py")
    writer_path = path.join(vmtk_path.format(conda_environment), "vmtkmeshwriter.py")

    path1 = path.join(install_path, centerlines_path)
    path2 = path.join(install_path, curvature_path)
    path3 = path.join(install_path, writer_path)

    print("=== Editing VMTK files located in: {} ".format(
        path.join(install_path, centerlines_path.rsplit("vmtk", 1)[0])))

    system("""sed -i -e 's/len(self.SourcePoints)\/3/len\(self.SourcePoints\)\/\/3/g' {}""".format(path1))
    system("""sed -i -e 's/len(self.TargetPoints)\/3/len\(self.TargetPoints\)\/\/3/g' {}""".format(path1))
    system("""sed -i -e 's/(len(values) - 1)\/2/\(len\(values\) - 1\)\/\/2/g' {}""".format(path2))
    system(
        """sed -i -e "s/file = open(self.OutputFileName,'r')/file = open\(self\.OutputFileName, \'rb\'\)/g" {}""".format(
            path3))
    system(
        """sed -i -e "s/file = open(self.OutputFileName, 'r')/file = open\(self\.OutputFileName, \'rb\'\)/g" {}""".format(
            path3))


if __name__ == "__main__":
    print('Enter your PC/Mac/Linux username:')
    username = input()
    print('Enter Anaconda version: (anaconda, miniconda):')
    anaconda_version = input()
    print('Enter Anaconda environment (default is: vmtk):')
    anaconda_environment = input()
    if anaconda_environment == '':
        anaconda_environment = "vmtk"

    apply_vmtk_hotfixes(username, anaconda_version, anaconda_environment)
