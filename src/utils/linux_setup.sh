apt update 
apt install -y openslide-tools
wget https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.1/ASAP-2.1-py38-Ubuntu2004.deb
apt install libdcmtk14
apt install libopenslide0
ls
dpkg -i ASAP-2.1-py38-Ubuntu2004.deb