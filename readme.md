# Video Transcription using Faster Whisper

* First of all install the latest nvidia cuda toolkit. Go to this website https://developer.nvidia.com/cuda-downloads downlaod and install the toolkit. For more inforkmation reviedw this documentation https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#installing-cuda-development-tools. 

* Verify the installion by run this command in command prompt "nvidia-smi" . The CUDA Version will be 12. Note: Faster Whisper only run on Cuda 12.

* Next download the CuDNN version 8 for CUDA 12.x form this website https://developer.nvidia.com/rdp/cudnn-archive. Follow the instructions given in this documentation https://docs.nvidia.com/deeplearning/cudnn/latest/installation/windows.html to install properly.

* Install Python 3.9 or greater If python is already installed skip this step. 

1. Clone this repository
2. Install the required dependencies "pip install -r requirements.txt"
3. Install ffmpeg. Follow this tutorial https://phoenixnap.com/kb/ffmpeg-windows
4. Select the model size, Tyoe and provide the URL/path and run the program.
