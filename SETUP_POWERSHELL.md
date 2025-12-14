# PowerShell Setup Commands

This document contains the essential PowerShell commands for setting up and using the White-Line Violation Detection System.

## Command 1: Enable PowerShell Script Execution
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
**Purpose**: Allows PowerShell scripts (like the virtual environment activation script) to run on your system.

## Command 2: Activate Virtual Environment
```powershell
.\white-line\Scripts\Activate.ps1
```
**Purpose**: Activates the Python virtual environment named `white-line` so you can use the installed packages.

## Command 3: Install/Update Dependencies
```powershell
pip install -r requirements.txt
```
**Purpose**: Installs all required Python packages listed in `requirements.txt` (ultralytics, torch, opencv-python, numpy, matplotlib).

## Command 4: Verify PyTorch and CUDA Setup
```powershell
python -c "import torch; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); if torch.cuda.is_available(): print('CUDA Device:', torch.cuda.get_device_name(0))"
```
**Purpose**: Verifies that PyTorch is installed correctly and CUDA is available for GPU acceleration.

## Command 5: Run the White-Line Violation Detection
```powershell
python white_line_violation.py --input test\test1.jpg --output outputs\images\result.jpg
```
**Purpose**: Processes an input image to detect white-line violations and saves the result to the output directory.

## Additional Useful Commands

### Process a Video File
```powershell
python white_line_violation.py --input path\to\video.mp4 --output outputs\videos\result.mp4 --video
```

### Run with Custom Settings
```powershell
python white_line_violation.py --input test\test1.jpg --output outputs\images\result.jpg --conf 0.5 --overlap 0.4 --device cuda
```

### Deactivate Virtual Environment
```powershell
deactivate
```

