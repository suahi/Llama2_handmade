import subprocess

# Install modelscope
print("Installing modelscope...")
subprocess.run(['pip', 'install', 'modelscope'], check=True)
print("modelscope installed successfully.")

# Install huggingface_hub for huggingface-cli
print("Installing huggingface_hub...")
subprocess.run(['pip', 'install', 'huggingface_hub'], check=True)
print("huggingface_hub installed successfully.")