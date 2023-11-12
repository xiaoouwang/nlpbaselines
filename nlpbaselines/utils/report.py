import torch

def report_gpu():
    if torch.cuda.is_available():
        print(f"There is/are {torch.cuda.device_count()} gpus.")
        for i in range(torch.cuda.device_count()):
            print(f"the model of card {i} is {torch.cuda.get_device_name(i)}")
        print("for more info, run nvidia-smi")
    if torch.backends.mps.is_available():
        print("Mac gpu is available")
    else:
        print("no gpu available")
