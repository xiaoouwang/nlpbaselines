import torch


def report_gpu():
    if torch.cuda.is_available():
        print(f"There is/are {torch.cuda.device_count()} gpus.")
        for i in range(torch.cuda.device_count()):
            print(f"the model of card {i} is {torch.cuda.get_device_name(i)}")
            memory = (
                torch.cuda.get_device_properties(0).total_memory // 1024**2 / 1024
            )
            print(f"the memory of card {i} is {memory}GB")
        print("for more info, run nvidia-smi")
    elif torch.backends.mps.is_available():
        print("Mac gpu is available")
        # memory = torch.cuda.get_device_properties(0).total_memory// 1024 ** 2 /1024
        # print(f"the memory of card {i} is {memory}GB")
    else:
        print("no gpu available")
