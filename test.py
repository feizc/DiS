import torch
import torchvision

from models_dis import timestep_embedding, DisModel


def test_timeembedding(): 
    times_steps = torch.randint(1, 100, (1,))
    print(timestep_embedding(times_steps, 1000)) 


def test_dismodel():
    model = DisModel().cuda()
    input_image = torch.randn(1, 3, 224, 224).cuda()
    times_steps = torch.randint(1, 100, (1,)).cuda()
    out = model(x=input_image, timesteps=times_steps)
    print(out.size())


def test_cifar10(): 
    data_path = "/TrainData/Multimodal/zhengcong.fei/dis/data"
    cifar10 = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=False
    )
    cifar10_test = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=False
    )
    print(cifar10)
    print(cifar10_test[0])

test_dismodel() 
#test_cifar10()
