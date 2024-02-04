import torch
import torchvision



def test_timeembedding(): 
    from models_dis import timestep_embedding, DisModel
    times_steps = torch.randint(1, 100, (1,))
    print(timestep_embedding(times_steps, 1000)) 


def test_dismodel():
    from models_dis import timestep_embedding, DisModel
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



def test_imagenet1k(): 
    data_path = '/TrainData/Multimodal/public/datasets/ImageNet/train' 
    import torchvision.datasets as datasets
    dataset_train = datasets.ImageFolder(data_path) 
    print(dataset_train[0])



def test_fid_score(): 
    from tools.fid_score import calculate_fid_given_paths 
    path1 = '/TrainData/Multimodal/zhengcong.fei/dis/results/cond_cifar10_small/his'
    path2 = '/TrainData/Multimodal/zhengcong.fei/dis/results/uncond_cifar10_small/his'
    fid = calculate_fid_given_paths((path1, path2))

# test_dismodel() 
# test_cifar10()
# test_imagenet1k()
test_fid_score()
