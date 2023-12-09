import torch
import torchmetrics

from torchmetrics.image.fid import FrechetInceptionDistance

def example_fid1():
    _ = torch.manual_seed(123)

    fid = FrechetInceptionDistance(feature=64)

    # generate two slightly overlapping image intensity distributions
    imgs_dist1 = torch.randint(0, 200, (2, 3, 10, 10), dtype=torch.uint8)
    imgs_dist2 = torch.randint(100, 255, (2, 3, 10, 10), dtype=torch.uint8)
    fid.update(imgs_dist1, real=True)
    fid.update(imgs_dist2, real=False)
    fid_result = fid.compute()
    
    print("deepl")
    
    print(fid_result)


def example_fid2():
    _ = torch.manual_seed(123)

    fid = FrechetInceptionDistance(feature=64)

    # generate two slightly overlapping image intensity distributions
    imgs_dist1 = torch.randint(0, 200, (2, 1, 10, 10), dtype=torch.uint8)
    imgs_dist2 = torch.randint(100, 255, (2, 1, 10, 10), dtype=torch.uint8)
    
    imgs_dist1_3chan = imgs_dist1.repeat(1, 3, 1, 1)
    imgs_dist2_3chan = imgs_dist2.repeat(1, 3, 1, 1)
    
    
    fid.update(imgs_dist1_3chan, real=True)
    fid.update(imgs_dist2_3chan, real=False)
    fid_result = fid.compute()
    
    print(fid_result)
    


# TODO: 3채널 이미지가 들어오는 것을 전제로 하는데, 1채널 이미지를 평가할 순 없나?
def compute_fid():
    fid = FrechetInceptionDistance(feature=64, normalize=True) 


def main():
    example_fid1()
    example_fid2()
    

if __name__ == '__main__':
    main()