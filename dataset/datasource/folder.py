from joblib import Memory
from PIL import Image
from torchvision import datasets

__all__ = ["CachedImageFolder", "read_pil_raw_image"]


def read_pil_raw_image(path):
    """
    independent function which is useful for function caching
    Please noticing that this function can not be placed in the CachedImageFolder class because the hash
    """
    img = Image.open(path).convert("RGB")
    return img


class CachedImageFolder:
    def __init__(self, root, cache_dir="/mnt/host0/tmp/", transform=None, target_transform=None):
        self.root = root
        self.cache_dir = cache_dir
        self.transform = transform
        self.target_transform = target_transform

        self.load_raw_image = Memory(location=cache_dir, verbose=0).cache(read_pil_raw_image)
        self.dataset = datasets.ImageFolder(root)

    def __getitem__(self, index):
        path, target = self.dataset.samples[index]
        image = self.load_raw_image(path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":

    train_dataset = CachedImageFolder(
        root="./data/imagenet/ILSVRC2012_img_train", exp_name="imagenet_exp_testtime", transform=None
    )

    train_dataset.__getitem__(0)
    from IPython import embed

    embed()
