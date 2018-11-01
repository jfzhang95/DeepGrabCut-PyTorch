import os
import numpy as np
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from mypath import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class COCOSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=Path.db_root_dir('coco'),
                 split='train',
                 year='2014',
                 transform=None,
                ):
        """
        :param base_dir: path to COCO dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'images', '{}{}'.format(split, year))
        self._annot_dir = os.path.join(self._base_dir, 'annotations', 'instances_{}{}.json'.format(split, year))

        self.year = year
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.transform = transform


        self.im_ids = []
        self.image_info = []
        self.objects = []

        coco = COCO(self._annot_dir)
        class_ids = sorted(coco.getCatIds())

        for i in class_ids:
            self.im_ids.extend(list(coco.getImgIds(catIds=[i])))
        # Remove duplicates
        self.im_ids = list(set(self.im_ids))

        for i in self.im_ids:
            image_info = {
                "id": i,
                "path": os.path.join(self._image_dir, coco.imgs[i]['file_name']),
                "width": coco.imgs[i]["width"],
                "height": coco.imgs[i]["height"],
                "annotations": coco.loadAnns(coco.getAnnIds(
                imgIds=[i], catIds=class_ids, iscrowd=None))
            }
            self.image_info.append(image_info)

        for image_info in self.image_info:
            path = image_info['path']
            width = image_info['width']
            height = image_info['height']
            annotations = image_info['annotations']
            for annotation in annotations:
                objects = {
                    'image_path': path,
                    'mask_annotation': annotation,
                    'height': height,
                    'width': width
                }
                self.objects.append(objects)

        # Display stats
        print('Number of images: {:d}\nNumber of objects: {:d}'.format(len(self.image_info),
                                                                        len(self.objects)))

    def __len__(self):
        return len(self.objects)

    def _make_img_gt_point_pair(self, index):
        object = self.objects[index]
        image_path = object['image_path']
        annotation = object['mask_annotation']
        height = object['height']
        width = object['width']

        _target = self.annToMask(annotation, height, width)

        if annotation['iscrowd']:
            # For crowd masks, annToMask() sometimes returns a mask
            # smaller than the given dimensions. If so, resize it.
            if _target.shape[0] != height or _target.shape[1] != width:
                _target = np.ones([height, width], dtype=bool)

        # Read Image
        _img = np.array(Image.open(image_path).convert('RGB')).astype(np.float32)

        return _img, _target

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



    def __str__(self):
        return 'COCO' + str(self.year) + '(split=' + str(self.split)


if __name__ == '__main__':
    from dataloaders import custom_transforms as tr
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    voc_train = COCOSegmentation(split='val', transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=4)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            dismap = sample['distance_map'][jj].numpy()
            gt = sample['gt'][jj].numpy()
            gt[gt > 0] = 255
            gt = np.array(gt[0]).astype(np.uint8)
            dismap = np.array(dismap[0]).astype(np.uint8)
            display = 0.9 * gt + 0.4 * dismap
            display = display.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.imshow(display, cmap='gray')

        if ii == 1:
            break
            
    plt.show(block=True)