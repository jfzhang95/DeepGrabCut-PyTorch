import torch.utils.data as data


class CombineDBs(data.Dataset):
    def __init__(self, dataloaders, excluded=None):
        self.dataloaders = dataloaders
        self.excluded = excluded
        self.im_ids = []

        # Combine object lists
        for dl in dataloaders:
            for elem in dl.im_ids:
                if elem not in self.im_ids:
                    self.im_ids.append(elem)

        # Exclude
        if excluded:
            for dl in excluded:
                for elem in dl.im_ids:
                    if elem in self.im_ids:
                        self.im_ids.remove(elem)

        # Get object pointers
        self.obj_list = []
        self.im_list = []
        new_im_ids = []
        obj_counter = 0
        num_images = 0
        for ii, dl in enumerate(dataloaders):
            for jj, curr_im_id in enumerate(dl.im_ids):
                if (curr_im_id in self.im_ids) and (curr_im_id not in new_im_ids):
                    flag = False
                    new_im_ids.append(curr_im_id)
                    for kk in range(len(dl.obj_dict[curr_im_id])):
                        if dl.obj_dict[curr_im_id][kk] != -1:
                            self.obj_list.append({'db_ii': ii, 'obj_ii': dl.obj_list.index([jj, kk])})
                            flag = True
                        obj_counter += 1
                    self.im_list.append({'db_ii': ii, 'im_ii': jj})
                    if flag:
                        num_images += 1

        self.im_ids = new_im_ids
        print('Combined number of images: {:d}\nCombined number of objects: {:d}'.format(num_images, len(self.obj_list)))

    def __getitem__(self, index):

        _db_ii = self.obj_list[index]["db_ii"]
        _obj_ii = self.obj_list[index]['obj_ii']
        sample = self.dataloaders[_db_ii].__getitem__(_obj_ii)

        if 'meta' in sample.keys():
            sample['meta']['db'] = str(self.dataloaders[_db_ii])

        return sample

    def __len__(self):
        return len(self.obj_list)

    def __str__(self):
        include_db = [str(db) for db in self.dataloaders]
        exclude_db = [str(db) for db in self.excluded]
        return 'Included datasets:'+str(include_db)+'\n'+'Excluded datasets:'+str(exclude_db)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloaders import pascal
    from dataloaders import sbd
    import torch
    import numpy as np
    import dataset.custom_transforms as tr
    from torchvision import transforms

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.15, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])

    pascal_voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts, retname=True)
    sbd = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr, retname=True)
    pascal_voc_train = pascal.VOCSegmentation(split='train', transform=composed_transforms_tr, retname=True)

    dataset = CombineDBs([pascal_voc_train, sbd], excluded=[pascal_voc_val])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

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