class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'pascal':
            return '/home/zvision/Databases/Segmentation/VOCdevkit/VOC2012'  # folder that contains VOCdevkit/.
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    @staticmethod
    def models_dir():
        return '.models/'
