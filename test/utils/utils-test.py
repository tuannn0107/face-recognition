from src.utils import utils




def get_file_path_test():
    file_path = utils.get_file_path(__file__)
    print(file_path)


def get_parent_path_test():
    parent_path = utils.get_parent_path(__file__)
    print(parent_path)


def get_root_path_test():
    print(utils.get_root_path())


get_file_path_test()
get_parent_path_test()
get_root_path_test()





