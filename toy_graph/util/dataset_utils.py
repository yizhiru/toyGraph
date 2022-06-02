def read_content_file(content_path):
    """
    读取cora格式数据集content文件，第一列为node-id，第二列至最后第二列为特征数组，最后一列为node-label

    :param content_path: content文件地址
    :return: node-id数组，node-label数组、特征数组
    """
    indices, labels, features = [], [], []
    with open(content_path, 'r') as fr:
        for line in fr.readlines():
            data_split = line.rstrip().split('\t')
            if len(data_split) > 2:
                indices.append(data_split[0])
                labels.append(data_split[-1])
                features.append(data_split[1:-1])
    return indices, labels, features
