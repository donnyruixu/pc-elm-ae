import torch
import numpy as np
from models import operations

def FPS(point_cloud, m):
    '''
    :param point_cloud: torch tensor of shape (batch_size, num_point, dims)
    :param m: The num of central points
    :return: torch tensor of shape (batch_size, m). each row indicates the
             m indices of the central points
    '''
    batch_size, num_point, dims = point_cloud.size()

    adjacent_mat = operations.pairwise_distance(point_cloud)

    start_point_index = np.random.randint(0, num_point, size=batch_size)
    adjacent_mat[np.arange(batch_size), :, start_point_index] = 0.0
    farthest_point_index = torch.from_numpy(start_point_index.reshape((batch_size, 1)))
    start_point_distance = adjacent_mat[np.arange(batch_size), start_point_index, :]
    min_distance = start_point_distance

    second_point_index = torch.argmax(min_distance, dim=-1)
    adjacent_mat[np.arange(batch_size), :, second_point_index] = 0.0
    farthest_point_index = torch.cat((farthest_point_index, second_point_index.reshape((batch_size,1))), dim=1)

    old_index = second_point_index
    for i in range(m-2):
        old_distance = adjacent_mat[np.arange(batch_size), old_index, :]
        min_distance = torch.min(min_distance, old_distance)
        new_index = torch.argmax(min_distance, dim=-1)
        adjacent_mat[np.arange(batch_size), :, new_index] = 0.0
        farthest_point_index = torch.cat((farthest_point_index, new_index.reshape((batch_size,1))), dim=1)
        old_index = new_index

    return farthest_point_index

if __name__ == '__main__':
    point_cloud = np.random.uniform(size=(1, 1024, 3))
    farthest_point_index = FPS(torch.from_numpy(point_cloud), 512)
    print(farthest_point_index)
    farthest_point_index = np.squeeze(farthest_point_index.numpy()).tolist()
    point_index_set = set(farthest_point_index)
    print(len(point_index_set))

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    color_list = ['g' for i in range(1024)]
    color_arr = np.array(color_list)
    color_arr[np.array(farthest_point_index)] = 'r'

    x = np.squeeze(point_cloud)[:,0]
    y = np.squeeze(point_cloud)[:,1]
    z = np.squeeze(point_cloud)[:,2]
    ax.scatter(x, y, z, c=color_arr)
    plt.show()