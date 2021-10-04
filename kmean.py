import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Cluster:
    def __init__(self, centroids_x, centroids_y):
        self.points_x = []
        self.points_y = []
        self.centroids_x = centroids_x
        self.centroids_y = centroids_y

    def __init__(self, centroids_x, centroids_y, points_x, points_y):
        self.points_x = points_x
        self.points_y = points_y
        self.centroids_x = centroids_x
        self.centroids_y = centroids_y

    def add_point(self, x, y):
        self.points_x.append(x)
        self.points_y.append(y)

    def mean(self):
        self.centroids_x = sum(self.points_x) / len(self.points_x)
        self.centroids_y = sum(self.points_y) / len(self.points_y)

    def clear_points(self):
        self.points_x = []
        self.points_y = []


def show_picture(clusters):
    color = ['r', 'b', 'g', 'c', 'm']
    i = 0
    for cl in clusters:
        plt.scatter(cl.centroids_x, cl.centroids_y, color=color[i], marker='x')
        plt.scatter(cl.points_x, cl.points_y, color=color[i])
        for x in cl.points_x:
            print(x)
        i += 1
    plt.show()
    print("_")


def map_to_array(clusters):
    x_c = []
    y_c = []
    for cl in clusters:
        x_c.append(cl.centroids_x)
        y_c.append(cl.centroids_y)
    return [x_c, y_c]


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def random_points(n, filename):
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    pd.DataFrame([x, y]).to_csv(filename)


def generate_points_file(points_count, filename):
    x = np.random.randint(0, 100, points_count)
    y = np.random.randint(0, 100, points_count)
    with open(filename, 'w') as file:
        file.write('x,y\n')
        for item_x, item_y in zip(x, y):
            file.write(f'{item_x},{item_y}\n')


def read_csv(filename):
    return pd.read_csv(filename)


def centroids(points, k):
    x_centr = points['x'].mean()
    y_centr = points['y'].mean()
    R = dist(x_centr, y_centr, points['x'][0], points['y'][0])
    for i in range(len(points)):
        R = max(R, dist(x_centr, y_centr, points['x'][i], points['y'][i]))
    x_c, y_c = [], []
    for i in range(k):
        x_c.append(x_centr + R * np.cos(2 * np.pi * i / k))
        y_c.append(y_centr + R * np.sin(2 * np.pi * i / k))
    return [x_c, y_c]


def nearest_centroid(points, centroids):
    clusters = [Cluster(x_c, y_c, [], []) for x_c, y_c in zip(centroids[0], centroids[1])]
    indx = -1
    for x, y in zip(points['x'], points['y']):
        r = float('inf')
        for i, cl in enumerate(clusters):
            if r > dist(x, y, cl.centroids_x, cl.centroids_y):
                r = dist(x, y, cl.centroids_x, cl.centroids_y)
                indx = i
        if indx > 0:
            clusters[indx].add_point(x, y)
    return clusters


def recalculate_centroid(clusters):
    new_clusters = []
    for cl in clusters:
        new_clusters.append(Cluster(cl.centroids_x, cl.centroids_y, cl.points_x, cl.points_y))
    for cl in new_clusters:
        if len(cl.points_x) != 0:
            cl.mean()
    return new_clusters


def centroid_not_equals(old_cluster, new_cluster):
    if len(new_cluster) <= 0 or len(old_cluster) <= 0:
        return True
    for i in range(0, len(new_cluster) - 1):
        if old_cluster[i].centroids_x != new_cluster[i].centroids_x \
                and old_cluster[i].centroids_y != new_cluster[i].centroids_y:
            return True
    return False
    # print("tyt")
    # for old_cl, new_cl in zip(old_cluster, new_cluster):
    #     print("ty2")
    #     for o_x, o_y in zip(old_cl.points_x, old_cl.points_y):
    #         print("ty3")
    #         isNotExist = True
    #         for n_x, n_y in zip(new_cl.points_x, new_cl.points_y):
    #             if o_x == n_x and o_y == n_y:
    #                 isNotExist = False
    #                 break
    #         if (isNotExist):
    #             return True
    #
    # print("F")
    # return False

def c_l(clusters):
    r = []
    for cl in clusters:
        for x, y in zip(cl.points_x, cl.points_y):
            r.append(dist(cl.centroids_x, cl.centroids_y, x, y))
    return r.sum()



if __name__ == "__main__":
    n = 10  # кол-во тчк
    k = 3  # кол-во кластеров
    filename = 'dataset.csv'
    generate_points_file(n, filename)
    points = read_csv(filename)
    centroids = centroids(points, k)
    # centroids = [x_c, y_c]
    clusters = nearest_centroid(points, centroids)
    show_picture(clusters)
    new_clusters = []
    old_clusters = []
    while centroid_not_equals(old_clusters, new_clusters):
        old_clusters = nearest_centroid(points, centroids)
        new_clusters = recalculate_centroid(old_clusters)
        centroids = map_to_array(new_clusters)
        show_picture(new_clusters)
        for cl in new_clusters:
            cl.clear_points()

