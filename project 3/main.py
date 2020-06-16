from sklearn import decomposition, preprocessing
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    file = open("iris.data", "r+")
    all_data = list()
    for line in file:
        line = line.strip().split(",")[:-1]
        data = [float(x) for x in line]
        all_data.append(data)
    del all_data[-1]
    return all_data


def pre_processing(all_data):
    min_max_scalar = preprocessing.MinMaxScaler()
    all_data_minmax = min_max_scalar.fit_transform(all_data)
    return all_data_minmax


def principle_component(all_data, dimension):
    pca = decomposition.PCA(n_components=dimension)
    return pca.fit_transform(all_data)


def task1(all_data):
    cov_mat = np.cov(all_data.T)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    print("eigen values \n", eig_values)
    print("eigen vectors \n", eig_vectors)


def task2(all_data):
    data = principle_component(all_data, 3)
    x_data = [x[0] for x in data]
    y_data = [y[1] for y in data]
    z_data = [z[2] for z in data]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, color in enumerate(["red", "green", "blue"]):
        for i in range(50):
            ax.scatter(x_data[index * 50:(index + 1) * 50], y_data[index * 50:(index + 1) * 50],
                       z_data[index * 50:(index + 1) * 50], color=color)
    ax.set_xlabel("principle component 1")
    ax.set_ylabel("principle component 2")
    ax.set_zlabel("principle component 3")
    plt.show()


def task3(all_data):
    data1 = principle_component(all_data, 3)
    data2 = principle_component(data1, 2)
    data3 = principle_component(all_data, 2)
    x_data2 = [x[0] for x in data2]
    y_data2 = [y[1] for y in data2]
    x_data3 = [x[0] for x in data3]
    y_data3 = [y[1] for y in data3]

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.scatter(x_data2, y_data2, color="red")
    ax2.scatter(x_data3, y_data3, color="green")
    plt.show()


def main():
    all_data = get_data()
    all_data = pre_processing(all_data)
    func = {
        "1": task1,
        "2": task2,
        "3": task3
    }
    try:
        func[input("enter task:  ")](all_data)
    except Exception as err:
        print(err)


if __name__ == "__main__":
    main()
