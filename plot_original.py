
import pandas as pd
import matplotlib.pyplot as plt
import os
def visualize_csv(csv_file):

    df = pd.read_csv(csv_file, header=None, names=['x', 'y', 'label'])

    x_values = df['x']
    y_values = df['y']

    plt.scatter(x_values, y_values, s=1)
    plt.xlabel('X')
    plt.ylabel('Y')
    '''plt.xlim(0.5, 0.6)
    plt.ylim(0.5, 0.6)'''
    plt.show()

if __name__ == "__main__":
    base_path = '/Users/linus/Desktop/data/'

    # 这里手动列举了不规则的部分
    variants = [
        '0.1S', '0.2S', '0.3S', '0.4S', '0.5S', '1.0S', '1.5S', '2.0S', '2.5S', '3.0S'
    ]

    file_names = [f"RN_{variant}_100K_50P.csv" for variant in variants]
    data_paths = [os.path.join(base_path, file_name) for file_name in file_names]
    data_paths = ['/Users/shirley/Desktop/data/labeled/external/Aggregation.csv']
    for data_path in data_paths:
        visualize_csv(data_path)


