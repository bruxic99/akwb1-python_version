import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class akwb1:
    boolean = {'adjust': False, 'linear': False, 'multigraph': False}
    biggest = 0
    dictionary = {}
    matrix = None
    original_graph = []

    def __init__(self):
        """
        Constructor
        Reset all parameters
        """
        self.boolean = {'adjust': False, 'linear': False, 'multigraph': False}
        self.biggest = 0
        self.dictionary.clear()
        self.matrix = np.delete
        self.original_graph = []

    def run(self, file_name):
        """
        Main method in which runs all process
        :param file_name: Name of file which will be read
        """
        self.load_file(file_name)
        self.draw_graph(file_name, True)
        if self.boolean['multigraph']:
            print("Graph is multigraph")
            return
        self.adjoint_check(self.matrix, self.biggest)
        self.info()
        if self.boolean['adjust']:
            self.convert_to_original(self.biggest, self.matrix)
            self.draw_graph(file_name, False)
            self.save(file_name, self.original_graph)

    def adjoint_check(self, matrix, biggest):
        """
        Check if graph is adjust
        :param matrix: neighborhood matrix
        :param biggest: last vertex
        """
        for a in range(0, biggest):
            for b in range(a + 1, biggest):
                for c in range(0, biggest):
                    if matrix[a][c] != 0 and matrix[b][c] != 0:
                        for d in range(0, biggest):
                            if (matrix[a][d] == 0 and matrix[b][d] != 0) or (matrix[a][d] != 0 and matrix[b][d] == 0):
                                return
        self.boolean['adjust'] = True
        self.linear_check(matrix, biggest)

    def linear_check(self, matrix, biggest):
        """
        Check if graph is linear
        :param matrix: neighborhood matrix
        :param biggest: last vertex
        """
        for a in range(0, biggest):
            for b in range(a + 1, biggest):
                for c in range(0, biggest):
                    if matrix[a][c] != 0 and matrix[b][c] != 0:
                        for d in range(0, biggest):
                            if matrix[d][a] != 0 and matrix[d][b] != 0:
                                return
        self.boolean['linear'] = True

    def load_file(self, file_name):
        """
        Load graph from file (list of successors)
        :param file_name: Name of file
        """
        f = open("./samples/" + file_name)
        for line in f.readlines():
            line = line.split()
            self.dictionary[int(line[0])] = [int(i) for i in line[1:] if i.isdigit()]
            self.biggest = int(line[0]) if int(line[0]) > self.biggest else self.biggest
        self.biggest += 1
        self.create_matrix(self.biggest, self.dictionary)
        f.close()

    def create_matrix(self, biggest, dictionary):
        """
        Create neighborhood matrix based on dictionary
        :param biggest: last vertex
        :param dictionary: list of successors in dictionary
        :return:
        """
        temp_matrix = np.zeros((biggest, biggest))
        for key in dictionary:
            for value in dictionary[key]:
                if temp_matrix[key][value] == 1:
                    self.boolean['multigraph'] = True
                temp_matrix[key][value] += 1
        self.matrix = temp_matrix

    def info(self):
        """
        Print info about read graph
        """
        for key in self.boolean.keys():
            if self.boolean[key] and not str(key).startswith('m'):
                print(f"Graph is {key}")
            elif not self.boolean[key] and not str(key).startswith('m'):
                print(f"Graph is not {key}")

    def convert_to_original(self, biggest, matrix):
        """
        If graph is adjusted, convert it to original graph (H)
        :param matrix: neighborhood matrix
        :param biggest: last vertex
        """
        edge_counter = 0
        original_graph = [[-1, -1] for i in range(biggest)]
        for i in range(biggest):
            if original_graph[i][0] == -1:
                original_graph[i][0] = edge_counter
                edge_counter += 1
            if original_graph[i][1] == -1:
                original_graph[i][1] = edge_counter
                edge_counter += 1
            for j in range(biggest):
                if matrix[j][i] == 1:
                    original_graph[j][1] = original_graph[i][0]
            for k in range(biggest):
                if matrix[i][k] == 1:
                    original_graph[k][0] = original_graph[i][1]
        self.original_graph = sorted(original_graph, key=lambda x: x[0])

    def draw_graph(self, file_name, option):
        """
        Create circular graph (original or read graph) and save it in figures folder
        :param file_name:
        :param option:
        :return:
        """
        G = nx.DiGraph()
        if option:
            self.get_edges_from_matrix(G, self.matrix)
            save_name = './figures/read/' + file_name.split('.')[0] + '-fig.png'
        else:
            for edge in self.original_graph:
                G.add_edge(edge[0], edge[1])
            save_name = './figures/original/' + file_name.split('.')[0] + '-original-fig.png'
        pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos, connectionstyle="arc3,rad=0.1")
        plt.savefig(save_name)
        plt.close()

    @staticmethod
    def save(file_name, original_graph):
        """
        Save original graph to file (.txt)
        :param file_name: Name of read file
        :param original_graph: original graph (H)
        """
        dictionary = {}
        for item in original_graph:
            if item[0] not in dictionary.keys():
                dictionary[item[0]] = item[1:]
            else:
                dictionary[item[0]].append(item[1])
        f = open("./results/" + "-".join([file_name.split('.')[0], "wynik.txt"]), "w")
        for key in dictionary:
            s = " ".join(map(str, dictionary[key]))
            f.write(f"{key} {s}\n")
        f.close()

    @staticmethod
    def get_edges_from_matrix(G, matrix):
        """
        Add edges to graph (G) based on neighborhood matrix
        :param G: Graph
        :param matrix: neighborhood matrix
        """
        dims = matrix.shape
        for i in range(dims[0]):
            for j in range(dims[1]):
                if matrix[i][j] == 1:
                    G.add_edge(i, j)
                elif matrix[i][j] >= 2:
                    for k in range(int(matrix[i][j])):
                        G.add_edge(i, j)


def console():
    """
    Function to read file by given input
    """
    print("Enter file name (1-6):")
    x = input()
    akb = akwb2()
    akb.run(x + ".txt")


def automatic():
    """
    Function to read all graphs (1-6)
    """
    test = list(range(1, 7))
    for i in test:
        akb = akwb2()
        akb.run(str(i) + ".txt")


if __name__ == '__main__':
    #console()
    automatic()
