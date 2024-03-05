import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from karateclub import Graph2Vec

from source.algorithm import Tree
from source.algorithm.utility import print_stats


class GraphModel:
    # Model parameters
    device: torch.device
    context_length: int
    n_embedding: int
    n_hidden: int
    n_graph_embedding: int

    # Checkpoint flags
    __load_graph2vec: bool
    __load_model: bool

    # Dataset info
    paths: list[str]
    graphs: list[Tree]
    stoi: dict[str, int]
    itos: list[str]

    base_model_path: Path

    # Stats
    __stats: list[dict[str, float]]

    def __init__(self,
                 paths: list[str],
                 graphs: list[Tree],
                 context_length: int = 8,
                 n_embedding: int = 10,
                 n_hidden: int = 100,
                 n_graph_embedding: int = 100,
                 batch_size: int = 2048,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 base_model_path: Path = Path('.'),
                 load_graph2vec: bool = False,
                 load_model: bool = False):

        self.base_model_path = base_model_path
        self.base_model_path.mkdir(exist_ok=True, parents=True)
        # Set model parameters
        self.context_length = context_length
        self.n_embedding = n_embedding
        self.n_hidden = n_hidden
        self.n_graph_embedding = n_graph_embedding
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size

        # Checkpoint flags
        self.__load_graph2vec = load_graph2vec
        self.__load_model = load_model

        # Build vocabulary
        assert len(paths) == len(graphs)
        self.paths = paths
        self.graphs = graphs
        self.graph_embeddings: np.ndarray = self.__get_graph_embeddings()
        self.itos, self.stoi = GraphModel.build_vocab(paths)
        self.vocab_size = len(self.stoi)
        # Initialize model
        self.device = GraphModel.get_device()
        self.__init_model()
        self.model.to(self.device)

        # Create dataset tensors
        assert len(paths) == len(self.graph_embeddings)
        self.__context_tensor = torch.tensor(
            [self.__path_to_context(path) for path in paths],
            device=self.device
        )
        self.__graph_embeddings = torch.tensor(
            self.graph_embeddings,
            device=self.device
        )

        self.__stats = []

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    @staticmethod
    def tokenize(path):
        assert len(path) % 2 == 0
        return [f'{path[idx]}|{path[idx + 1]}' for idx in range(0, len(path), 2)]

    @staticmethod
    def build_vocab(paths: list[str]) -> tuple[list[str], dict[str, int]]:
        """
        Builds the vocabulary for the model.
        :param paths: list of paths
        :return: integer to string, and string to integer mappings
        """
        token_set = set()
        distinct_paths = set()
        for path in paths:
            path = GraphModel.tokenize(path.split(' '))
            token_set.update(path)
            distinct_paths.add(' '.join(path))
        tokens = ['.'] + list(token_set)
        print(f'  Found {len(tokens)} tokens and {len(distinct_paths)} distinct paths in {len(paths)} entries')
        return tokens, {token: i for i, token in enumerate(tokens)}

    def __init_model(self):
        # Multi-layer perceptron
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.n_embedding),
            nn.Flatten(),
            nn.Linear(self.n_embedding * self.context_length, self.n_embedding * self.context_length), nn.ReLU(),
            nn.Linear(self.n_embedding * self.context_length, self.n_hidden), nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_graph_embedding)
        )

    def __get_graph_embeddings(self) -> np.ndarray:
        graph2vec_path = self.base_model_path / 'graph2vec.pkl'
        # Embed graphs using graphviz
        if self.__load_graph2vec and graph2vec_path.exists():
            # Load model
            with open(graph2vec_path, 'rb') as file:
                graph2vec = pickle.load(file)
            print(f'  Loaded graph2vec from {graph2vec_path}')
        else:
            # Fit model
            nx_graphs = [graph.to_nx() for graph in self.graphs]
            graph2vec = Graph2Vec(
                wl_iterations=80,
                attributed=True,
                dimensions=self.n_graph_embedding,
                workers=4,
                epochs=5
            )
            print('  Fitting graph2vec')
            start = time.time()
            graph2vec.fit(nx_graphs)
            print(f'  Fitted graph2vec in {time.time() - start:.4f} seconds')

            # Save model
            with open(graph2vec_path, 'wb') as file:
                pickle.dump(graph2vec, file)
            print(f'  Saved graph2vec to {graph2vec_path}')

        # This return a list of embeddings, one for each graph in self.graphs
        graph_embeddings = graph2vec.get_embedding()
        assert len(graph_embeddings) == len(self.graphs)
        return graph_embeddings

    def __train_model(self, epochs: int):
        # Input: path context tensor, output: graph embedding
        X, Y = self.__context_tensor, self.__graph_embeddings

        # Train model
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        loss_samples = []
        log_interval = epochs // 10
        for epoch in range(epochs):
            optimizer.zero_grad()

            # Sample batch
            ix = torch.randint(0, Y.shape[0], (self.batch_size,))
            X_batch, Y_batch = X[ix], Y[ix]

            # Forward pass
            output = self.model(X_batch)

            # Backward pass
            loss = F.mse_loss(output, Y_batch)
            loss.backward()
            optimizer.step()

            # Log rolling average loss
            loss_samples.append(loss.log().item())
            if (epoch+1) % log_interval == 0:
                num_digits = len(str(epochs))
                print(f'  ({epoch+1:{num_digits}d}/{epochs:{num_digits}d}) log(loss) = {np.average(loss_samples):.6f}')
                loss_samples = []

    def train(self, epochs: int):
        model_path = self.base_model_path / 'graph_model.tar'

        # Try to use trained model checkpoint
        if self.__load_model and model_path.exists():
            with open(model_path, 'rb') as file:
                self.model.load_state_dict(torch.load(file))
            print(f'  Loaded model from {model_path}')
        # Otherwise, train the model
        else:
            self.__train_model(epochs)
            # Save model parameters
            torch.save(self.model.state_dict(), model_path)
            print(f'  Saved model to {model_path}')

    def __path_to_context(self, path: str) -> list[int]:
        # Convert a string path into a list of tokenized integers
        path_tokens = GraphModel.tokenize(path.split(' '))
        path = [self.stoi[token] for token in path_tokens]
        context = [0] * self.context_length
        # Pad the context with 0s and truncate to context_length
        for i in range(min(self.context_length, len(path))):
            context[i] = path[i]
        return context

    @torch.no_grad()
    def predict(self, paths: list[str]) -> list[Tree]:
        # Convert the batch of paths a context tensor and predict the
        self.model.eval()
        batch = torch.tensor(
            [self.__path_to_context(path) for path in paths],
            device=self.device)
        predictions = self.model(batch)  # row x n_graph_embedding

        # Compute a probability distribution based on the distance to the prediction
        # batch_size x len(graph_embeddings) x n_graph_embedding
        # M[i,j,k] = prediction[i,k] - graph_embeddings[j,k]
        embedding_differences = self.__graph_embeddings.unsqueeze(0) - predictions.unsqueeze(1)

        # batch_size x len(graph_embeddings)
        # M[i,j] = distance[prediction_i,embedding_j]
        distances = torch.norm(embedding_differences, dim=2)

        # batch_size x len(graph_embeddings)
        # M[i, j] = P(graph_embeddings[j] | prediction_i)
        probabilities = 1 / distances  # high distance -> low probability

        # batch_size x 1
        # M[i] = index of sampled graph for prediction_i
        choice_tensor = torch.multinomial(probabilities, num_samples=1)
        choices = choice_tensor.cpu().numpy().flatten()

        self.__log_distance_stats(distances)

        # Map choice indices to graphs
        return [self.graphs[choice] for choice in choices]

    def __log_distance_stats(self, distances: torch.tensor) -> None:
        # Distance statistics
        mean_distance_tensor = torch.mean(distances, dim=1)
        std_distance_tensor = torch.std(distances, dim=1)
        min_distance_tensor, _ = torch.min(distances, dim=1)
        max_distance_tensor, _ = torch.max(distances, dim=1)
        for i in range(distances.shape[0]):
            self.__stats.append({
                'mean': mean_distance_tensor[i].item(),
                'std': std_distance_tensor[i].item(),
                'min': min_distance_tensor[i].item(),
                'max': max_distance_tensor[i].item()
            })

    def print_distance_stats(self) -> None:
        print(f'  Per-graph embedding distance distributions:')
        for stat in ['Mean', 'Std', 'Min', 'Max']:
            print('  ', end='')
            if len(stat) == 0:
                print(f'{stat} is empty')
                continue
            print_stats(stat, [p[stat.lower()] for p in self.__stats])
