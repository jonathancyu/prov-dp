import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from karateclub import Graph2Vec

from source.algorithm import GraphWrapper
from source.algorithm.utility import tokenize, build_vocab, get_device, to_nx, print_stats


class GraphModel:
    # Model parameters
    device: torch.device
    context_length: int
    n_embedding: int
    n_hidden: int
    n_graph_embedding: int

    # Dataset info
    paths: list[str]
    graphs: list[GraphWrapper]
    stoi: dict[str, int]
    itos: list[str]

    base_model_path: Path

    # Stats
    __distances: list[dict[str,float]]

    def __init__(self,
                 paths: list[str],
                 graphs: list[GraphWrapper],
                 context_length: int = 8,
                 n_embedding: int = 10,
                 n_hidden: int = 100,
                 n_graph_embedding: int = 100,
                 batch_size: int = 2048,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 base_model_path: Path = Path('.'),
                 load_graphson: bool = False) -> None:
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

        # Build vocabulary
        assert len(paths) == len(graphs)
        self.paths = paths
        self.graphs = graphs
        self.graph_embeddings: np.ndarray = self.__get_graph_embeddings(load_graphson)
        self.itos, self.stoi = build_vocab(paths)
        self.vocab_size = len(self.stoi)

        # Initialize model
        self.device = get_device()
        self.__init_model()
        self.model.to(self.device)

        self.__distances = []

    def __init_model(self):
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.n_embedding),
            nn.Flatten(),
            nn.Linear(self.n_embedding * self.context_length, self.n_embedding * self.context_length), nn.ReLU(),
            nn.Linear(self.n_embedding * self.context_length, self.n_hidden), nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_graph_embedding)
        )

    def __get_graph_embeddings(self, load_graphson: bool) -> np.ndarray:
        pickle_path = self.base_model_path / 'graph_model.pkl'
        # Embed graphs using graphviz
        if load_graphson and pickle_path.exists():
            # Load model
            with open(pickle_path, 'rb') as file:
                graph2vec = pickle.load(file)
            print(f'  Loaded graph2vec from {pickle_path}')
        else:
            # Fit model
            nx_graphs = [to_nx(graph) for graph in self.graphs]
            graph2vec = Graph2Vec(
                wl_iterations=80,
                attributed=True,
                dimensions=self.n_graph_embedding,
                workers=4,
                epochs=5
            )
            print('  Fitting graph2vec')
            graph2vec.fit(nx_graphs)
            # Save model
            with open(pickle_path, 'wb') as file:
                pickle.dump(graph2vec, file)

            print(f'  Saved graph2vec to {pickle_path}')

        # This return a list of embeddings, one for each graph in self.graphs
        graph_embeddings = graph2vec.get_embedding()
        assert len(graph_embeddings) == len(self.graphs)
        return graph_embeddings

    def train(self, epochs: int):
        # Create dataset
        X, Y = self.__build_dataset(self.paths, self.graph_embeddings)

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
        model_path = self.base_model_path / 'graph_model.tar'
        torch.save(self.model.state_dict(), model_path)
        print(f'  Saved model to {model_path}')

    def __path_to_context(self, path: str) -> list[int]:
        # Convert a string path into a list of tokenized integers
        path_tokens = tokenize(path.split(' '))
        path = [self.stoi[token] for token in path_tokens]
        # Pad the context with 0s and truncate to context_length
        context = [0] * self.context_length
        for i in range(min(self.context_length, len(path))):
            context[i] = path[i]
        return context

    def __build_dataset(self,
                        paths: list[str],
                        graph_embeddings: np.ndarray) -> tuple[torch.tensor, torch.tensor]:
        assert len(paths) == len(graph_embeddings)
        X, Y = [], []
        for path, graph_embedding in zip(paths, graph_embeddings):
            context = self.__path_to_context(path)
            X.append(context)
            Y.append(graph_embedding)

        return torch.tensor(np.array(X), device=self.device), torch.tensor(np.array(Y), device=self.device)

    def predict(self, path: str) -> GraphWrapper:
        # Convert the path to a context tensor and predict a graph embedding
        with torch.no_grad():
            self.model.eval()
            context = self.__path_to_context(path)
            prediction_tensor = self.model(torch.tensor([context], device=self.device))
            prediction = prediction_tensor.cpu().data.numpy()

        # Compute a probability distribution based on the distance to the prediction
        distances = np.linalg.norm(self.graph_embeddings - prediction, axis=1)
        inverse_distances = 1 / distances  # Low distance => high probability
        probabilities = inverse_distances / np.sum(inverse_distances)

        # Sample based on the computed distribution
        choice = np.random.choice(len(probabilities), p=probabilities)
        self.__distances.append({
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        })
        return self.graphs[choice]

    def print_distance_stats(self) -> None:
        print(f'Per-graph embedding distance distributions:')
        for stat in ['Mean', 'Std', 'Min', 'Max']:
            print_stats(stat, [p[stat.lower()] for p in self.__distances])
