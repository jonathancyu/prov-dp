import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from karateclub import Graph2Vec

from source.algorithm import GraphWrapper
from source.algorithm.utility import tokenize, build_vocab, get_device, to_nx


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
                 base_model_path: Path = Path('.')):
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
        self.graph_embeddings = self._get_graph_embeddings()
        self.itos, self.stoi = build_vocab(paths)
        self.vocab_size = len(self.stoi)

        # Initialize model
        self.device = get_device()
        self._init_model()
        self.model.to(self.device)


    def _init_model(self):
        self.model = nn.Sequential(
            nn.Embedding(self.vocab_size, self.n_embedding),
            nn.Flatten(),
            nn.Linear(self.n_embedding * self.context_length, self.n_embedding * self.context_length), nn.ReLU(),
            nn.Linear(self.n_embedding * self.context_length, self.n_hidden), nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_graph_embedding)
        )

    def _get_graph_embeddings(self) -> list[np.array]:
        # Embed graphs using graphviz
        if True:  # not (self.base_model_path / 'graph2vec.pkl').exists():
            # Fit model
            nx_graphs = [to_nx(graph) for graph in self.graphs]
            graph2vec = Graph2Vec(
                wl_iterations=80,
                attributed=True,
                dimensions=self.n_graph_embedding,
                workers=4,
                epochs=5
            )
            print('Fitting graph2vec')
            graph2vec.fit(nx_graphs)
            # Save model
            with open(self.base_model_path / 'graph2vec.pkl', 'wb') as file:
                pickle.dump(graph2vec, file)
        else:
            # Load model
            with open(self.base_model_path / 'graph2vec.pkl', 'rb') as file:
                graph2vec = pickle.load(file)
        graph_embeddings = graph2vec.get_embedding()
        assert len(graph_embeddings) == len(self.graphs)
        return graph_embeddings

    def train(self, epochs: int):
        # Create dataset
        X, Y = self._build_dataset(self.paths, self.graph_embeddings)
        print(f'X: {X.shape}, Y: {Y.shape}')

        # Train model
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        with tqdm.tqdm(total=epochs) as bar:
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
                bar.update(1)

                # Log loss
                if epoch % 100 == 0:
                    bar.set_description(f'Loss: {loss.item():.8f}')
        model_path = self.base_model_path / 'graph_model.tar'
        torch.save(self.model.state_dict(), model_path)
        print(f'Saved model to {model_path}')

    def _path_to_context(self, path: str) -> list[int]:
        path_tokens = tokenize(path.split(' '))
        path = [self.stoi[token] for token in path_tokens]
        context = [0] * self.context_length
        for i in range(min(self.context_length, len(path))):
            context[i] = path[i]
        return context

    def _build_dataset(self,
                       paths: list[str],
                       graph_embeddings: list[np.array]) -> tuple[torch.tensor, torch.tensor]:
        X, Y = [], []
        for path, graph_embedding in zip(paths, graph_embeddings):
            context = self._path_to_context(path)
            X.append(context)
            Y.append(graph_embedding)

        return torch.tensor(X, device=self.device), torch.tensor(Y, device=self.device)

    def predict(self, path: str) -> GraphWrapper:
        with torch.no_grad():
            self.model.eval()
            context = self._path_to_context(path)
            prediction_tensor = self.model(torch.tensor([context], device=self.device))
            prediction = prediction_tensor.cpu().data.numpy()

        min_distance = float('inf')
        best_i: int = -1
        for i in range(len(self.graph_embeddings)):
            embedding = self.graph_embeddings[i]
            distance = np.linalg.norm(prediction - embedding, ord=1)
            if distance < min_distance:
                min_distance = distance
                best_i = i

        assert best_i >= 0
        print(f'Closest match: {best_i}')
        return self.graphs[best_i]
