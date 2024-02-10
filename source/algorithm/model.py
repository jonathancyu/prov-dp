import torch

from source.algorithm import GraphWrapper


class Model:
    context_length: int
    stoi: dict[str, int]
    itos: list[str]

    def __init__(self,
                 context_length: int):
        self.context_length = context_length

    def train(self,
              paths: list[str],
              graphs: list[GraphWrapper]):
        assert len(paths) == len(graphs)
        self.stoi, self.itos = Model.build_vocab(paths)


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
            path = Model.tokenize(path)
            token_set.update(path)
            distinct_paths.add(' '.join(path))
        tokens = ['.'] + list(token_set)
        print(f'Found {len(tokens)} tokens and {len(distinct_paths)} distinct paths in {len(paths)} entries')
        return tokens, {token: i for i, token in enumerate(tokens)}

    def path_to_context(self, path: str) -> list[int]:
        path_tokens = Model.tokenize(path.split(' '))
        path = [self.stoi[token] for token in path_tokens]
        context = [0] * self.context_length
        for i in range(min(self.context_length, len(path))):
            context[i] = path[i]
        return context

    def build_dataset(self, paths: list[str], graph_embeddings: list[GraphWrapper]) -> tuple[torch.tensor, torch.tensor]:


    @torch.no_grad()
    def forward(self, x):
        raise NotImplementedError