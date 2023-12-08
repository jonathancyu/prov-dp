import multiprocessing
from pathlib import Path

from tqdm import tqdm

from algorithm import GraphWrapper, EdgeWrapper


def path_to_string(graph: GraphWrapper, path: list[EdgeWrapper]):
    source_edge = path[0]
    source_node = graph.get_node(source_edge.get_src_id())
    result = f'{graph.source_edge_id}: {source_node.get_type()}'
    for edge in path:
        node = graph.get_node(edge.get_dst_id())
        result += f' {edge.get_token()} {node.get_token()}'
    return result

def extract_paths(graph: GraphWrapper):
    paths = graph.get_paths()
    with open(f'F:/data/prov_dp/tc3-theia/{graph.source_edge_id}.txt', 'w') as file:
        for path in paths:
            file.write(f'{path_to_string(graph=graph, path=path)}\n')

if __name__ == '__main__':
    input_path = Path('F:\\data\\benign_graphs\\tc3-theia\\firefox\\nd')
    output_directory = Path('F:\\data\\data\\output\\tc3-theia\\firefox')
    input_paths = list(input_path.glob('*.json'))
    input_graphs = [GraphWrapper(input_path) 
                    for input_path in tqdm(input_paths, desc='Reading graphs')]
        
    num_processes = multiprocessing.cpu_count() * (3/4)
    print(f'Starting to process {len(input_graphs)} graphs')
    with multiprocessing.Pool(processes=num_processes) as pool:
        try:
            pool.map(extract_paths, tqdm(input_graphs, desc='Processing graphs'))
        except KeyboardInterrupt:
            print('Got SIGINT')
            pool.terminate()