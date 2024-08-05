import sys
import torch

from pathlib import PureWindowsPath
from transformers import AutoTokenizer, AutoModel

import json
import glob
import pathlib
import os
import logging
import pickle
import io

import pandas as pd
import numpy as np

from base64 import b64encode
from tqdm import tqdm

# Converts .json graph file into a set of .csv files

# data/
# |-- ProcessNode.csv        # process nodes
# |-- FileNode.csv           # file nodes
# |-- SocketChannelNode.csv  # ip nodes
# |-- RegistryNode.csv       # TODO
# |-- ProcessNode_PROC_CREATE_ProcessNode.csv  # Process creates another process
# |-- ProcessNode_READ_WRITE_FileNode.csv  # Process writes to a file
# |-- ...etc

# Supress warinings from Pandas complaining about the usage of df.append
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

#  VERTEX FIELDS
EXCLUDED_FIELDS_VER = [
    "_id",
    "_type",
    "SEND_SCORE",
    "RECV_SCORE",
    "PROP_VAL",
    "BT_HOPCOUNT",
    "FT_HOPCOUNT",
    "REF_DB",
]

# These vertex fields are currently excluded. This variable should not be used anywhere in code.
STR_FIELDS_VER_SPECIFIC_EXCLUDED = {
    "ProcessNode": [
        "OWNER_GROUP_ID",
        "PROC_OWNER_UID",
        "PROC_OWNER_GROUP_ID",
        "OWNER_UID",
    ],
    "FileNode": [
        "FILE_OWNER_UID",
        "FILE_OWNER_GROUP_ID",
        "RENAME_SET",
    ],
    "SocketChannelNode": [],
    "VirtualNode": [],
}

# These vertex fields are included
INT_FIELDS_COMMON_VER = ["AGENT_ID", "TYPE", "REF_ID"]

INT_FIELDS_VER_SPECIFIC = {
    "ProcessNode": ["PROC_ORDINAL", "PID", "PROC_STARTTIME"],
    "SocketChannelNode": ["IS_INCOMING", "CONN_TYPE", "CHANNEL_STATE", "CHANNEL_TYPE"],
    "FileNode": [],
    "VirtualNode": [],
}

# String fields to convert
STR_FIELDS_VER_SPECIFIC = {
    "ProcessNode": ["EXE_NAME", "CMD"],
    "SocketChannelNode": [
        "LOCAL_INET_ADDR",
        "LOCAL_PORT",
        "REMOTE_PORT",
        "REMOTE_INET_ADDR",
    ],
    "FileNode": ["FILENAME_SET", "VOL_ID", "DATA_ID"],
    "VirtualNode": [],
}

# Of the subset of string fields to convert, which of these fields require preprocessing in terms of being a path
# string?
PATH_FIELDS_VER_SPECIFIC = {
    "ProcessNode": ["EXE_NAME", "CMD"],
    "SocketChannelNode": [],
    "FileNode": ["FILENAME_SET"],
    "VirtualNode": [],
}

# EDGE FIELDS
EXCLUDED_FIELDS_EDGE = [
    "_id",
    "_outV",
    "_inV",
    "_type",
    "BT_SOURCE",
    "FT_SOURCE",
    "TOTAL_FREQ_VAL",
    "FREQ_VAL",
    "TRANSITION_VAL",
]

# These string fields are currently excluded. This variable should not be used anywhere in code.
STR_FIELDS_COMMON_EDGE = [
    "TOTAL_FREQ_KEY",
    "FREQ_KEY",
    "ALERT_INFO",
    "PATH_NAME",
    "IS_RANKED",
]

INT_FIELDS_COMMON_EDGE = [
    "EVENT_START",
    "EVENT_END",
    "TIME_START",
    "TIME_END",
    "EVENT_START_STR",
    "EVENT_END_STR",
    "_label",
    "IS_ALERT",
]

INT_FIELDS_EDGE_SPECIFIC = {
    "IP_CONNECTION_EDGE": [
        "REMOTE_PORT",
        "LOCAL_PORT",
        "CHANNEL_TYPE",
        "REL_TIME_START",
        "REL_TIME_END",
    ],
    "READ_WRITE": ["ACCESS_AMOUNT", "OPTYPE"],
    "PROC_CREATE": ["OPTYPE", "REL_TIME_START", "REL_TIME_END"],
    "FILE_EXEC": ["OPTYPE", "REL_TIME_START", "REL_TIME_END"],
    "READ": ["ACCESS_AMOUNT", "OPTYPE"],
    "WRITE": ["ACCESS_AMOUNT", "OPTYPE"],
    "PROC_END": ["OPTYPE", "TIME_START", "TIME_END"],
}

NUMERICAL_MAPPING = {
    "TYPE": ["ProcessNode", "SocketChannelNode", "FileNode", "VirtualNode"],
    "OPTYPE": [
        "Start_Processlet",
        "Close_IP_Connection_Descriptor",
        "Transfer_IP_Data",
        "Access_File",
        "ACCESS",
        "Open_IP_Connection_Descriptor",
        "Close_File_Descriptor",
        "Open_File_Descriptor",
        "End_IP_Connection",
        "Start_IP_Connection",
        "Execute_Processlet_Library",
        "End_Processlet",
        "FILE_EXEC",
    ],
    "_label": [
        "PROC_CREATE",
        "READ_WRITE",
        "IP_CONNECTION_EDGE",
        "FILE_EXEC",
        "READ",
        "WRITE",
        "PROC_END",
    ],
    "CHANNEL_STATE": ["SOCK_ESTABLISHED"],
    "CHANNEL_TYPE": ["INET_CHANNEL", "6"],
}

bert_tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
bert_model = AutoModel.from_pretrained("microsoft/codebert-base")


def embed_string(node_type, attribute, input_str):
    if input_str == "":
        input_str = "null"

    if attribute in PATH_FIELDS_VER_SPECIFIC[node_type]:
        if len(input_str) == 1 or (
            len(input_str) <= 3 and input_str[-1] in ["/" or "'"]
        ):
            input_str = "root"  # account for only "/" or "C:/" edge case
        else:
            input_str = PureWindowsPath(
                input_str
            ).name  # get last element of path to embed

    if not input_str:
        input_str = "None"

    tokens = bert_tokenizer.tokenize(input_str)
    ids = bert_tokenizer.convert_tokens_to_ids(tokens)

    embedding = bert_model(torch.tensor(ids)[None, :])[0][0]

    return (
        embedding.sum(0).detach().numpy()
    )  # sum all rows together and return embedding


# map nodes into node_type => [list of nodes] format
def parseNodesFromJSON(jsonObject):
    nodes = dict()

    for item in jsonObject["vertices"]:
        item_type = item["TYPE"]["value"]
        if item_type not in nodes:
            nodes[item_type] = [item]
        else:
            nodes[item_type].append(item)

    count = 0
    for node_list in nodes.values():
        count += len(node_list)

    print(f"Parsed {count} nodes from input json")

    return nodes


# transforms node map to .csv files
def toNodeCSV(node_map, outputDir):
    for node_type, node_list in node_map.items():
        # map data into a good csv format
        mapped_node_list = []
        mapped_node_list_extra_attributes = []

        for n in node_list:
            mapped_node, extra_attributes = mapSingleNodeToCSVFormat(n)
            mapped_node_list.append(mapped_node)
            mapped_node_list_extra_attributes.append(extra_attributes)

        print(f"Mapped {len(mapped_node_list)} {node_type} nodes to csv format")

        csv_format = (
            INT_FIELDS_COMMON_VER
            + INT_FIELDS_VER_SPECIFIC[node_type]
            + STR_FIELDS_VER_SPECIFIC[node_type]
        )

        # save into csv
        df = pd.DataFrame.from_records(mapped_node_list, columns=csv_format)

        output_csv_file_path = os.path.join(outputDir, node_type + ".csv")
        df.to_csv(output_csv_file_path)

        print(f"Saved {df.shape[0]} nodes to {output_csv_file_path}")

        # save into pickle
        output_pickle_file_path = os.path.join(outputDir, node_type + ".pickle")

        with open(output_pickle_file_path, "wb") as f:
            pickle.dump(mapped_node_list_extra_attributes, f)

        print(
            f"Saved {len(mapped_node_list_extra_attributes)} nodes to {output_pickle_file_path}"
        )


string_embedding_cache = {}


# takes in a node and returns mapped csv value of the node
def mapSingleNodeToCSVFormat(node):
    node_type = node["TYPE"]["value"]
    return_node = dict()  # this will contain embedded strings

    # raw string attributes we which to save about a node
    return_node_extra_string_attributes = dict()

    # apply numerical mapping first
    for feature in node.keys():
        if feature in NUMERICAL_MAPPING:
            return_node[feature] = NUMERICAL_MAPPING[feature].index(
                node[feature]["value"]
            )

    ### integer field mapping ###

    # common node integer features
    for common_feature in INT_FIELDS_COMMON_VER:
        if common_feature not in return_node:
            if node[common_feature]["value"] is None:
                continue
            return_node[common_feature] = int(node[common_feature]["value"])

    # node specific integer features
    assert node_type in INT_FIELDS_VER_SPECIFIC, (
        f"Error mapping node {node} to CSV format. Node type {node_type} "
        f"unrecognized for integer field mapping."
    )

    for specific_feat in INT_FIELDS_VER_SPECIFIC[node_type]:
        if specific_feat not in return_node and specific_feat in node:
            return_node[specific_feat] = int(node[specific_feat]["value"])

    ### string feature mapping ###

    assert node_type in STR_FIELDS_VER_SPECIFIC, (
        f"Error mapping node {node} to CSV format. Node type {node_type} "
        f"unrecognized for string field mapping."
    )

    for specific_feat in STR_FIELDS_VER_SPECIFIC[node_type]:
        assert (
            specific_feat not in return_node
        ), f"{specific_feat} is a categorical feature defined in NUMERICAL_MAPPING"

        try:
            if node[specific_feat]["type"] == "list":
                # use first value if type is of list by default
                extracted_str_field = node[specific_feat]["value"][0]["value"]
            else:
                extracted_str_field = node[specific_feat]["value"]

            preprocessed_str_field = preprocessStringField(extracted_str_field)

            return_node_extra_string_attributes[specific_feat] = preprocessed_str_field

            # embed string and serialize the output
            # cache_dir_path = os.path.join(os.getcwd(), 'cache')
            # cache_path = os.path.join(cache_dir_path,
            #                           'string_embedding_cache.pickle')
            # if os.path.exists(cache_path):
            #     with open(cache_path, 'rb') as cache:
            #         string_embedding_cache = pickle.load(cache)
            # else:
            #     string_embedding_cache = {}

            if preprocessed_str_field not in string_embedding_cache:
                string_embedding_cache[preprocessed_str_field] = embed_string(
                    node_type, specific_feat, preprocessed_str_field
                )
            # if not os.path.exists(cache_dir_path):
            #     os.makedirs(cache_dir_path)
            # with open(cache_path, 'wb') as cache:
            #     pickle.dump(string_embedding_cache, cache)

            string_embedding = string_embedding_cache[preprocessed_str_field]

            output_buff = io.BytesIO()
            np.save(output_buff, string_embedding)

            return_node[specific_feat] = b64encode(
                output_buff.getvalue()
            ).decode()  # decode() used to transfrom from
            # byte string output to string
        except KeyError as e:
            print(e)

    return return_node, return_node_extra_string_attributes


def preprocessStringField(string_field):
    return str(string_field).strip().lower()


def toEdgeCSV(jsonObject, node_map, output_dir):
    # node_id => (node_type, index in node_list) map
    edge_node_map = dict()

    for node_type, node_list in node_map.items():
        for index, node in enumerate(node_list):
            if node["_id"] in edge_node_map:
                print(f"Error in toEdgeCSV(). Duplicate node id found. Node {node}")
                exit(-1)

            edge_node_map[node["_id"]] = (node_type, index)

    # edge_relation => edges map
    edge_relation_map = dict()

    for edge in jsonObject["edges"]:
        out_vertex_type, out_vertex_id = edge_node_map[edge["_outV"]]
        in_vertex_type, in_vertex_id = edge_node_map[edge["_inV"]]

        edge_type = edge["_label"]

        if (
            edge_type in ["READ", "FILE_EXEC"] and in_vertex_type == "ProcessNode"
        ):  # undo flipping operation in ft/bt Java code for when ProcessNode is an in vertex
            edge["u"] = in_vertex_id
            edge["v"] = out_vertex_id
            relation = f"{in_vertex_type}~{edge_type}~{out_vertex_type}"
        else:
            edge["u"] = out_vertex_id
            edge["v"] = in_vertex_id
            relation = f"{out_vertex_type}~{edge_type}~{in_vertex_type}"

        if relation in edge_relation_map:
            edge_relation_map[relation].append(edge)
        else:
            edge_relation_map[relation] = [edge]

    odd_relations = {
        relation_name: edges
        for relation_name, edges in edge_relation_map.items()
        if relation_name.split("~")[0] != "ProcessNode"
    }
    if odd_relations:
        logging.info(f"Odd Relations Found: {odd_relations.keys()} in {output_dir}")

    # to csv
    for edge_relation, edge_list in edge_relation_map.items():
        # map data into csv format
        mapped_edge_list = [mapSingleEdgeToCSVFormat(edge) for edge in edge_list]
        edge_type = edge_list[0]["_label"]

        print(
            f"Mapped {len(mapped_edge_list)} edges with relation {edge_relation} to csv format"
        )

        csv_format = (
            ["u", "v"] + INT_FIELDS_COMMON_EDGE + INT_FIELDS_EDGE_SPECIFIC[edge_type]
        )

        df = pd.DataFrame.from_records(mapped_edge_list, columns=csv_format)

        output_file_path = os.path.join(output_dir, edge_relation + ".csv")

        df.to_csv(output_file_path)

        print(
            f"Saved {df.shape[0]} edges with relation {edge_relation} to {output_file_path}"
        )


def mapSingleEdgeToCSVFormat(edge):
    edge_type = edge["_label"]
    return_edge = dict()

    # add u and v
    return_edge["u"] = edge["u"]
    return_edge["v"] = edge["v"]

    # apply numerical mapping first
    for feature in edge.keys():
        if feature in NUMERICAL_MAPPING:
            if feature == "_label":
                return_edge[feature] = NUMERICAL_MAPPING[feature].index(edge[feature])
            else:
                return_edge[feature] = NUMERICAL_MAPPING[feature].index(
                    str(edge[feature]["value"])
                )

    # common edge features
    for common_feature in INT_FIELDS_COMMON_EDGE:
        if common_feature not in return_edge:
            return_edge[common_feature] = int(edge[common_feature]["value"])

    # edge specific features
    if edge_type not in INT_FIELDS_EDGE_SPECIFIC:
        print(
            f"Error mapping edge {edge} to CSV format. Edge type {edge_type} unrecognized."
        )
        exit(-1)

    for specific_feat in INT_FIELDS_EDGE_SPECIFIC[edge_type]:
        if specific_feat not in return_edge:
            return_edge[specific_feat] = int(edge[specific_feat]["value"])

    return return_edge


def mapJSONToCSV(input_json_file_name, output_dir):
    with open(input_json_file_name, "r", encoding="utf-8") as input_file_obj:
        jsonObject = json.load(input_file_obj)

    node_map = parseNodesFromJSON(jsonObject)

    toNodeCSV(node_map, output_dir)
    print()
    toEdgeCSV(jsonObject, node_map, output_dir)


logging.basicConfig(filename="odd_relations.log", level=logging.INFO)


def main():
    # FIXME(@kjee): this is not pretty. Let's use `argparse` module.
    if len(sys.argv) < 2:
        print(
            "Error. If you wish to convert one .json file into .csv, please use python3 jsonToCSV.py single <json_file> <output_dir>"
        )
        print(
            "If you wish to convert a directory of .json file into .csv, please use python3 jsonToCSV.py all <input dir>"
        )
        exit(-1)

    if sys.argv[1] == "single":
        if len(sys.argv) != 4:
            print("Format: python3 jsonToCsv.py single <json_file> <output_dir>")
            exit(-1)

        input_file_name = sys.argv[2]
        output_dir = sys.argv[3]

        mapJSONToCSV(input_file_name, output_dir)
    elif sys.argv[1] == "all":
        if len(sys.argv) != 3:
            print("Format: python3 jsonToCsv.py all <input_dir>")
            exit(-1)

        input_dir = sys.argv[2]

        print("Gathering files...")
        # grab path to every .json file contained under the input directory
        json_file_paths = list(
            glob.iglob(input_dir + "**/nd[- _]*.json", recursive=True)
        ) + list(glob.iglob(input_dir + "**/graph.json", recursive=True))

        # print('Checking finished files...')
        # import datetime
        # finished_files = []
        # for json_file in tqdm(json_file_paths):
        #     output_dir = str(pathlib.Path(json_file).parent.resolve())
        #     # Skip the files that's already converted. Change the date to when last time the script is executed.
        #     last_result_file = os.path.join(
        #         output_dir, 'ProcessNode~PROC_CREATE~ProcessNode.csv')
        #     if os.path.exists(last_result_file) and os.path.getmtime(
        #             last_result_file) > datetime.datetime(2022, 12,
        #                                                   5).timestamp():
        #         finished_files.append(json_file)

        # json_file_paths = [
        #     json_file for json_file in json_file_paths
        #     if json_file not in finished_files
        # ]
        # print('Skipped', len(finished_files), 'files.')

        for json_file in tqdm(json_file_paths):
            output_dir = str(
                pathlib.Path(json_file).parent.resolve()
            )  # get directory of file

            mapJSONToCSV(json_file, output_dir)
    else:
        # FIXME (@kjee): let's use logging module.
        print(
            "Error. If you wish to convert one .json file into .csv, please use python3 jsonToCSV.py single <json_file> <output_dir>"
        )
        print(
            "If you wish to convert a directory of .json file into .csv, please use python3 jsonToCSV.py all <input dir>"
        )
        exit(-1)

    print()
    print("Done!")


if __name__ == "__main__":
    main()
