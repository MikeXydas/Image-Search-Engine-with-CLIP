import json

import elasticsearch.client
import logging
import glob

from elasticsearch import Elasticsearch

from backend.clip_processing import StoredImage, ClipWrapper


class ElasticSearchImageController:
    def __init__(self, host: str, port: int, index_name: str, init_index: bool):
        self.index_name = index_name
        self.es_conn = self.init_connection(host, port)
        if init_index:
            self.create_mapping()

    def store_image(self, image: StoredImage) -> None:
        data = {
            "path": image.path,
            "embedding": image.embedding
        }
        try:
            outcome = self.es_conn.index(index=self.index_name,
                                         doc_type='_doc', body=json.dumps(data))
            logging.info(outcome)
        except Exception as ex:
            logging.error('Error in indexing data')
            logging.error(str(ex))

    def retrieve_image(self, text_embedding):
        s_body = {
            "_source": ["path"],
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": text_embedding}
                    }
                }
            }
        }
        results = self.es_conn.search(index=self.index_name, body=s_body)

        return results

    @staticmethod
    def init_connection(host, port) -> elasticsearch.client.Elasticsearch:
        es_conn = Elasticsearch([{'host': host, 'port': port}])
        if es_conn.ping():
            logging.info("Connected to Elasticsearch.")
        else:
            logging.error("Failed to connect to Elasticsearch.")
        return es_conn

    def create_mapping(self) -> None:
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "path": {
                        "type": "text"
                    },
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 512
                    }
                }
            }
        }

        response = self.es_conn.indices.create(
            index=self.index_name,
            body=mapping,
            ignore=400  # ignore 400 already exists code
        )

        if 'acknowledged' in response:
            if response['acknowledged']:
                logging.info(f"INDEX MAPPING SUCCESS FOR INDEX: {response['index']}")
        elif 'error' in response:
            if response['error']['type'] == "resource_already_exists_exception":
                logging.warning("Index already existed.")
            else:
                logging.error(f"ERROR: {response['error']['root_cause']}")
                logging.error(f"TYPE: {response['error']['type']}")


def elastic_search_initializer(es_controller: ElasticSearchImageController, directory: str, clip):
    image_paths = glob.glob(f"{directory}*")

    for img_path in image_paths:
        img_info = clip.create_image_embedding(img_path)
        es_controller.store_image(img_info)
