import logging

from backend.clip_processing import ClipWrapper
from init_elastic_search import ElasticSearchImageController
from init_elastic_search import elastic_search_initializer

import yaml


def init_backend():
    with open('configs/config.yaml') as file:
        settings = yaml.full_load(file)
    elastic_settings = settings['Elastic']
    storage_settings = settings['Storage']

    es_controller = ElasticSearchImageController(elastic_settings['host'], elastic_settings['port'],
                                                 elastic_settings['image_embeddings_index'], elastic_settings['init'])
    clip = ClipWrapper()

    # If the index is not populated then we should add the embeddings and paths of all the images
    # in the specified directory
    if not es_controller.is_index_populated():
        logging.info("Populating empty index...")
        elastic_search_initializer(es_controller, storage_settings['image_dir'], clip)
    else:
        logging.info("Index was already populated.")
    return es_controller, clip


if __name__ == '__main__':
    init_backend()
