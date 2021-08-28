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

    if elastic_settings['init']:
        elastic_search_initializer(es_controller, storage_settings['image_dir'], clip)

    return es_controller, clip


if __name__ == '__main__':
    init_backend()
