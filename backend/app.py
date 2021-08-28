from init_elastic_search import ElasticSearchImageController
from init_elastic_search import elastic_search_initializer


def main():
    # Hard coded variables. Should be at a config yaml file.
    host = "localhost"
    port = 9200
    index_name = "image"
    img_dir = "/home/mikexydas/PycharmProjects/CLIP_Image_Search_Engine/storage/images/"
    init_elastic = False

    es_controller = ElasticSearchImageController(host, port, index_name, init_elastic)
    if init_elastic:
        elastic_search_initializer(es_controller, img_dir)


if __name__ == '__main__':
    main()