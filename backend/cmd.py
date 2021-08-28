from typing import Tuple, List
from backend.app import init_backend

import webbrowser
import logging

logging.getLogger().setLevel(logging.WARNING)


class CommandLineEndpoint:
    def __init__(self):
        self.es_controller, self.clip = init_backend()

    def retrieve_top_k(self, query: str, k: int = 2) -> List[Tuple[str, float]]:
        query_embedding = self.clip.create_text_embedding(query)
        results = self.es_controller.retrieve_image(query_embedding)

        path_score = [(res["_source"]["path"], res["_score"]) for res in results["hits"]["hits"]]

        return path_score[:k]

    def input_loop(self):
        exited = False
        while not exited:
            print(">>> Provide input query or type EXIT")
            input_query = input("> Query: ")

            if input_query == "EXIT":
                exited = True
            else:
                results = self.retrieve_top_k(input_query, k=2)
                self.pretty_print_results(results)

    @staticmethod
    def pretty_print_results(results: List[Tuple[str, float]]) -> None:
        print()
        for ind, (path, score) in enumerate(results, start=1):
            print(f"\t{ind}. {path} with a score of {score}")
            webbrowser.open_new_tab(path)
        print()


if __name__ == '__main__':
    cmd = CommandLineEndpoint()
    cmd.input_loop()
