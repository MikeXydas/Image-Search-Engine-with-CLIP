# Image Search Engine
*powered by [OpeanAI's CLIP](https://github.com/openai/CLIP)*

CLIP is a powerful pre-trained model able to calculate compatibility between an 
image, and a text prompt. We can easily see how useful this model can be for an image search
engine. 

So, I decided to create a simple app that takes as input a directory of images, calculates their
characteristics (embeddings), and stores them to an Elasticsearch index. Then when the user
enters the prompt "dog", we calculate the text embedding using CLIP again and perform a [cosine
similarity query](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-script-score-query.html#vector-functions) 
on our index.

Then the paths of the selected image are returned to the user.

## Video Demo

*Yep, I have not yet implemented a frontend app but it's definitely in my TODOs.*

https://user-images.githubusercontent.com/35707606/131347529-52d8f304-17cb-405b-9602-70467d14c5b2.mp4

## Setting up

**Setup the image directory**

We must point the app to the image directory we want to index.
1. Create a directory in the root of the repo named `storage`
2. Add any images you want to index and search over (`jpeg, jpg, png`)

Then there are two possible of installing the app:

**Option 1: Docker compose installation**

1. Make sure in `backend/configs/config.yaml` Elastic.host is set to `"elastic"`
2. Run `docker-compose up --build` in the directory root. (Takes ~5 minutes creating a 7GB image)

*Since, I have not implemented a frontend we will use it as a command line app.*

3. Run `docker ps` and note the container id of the clip image (NOT the elasticsearch id).
4. Run `docker exec -it CONTAINER_ID /bin/bash`
5. In the container run `python cmd.py`. You will have to for the model to be downloaded.

**Option 2: Local installation**

1. Install pytorch using conda. I suggest looking at the [installation procedure in the docs](https://pytorch.org/).
2. Install the remaining requirements by `pip install -r requirements.txt`
3. Make sure in `backend/configs/config.yaml` Elastic.host is set to `"localhost"`
4. Set the image directory path in `backend/configs/config.yaml`
4. Start the elasticsearch container:
   
`docker run -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.14.0`

5. Run `python backend/cmd.py`


## Issues / TODOs

* Cannot add an image in the Elasticsearch database without recalculating all of the image embeddings.
* The Docker image created is relatively big (7GB) since it includes the cuda toolikit. No cpu-only solution.
* No Elasticsearch persistence, which can be easily added through an elasticsearch data volume.
* A web-like frontend instead of the current command line utility.
* Currently, when initializing we calculate each image embedding sequentially. Should add batch encoding to speed up the process.
