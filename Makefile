COMMIT = $(shell git rev-parse HEAD)
VERSION ?= $(shell bash -o pipefail -c "$(GIT) show-ref --tags -d 2> /dev/null | grep $(COMMIT) | sed -e 's,.* refs/tags/,,' -e 's/\^{}//' | sed -n 1p"  || echo "$(COMMIT)")

ROOT := hamgit.ir/aliabdollahi024a/stockprediction
DOCKER_REGISTRY = registry.hamdocker.ir/mlsd-indicatorss
IMAGE_TAG := $(VERSION)
IMAGE_NAME := $(DOCKER_REGISTRY)/stockprediction
IMAGE_NAME_TAG := $(IMAGE_NAME):$(IMAGE_TAG)

docker-build:
	docker build \
    --file Dockerfile \
    --tag "$(IMAGE_NAME_TAG)"\
    .

	docker tag $(IMAGE_NAME_TAG) $(IMAGE_NAME):latest

docker-push:
	docker push $(IMAGE_NAME_TAG)
	docker push $(IMAGE_NAME):latest

