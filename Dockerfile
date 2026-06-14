FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /workspace

RUN python -m pip install --upgrade pip 

RUN python -m pip install --no-cache-dir \
	hydra-core \
	natsort \
	nibabel \
	numpy \
	pandas \
	scikit-image \
	scipy \
	tqdm

COPY . /workspace
RUN chmod +x /workspace/src/run_submission.sh

# Default challenge paths expected at runtime.
ENV INPUT_DIR=/input \
	OUTPUT_DIR=/output

ENTRYPOINT ["bash", "src/run_submission.sh"]
