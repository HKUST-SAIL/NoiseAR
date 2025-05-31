.PHONY: install
install:
	# conda create --name noise_ar python=3.10.8 
	# conda activate noise_ar
	pip install -r requirements.txt

	pip install image-reward==1.5
	pip install hpsv2==1.2.0
	pip install timm==1.0.13
	
	# pip install git+https://github.com/openai/CLIP.git
	cd util_models/CLIP && \
	rm -r clip.egg-info 2>/dev/null || true && \
	pip install . && \
	cd ../

	# hpsv2 requires a specific version of open_clip
	cp util_models/hpsv2/bpe_simple_vocab_16e6.txt.gz $(CONDA_PREFIX)/lib/python3.10/site-packages/hpsv2/src/open_clip/